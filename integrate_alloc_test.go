package interp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// integrateViaAntiderivative is Integrate as it was written before antiAt: it
// builds the antiderivative and evaluates it at the two bounds.
func integrateViaAntiderivative(b *BSpline, lo, hi float64) float64 {
	sign := 1.0
	if hi < lo {
		lo, hi = hi, lo
		sign = -1
	}
	k, n := b.K, b.N()
	tb, te := b.T[k], b.T[n]
	anti := b.Antiderivative(1)
	diff := func(a, c float64, extrap Extrapolation) float64 {
		var out [2]float64
		anti.Extrap = extrap
		anti.EvalNu(out[:], []float64{a, c}, 0)
		return out[1] - out[0]
	}
	switch b.Extrap {
	case NaNOutside:
		lo, hi = math.Max(lo, tb), math.Min(hi, te)
		if hi < lo {
			return 0
		}
		return sign * diff(lo, hi, Extend)
	case Wrap:
		period := te - tb
		interval := hi - lo
		nper := math.Floor(interval / period)
		left := interval - nper*period
		tot := 0.0
		if nper > 0 {
			tot = nper * diff(tb, te, Extend)
		}
		lo = tb + floorMod(lo-tb, period)
		hi = lo + left
		if hi <= te {
			tot += diff(lo, hi, Extend)
		} else {
			tot += diff(lo, te, Extend)
			tot += diff(tb, tb+hi-te, Extend)
		}
		return sign * tot
	}
	return sign * diff(lo, hi, Extend)
}

func bitsEqual(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (math.IsNaN(a) && math.IsNaN(b))
}

// TestIntegrateBitIdentical: the allocation-free Integrate returns the bits of
// the antiderivative route on random splines of degree 0 to 9 (9 takes the
// route that builds the antiderivative), with repeated interior and end knots,
// every extrapolation mode, and bounds inside, outside, swapped and NaN.
func TestIntegrateBitIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	for iter := 0; iter < 3000; iter++ {
		k := r.IntN(10)
		n := k + 1 + r.IntN(12)
		knots := make([]float64, n+k+1)
		v := r.Float64()
		for i := range knots {
			if r.IntN(4) != 0 || i == 0 { // a quarter of the knots repeat the previous one
				v += r.Float64() + 0.01
			}
			knots[i] = v
		}
		if iter%2 == 0 { // clamped ends
			for i := 0; i <= k; i++ {
				knots[i] = knots[0]
				knots[len(knots)-1-i] = knots[len(knots)-1]
			}
		}
		c := make([]float64, n)
		for i := range c {
			c[i] = r.NormFloat64()
		}
		b, err := NewBSpline(knots, c, k)
		if err != nil {
			t.Fatal(err)
		}
		tb, te := knots[k], knots[n]
		if !(tb < te) {
			continue
		}
		b.Extrap = Extrapolation(iter % 3)
		span := te - tb
		pick := func() float64 {
			switch r.IntN(8) {
			case 0:
				return tb
			case 1:
				return te
			case 2:
				return tb - span*r.Float64()*2
			case 3:
				return te + span*r.Float64()*2
			case 4:
				return knots[r.IntN(len(knots))]
			}
			return tb + span*r.Float64()
		}
		for j := 0; j < 8; j++ {
			lo, hi := pick(), pick()
			got, want := b.Integrate(lo, hi), integrateViaAntiderivative(b, lo, hi)
			if !bitsEqual(got, want) {
				t.Fatalf("k=%d n=%d extrap=%d [%v, %v]: %v, want %v", k, n, b.Extrap, lo, hi, got, want)
			}
		}
	}
	b, _ := NewBSpline([]float64{0, 0, 0, 0, 1, 1, 1, 1}, []float64{1, 2, 3, 4}, 3)
	if got, want := b.Integrate(math.NaN(), 1), integrateViaAntiderivative(b, math.NaN(), 1); !bitsEqual(got, want) {
		t.Fatalf("NaN bound: %v, want %v", got, want)
	}
}

// TestIntegrateAllocs: Integrate allocates nothing for the degrees it handles on
// stack buffers (it used to build the whole antiderivative: 3 allocations and
// two arrays of the size of the spline).
func TestIntegrateAllocs(t *testing.T) {
	const m = 2000
	knots := make([]float64, 0, m+4)
	knots = append(knots, 0, 0, 0, 0)
	for i := 1; i < m-3; i++ {
		knots = append(knots, float64(i))
	}
	knots = append(knots, m-3, m-3, m-3, m-3)
	c := make([]float64, m)
	for i := range c {
		c[i] = math.Sin(float64(i))
	}
	b, err := NewBSpline(knots, c, 3)
	if err != nil {
		t.Fatal(err)
	}
	var sink float64
	if a := testing.AllocsPerRun(100, func() { sink += b.Integrate(10.5, 1500.25) }); a != 0 {
		t.Fatalf("Integrate: %v allocations per call, want 0", a)
	}
	_ = sink
}
