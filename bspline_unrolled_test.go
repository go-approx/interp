package interp

import (
	"math"
	"testing"
)

// evalGeneric is EvalNu's general loop for nu = 0 (the reference for
// evalUnrolled).
func evalGeneric(b *BSpline, dst, xs []float64) {
	k := b.K
	n := b.N()
	h := make([]float64, k+1)
	tb, te := b.T[k], b.T[n]
	hint := k
	for i, x := range xs {
		if math.IsNaN(x) || (b.Extrap == NaNOutside && (x < tb || x > te)) {
			dst[i] = math.NaN()
			continue
		}
		ell := findKnotInterval(b.T, k, x, hint)
		hint = ell
		basis(b.T, x, k, ell, 0, h)
		s := 0.0
		for a := 0; a <= k; a++ {
			s += float64(b.C[ell+a-k] * h[a])
		}
		dst[i] = s
	}
}

// TestEvalUnrolledMatchesGeneric: degrees 3 and 5, repeated interior knots
// (zero-width intervals), zero coefficients (signed zeros), queries on the
// knots, outside, NaN, sorted and shuffled, every extrapolation mode but
// Wrap — bit for bit.
func TestEvalUnrolledMatchesGeneric(t *testing.T) {
	r := newRNG(12)
	for trial := 0; trial < 400; trial++ {
		k := 3
		if trial%2 == 1 {
			k = 5
		}
		ni := r.IntN(12)
		inner := sortedX(r, ni, 0, 10, false)
		if ni > 2 && trial%3 == 0 {
			inner[1] = inner[0] // a double knot
			if ni > 4 && trial%9 == 0 {
				inner[3], inner[4] = inner[2], inner[2] // a knot of multiplicity 3
			}
		}
		knots := []float64{}
		for i := 0; i <= k; i++ {
			knots = append(knots, -1)
		}
		knots = append(knots, inner...)
		for i := 0; i <= k; i++ {
			knots = append(knots, 11)
		}
		nc := len(knots) - k - 1
		c := make([]float64, nc)
		for i := range c {
			switch r.IntN(6) {
			case 0:
				c[i] = 0
			case 1:
				c[i] = math.Copysign(0, -1)
			default:
				c[i] = r.Float64()*4 - 2
			}
		}
		b, err := NewBSpline(knots, c, k)
		if err != nil {
			t.Fatal(err)
		}
		if trial%4 == 2 {
			b.Extrap = NaNOutside
		}
		m := 1 + r.IntN(150)
		xs := make([]float64, m)
		for i := range xs {
			switch r.IntN(8) {
			case 0:
				xs[i] = knots[r.IntN(len(knots))]
			case 1:
				xs[i] = r.Float64()*20 - 5
			case 2:
				xs[i] = math.NaN()
			default:
				xs[i] = r.Float64()*12 - 1
			}
		}
		if trial%2 == 0 {
			sortFloats(xs)
		}
		got, want := make([]float64, m), make([]float64, m)
		b.EvalAll(got, xs)
		evalGeneric(b, want, xs)
		for i := range got {
			if !sameBits(got[i], want[i]) {
				t.Fatalf("trial %d k=%d x=%v: unrolled %v (%x), generic %v (%x)", trial, k, xs[i], got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
			}
		}
	}
}

func TestDeBoorStep(t *testing.T) {
	l, r := deBoorStep(1, 2, 2, 1.5)
	if !math.Signbit(l) || l != 0 || r != 0 || math.Signbit(r) {
		t.Errorf("zero width: %v %v", l, r)
	}
	if v := math.Copysign(0, -1) + l; !math.Signbit(v) {
		t.Errorf("-0 + l must stay -0")
	}
	if l, r := deBoorStep(1, 3, 1, 2); l != 0.5 || r != 0.5 {
		t.Errorf("split: %v %v", l, r)
	}
}
