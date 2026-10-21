package interp

import (
	"math"
	"testing"
)

// TestSplrepLegacyFixture checks Splrep (the fpcurf port) against
// scipy.interpolate.splrep: the knot vectors must be identical and the
// coefficients and values agree to rounding.
func TestSplrepLegacyFixture(t *testing.T) {
	fx := loadFixture(t, "splrep")
	var st Stats
	fits, rejected := 0, 0
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		k := int(c.n("k"))
		var o *SplrepOptions
		if w := c.v("w"); w != nil {
			o = &SplrepOptions{W: w}
		}
		sm, err := Splrep(x, y, k, c.n("s"), o)
		if c.Err != "" || c.o("lok")[0] == 0 {
			if err == nil {
				t.Errorf("%s: splrep raised, Go accepted", c.Name)
			}
			rejected++
			continue
		}
		if err == ErrNonFinite { // deliberate: splrep fits NaN data into NaN
			rejected++
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		fits++
		b := sm.Spline
		lt := c.o("lt")
		if len(b.T) != int(c.o("lnt")[0]) || ulpAll(sampleIf(c, b.T), lt) != 0 {
			t.Errorf("%s: knots differ from splrep: %d vs %d", c.Name, len(b.T), int(c.o("lnt")[0]))
			continue
		}
		scale := 1e-6 * maxAbs(y)
		st.cmpF(t, c.Name+" c", sampleIf(c, b.C), c.o("lc"), scale)
		buf := make([]float64, len(xq))
		b.EvalAll(buf, xq)
		st.cmpF(t, c.Name+" y0", sampleIf(c, buf), c.o("ly0"), scale)
	}
	st.report(t, "splrep (fpcurf port)")
	t.Logf("splrep: %d fits with identical knots, %d rejected (SciPy raises, or NaN data)", fits, rejected)
	st.within(t, "splrep (fpcurf port)", 1<<62, 1e-9)
}

// FuzzSplrep: no input makes Splrep panic or return a spline whose knot
// vector is not sorted or whose coefficient count does not match it.
func FuzzSplrep(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, uint8(3), 0.5)
	f.Add([]byte{0, 0, 0, 0, 0, 9, 9, 9}, uint8(1), 0.0)
	f.Add([]byte{5, 1, 200, 3, 3, 3, 3, 7, 250, 0, 13}, uint8(5), 1e-9)
	f.Fuzz(func(t *testing.T, data []byte, k uint8, s float64) {
		m := len(data) / 2
		x := make([]float64, m)
		y := make([]float64, m)
		acc := 0.0
		for i := 0; i < m; i++ {
			acc += float64(data[2*i] % 4) // repeated abscissae included
			x[i] = acc
			y[i] = float64(int8(data[2*i+1]))
		}
		sm, err := Splrep(x, y, int(k%7), s, nil)
		if err != nil {
			return
		}
		b := sm.Spline
		if len(b.C) != len(b.T)-b.K-1 {
			t.Fatalf("%d coefficients for %d knots, k=%d", len(b.C), len(b.T), b.K)
		}
		for i := 1; i < len(b.T); i++ {
			if b.T[i] < b.T[i-1] {
				t.Fatalf("knots not sorted: %v", b.T)
			}
		}
	})
}

// TestFitpackRoot drives the part-2 iteration through each of its branches
// with synthetic f(p).
func TestFitpackRoot(t *testing.T) {
	if ier, p := fitpackRoot(func(p float64) float64 { return 2 - p }, 1, 2, -1, 1e-12); ier != 0 || math.Abs(p-2) > 1e-9 {
		t.Errorf("root: %d %g", ier, p)
	}
	// A constant f: one rational step, then f2 >= f1 is not convex (ier 2).
	if ier, _ := fitpackRoot(func(p float64) float64 { return 1 }, 1, 3, -1, 1e-12); ier != 2 {
		t.Errorf("constant: %d", ier)
	}
	seq := func(v ...float64) func(float64) float64 {
		i := 0
		return func(float64) float64 { i++; return v[min(i-1, len(v)-1)] }
	}
	// p too large, then too small with p3 known: both bracket updates, then
	// values that never meet the target: p grows by 1/0.04 until the
	// iteration limit (ier 3).
	if ier, _ := fitpackRoot(seq(-1, 1, 5), 1, 1, -1, 1e-12); ier != 3 {
		t.Errorf("brackets: %d", ier)
	}
	// p too small twice with p3 still infinite, then too large with p1 > 0
	// so that p is pulled back between p1 and p2.
	if ier, _ := fitpackRoot(seq(1, 1, -1, -1, 0.5, -0.5), 1, 1, -1, 1e-12); ier == 0 {
		t.Errorf("pull back: %d", ier)
	}
	// fpknot's error exit: no interval holds a data point.
	if n, nr := fpknotAdd([]float64{0, 1}, make([]float64, 10), 8, make([]float64, 10), make([]int, 10), 1); n != 9 || nr != 2 {
		t.Errorf("fpknot exit: %d %d", n, nr)
	}
}

func TestSplrepErrorsAndStatus(t *testing.T) {
	x, y, _ := perfDataPlain(40)
	bad := func(name string, x, y []float64, k int, s float64, o *SplrepOptions, want error) {
		t.Helper()
		if _, err := Splrep(x, y, k, s, o); err != want {
			t.Errorf("%s: got %v, want %v", name, err, want)
		}
	}
	bad("k0", x, y, 0, 1, nil, ErrShape)
	bad("k6", x, y, 6, 1, nil, ErrShape)
	bad("few", x[:3], y[:3], 3, 1, nil, ErrShape)
	bad("len y", x, y[:5], 3, 1, nil, ErrShape)
	bad("len w", x, y, 3, 1, &SplrepOptions{W: ones(3)}, ErrShape)
	bad("user knots", x, y, 3, 1, &SplrepOptions{T: []float64{0, 0, 0, 0, 10, 10, 10, 10}}, ErrShape)
	bad("NaN y", x, append([]float64{math.NaN()}, y[1:]...), 3, 1, nil, ErrNonFinite)
	bad("NaN w", x, y, 3, 1, &SplrepOptions{W: append([]float64{math.NaN()}, ones(39)...)}, ErrNonFinite)
	bad("NaN s", x, y, 3, math.NaN(), nil, ErrNonFinite)
	bad("Inf s", x, y, 3, math.Inf(1), nil, ErrNonFinite)
	bad("unsorted", append([]float64{5}, x[1:]...), y, 3, 1, nil, ErrNotIncreasing)
	bad("negative s", x, y, 3, -1, nil, ErrShape)
	bad("nest", x, y, 3, 1, &SplrepOptions{Nest: 7}, ErrShape)
	bad("nest for s=0", x, y, 3, 0, &SplrepOptions{Nest: 20}, ErrShape)
	bad("bounds inside", x, y, 3, 1, &SplrepOptions{Bounds: &[2]float64{1, 10}}, ErrShape)
	bad("bounds inside right", x, y, 3, 1, &SplrepOptions{Bounds: &[2]float64{0, 9}}, ErrShape)
	bad("one abscissa", []float64{1, 1, 1, 1, 1}, []float64{1, 2, 3, 4, 5}, 3, 1, nil, ErrShape)

	// s = 0: the interpolating spline (ier -1), knots at the data for odd k
	// and at midpoints for even k.
	for _, k := range []int{2, 3} {
		sm, err := Splrep(x, y, k, 0, nil)
		if err != nil || sm.Ier != -1 || len(sm.Spline.T) != len(x)+k+1 {
			t.Fatalf("k=%d interpolation: %v %+v", k, err, sm)
		}
		v := make([]float64, len(x))
		sm.Spline.EvalAll(v, x)
		for i := range v {
			if math.Abs(v[i]-y[i]) > 1e-9 {
				t.Fatalf("k=%d: not interpolating at %d: %g vs %g", k, i, v[i], y[i])
			}
		}
	}
	// A huge s: the least-squares polynomial (ier -2).
	if sm, _ := Splrep(x, y, 3, 1e9, nil); sm.Ier != -2 || len(sm.Spline.T) != 8 {
		t.Errorf("polynomial: %+v", sm)
	}
	// nest too small for s: ier 1 with nest knots.
	if sm, _ := Splrep(x, y, 3, 1e-6, &SplrepOptions{Nest: 12, Bounds: &[2]float64{-1, 11}}); sm.Ier != 1 || len(sm.Spline.T) != 12 {
		t.Errorf("nest: %+v", sm)
	}
	// s within 0.1% of the least-squares fit's fp: accepted at once.
	poly, _ := Splrep(x, y, 3, 1e9, nil)
	if sm, _ := Splrep(x, y, 3, poly.Fp*(1+1e-5), nil); sm.Ier != -2 || sm.P != 0 || len(sm.Spline.T) != 8 {
		t.Errorf("accept: %+v", sm)
	}
	// Weights and the s ~ m - sqrt(2m) default of splrep with weights.
	if sm, err := Splrep(x, y, 3, 0.01, &SplrepOptions{W: ones(40)}); err != nil || sm.Ier != 0 || sm.P <= 0 {
		t.Errorf("weights: %v %+v", err, sm)
	}
}
