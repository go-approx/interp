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
	// Weights and the s ~ m - sqrt(2m) default of splrep with weights.
	if sm, err := Splrep(x, y, 3, 0.01, &SplrepOptions{W: ones(40)}); err != nil || sm.Ier != 0 || sm.P <= 0 {
		t.Errorf("weights: %v %+v", err, sm)
	}
}
