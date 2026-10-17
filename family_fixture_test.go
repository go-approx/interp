package interp

import (
	"math"
	"strings"
	"testing"
)

func TestShapeFixture(t *testing.T) {
	fx := loadFixture(t, "shape")
	var st, stX Stats
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		var p *PPoly
		var err error
		switch c.Op {
		case "pchip":
			p, err = NewPchip(x, y)
		default:
			p, err = NewAkima(x, y, c.Op == "makima")
		}
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		s := &st
		if strings.Contains(c.Name, "clustered") || strings.Contains(c.Name, "tinyspan") {
			s = &stX
		}
		ppolyChecks(t, s, c, p, xq, 0, 1, 2)
		q := *p
		q.Extrap = NaNOutside
		buf := make([]float64, len(xq))
		q.EvalAll(buf, xq)
		s.cmpF(t, c.Name+" y0ne", sampleIf(c, buf), c.o("y0ne"), c.floor())
	}
	st.report(t, "Pchip/Akima/makima")
	stX.report(t, "  clustered/tiny span")
	st.within(t, "Pchip/Akima/makima", 512, 1e-13)
	stX.within(t, "shape clustered", 1024, 1e-13)
}

func interp1Options(num map[string]float64) *Interp1Options {
	o := &Interp1Options{AssumeSorted: num["sorted"] == 1, BoundsError: num["be"] == 1, Extrapolate: num["extrap"] == 1}
	if v, ok := num["fb"]; ok {
		o.Fill = []float64{v, num["fa"]}
	} else if v, ok := num["fill"]; ok {
		o.Fill = []float64{v}
	}
	return o
}

func TestInterp1DFixture(t *testing.T) {
	fx := loadFixture(t, "interp1d")
	var st Stats
	kinds := map[string]Kind{"linear": Linear, "nearest": Nearest, "nearest-up": NearestUp, "previous": Previous, "next": Next}
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		dst := make([]float64, len(xq))
		var err error
		if c.Op == "npinterp" {
			left, right := math.NaN(), math.NaN()
			if len(y) > 0 {
				left, right = y[0], y[len(y)-1]
			}
			if v, ok := c.Num["left"]; ok {
				left, right = v, c.Num["right"]
			}
			err = Interp(dst, xq, x, y, left, right)
		} else {
			var f *Interp1D
			f, err = NewInterp1D(x, y, kinds[c.Op], interp1Options(c.Num))
			if err == nil {
				err = f.EvalAll(dst, xq)
			}
		}
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		st.cmp(t, c.Name, sampleIf(c, dst), c.o("y"))
	}
	st.report(t, "interp1d / np.interp")
	st.within(t, "interp1d", 0, 0) // bit-identical to scipy on every value
}

func bsplineFor(c *Case, x, y []float64) (*BSpline, error) {
	k := int(c.n("k"))
	o := &SplineOptions{}
	if t := c.v("t"); t != nil {
		o.T = t
	}
	switch c.Op {
	case "periodic":
		o.Periodic = true
	case "clamped":
		o.Left, o.Right = ClampedDerivs(), ClampedDerivs()
	case "natural":
		o.Left, o.Right = NaturalDerivs(), NaturalDerivs()
	case "deriv":
		for i, ord := range c.v("ol") {
			o.Left = append(o.Left, DerivBC{int(ord), c.v("vl")[i]})
		}
		for i, ord := range c.v("or") {
			o.Right = append(o.Right, DerivBC{int(ord), c.v("vr")[i]})
		}
	}
	return MakeInterpSpline(x, y, k, o)
}

func TestBSplineFixture(t *testing.T) {
	fx := loadFixture(t, "bspline")
	var st, stX, stI, stK, stBig, stBigI Stats
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		b, err := bsplineFor(c, x, y)
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		s, sI := &st, &stI
		if strings.Contains(c.Name, "clustered") {
			s = &stX
		}
		if c.Num["big"] == 1 { // 1e5 intervals: antiderivatives and integrals accumulate 1e5 terms
			s, sI = &stBig, &stBigI
		}
		k := b.K
		big := c.Num["big"] == 1
		stK.cmp(t, c.Name+" t", sampleIf(c, b.T), c.o("t"))
		s.cmpF(t, c.Name+" c", sampleIf(c, b.C), c.o("c"), c.floor())
		buf := make([]float64, len(xq))
		for nu := 0; nu <= 3; nu++ {
			b.EvalNu(buf, xq, nu)
			key := "y" + string(rune('0'+nu))
			s.cmpF(t, c.Name+" "+key, sampleIf(c, buf), c.o(key), c.floor())
		}
		q := *b
		q.Extrap = NaNOutside
		q.EvalAll(buf, xq)
		s.cmpF(t, c.Name+" y0ne", sampleIf(c, buf), c.o("y0ne"), c.floor())
		if k >= 1 && !big {
			d, err := b.Derivative(1)
			if err != nil {
				t.Errorf("%s: derivative: %v", c.Name, err)
			} else {
				s.cmpF(t, c.Name+" der", d.C, c.o("der")[:len(d.C)], c.floor())
			}
			if k >= 2 {
				d2, err := b.Derivative(2)
				if err != nil {
					t.Errorf("%s: derivative 2: %v", c.Name, err)
				} else {
					s.cmpF(t, c.Name+" der2", d2.C, c.o("der2")[:len(d2.C)], c.floor())
				}
			}
		}
		a := b.Antiderivative(1)
		a.EvalAll(buf, xq)
		wantAnti := c.o("anti")
		if k == 0 { // scipy's antiderivative of a piecewise constant is degenerate at and beyond the right end
			for i, v := range xq {
				if !(v < x[len(x)-1]) {
					buf[i], wantAnti[i] = 0, 0
				}
			}
		}
		s.cmpF(t, c.Name+" anti", sampleIf(c, buf), wantAnti, c.floor())
		if !big && k > 0 {
			s.cmpF(t, c.Name+" antic", a.C, c.o("antic")[:len(a.C)], c.floor())
		}
		lo, hi := x[0], x[len(x)-1]
		sp := hi - lo
		ab := []float64{lo + 0.2*sp, lo + 0.9*sp, hi + 0.3*sp, lo - 0.2*sp, lo - sp, lo + 3*sp}
		got := []float64{b.Integrate(ab[0], ab[1]), b.Integrate(ab[2], ab[3]), b.Integrate(ab[4], ab[5])}
		wantInt := c.o("int")
		if k == 0 { // beyond the right end, as above
			got, wantInt = got[:1], wantInt[:1]
		}
		sI.cmpF(t, c.Name+" int", got, wantInt, c.floor())
		got = []float64{q.Integrate(ab[0], ab[1]), q.Integrate(ab[2], ab[3]), q.Integrate(ab[4], ab[5])}
		want := c.o("intne")
		if k == 0 { // scipy's splint returns NaN when the clipped interval reaches the end of the knots
			got, want = got[:1], want[:1]
		}
		sI.cmpF(t, c.Name+" intne", got, want, c.floor())
	}
	st.report(t, "make_interp_spline")
	stX.report(t, "  clustered")
	stK.report(t, "  knots")
	stI.report(t, "  integrate")
	stBig.report(t, "  n = 1e5")
	stBigI.report(t, "  n = 1e5 integrate")
	st.within(t, "make_interp_spline", 1<<15, 1e-8) // periodic with n <= k is a dense, ill-conditioned system
	stX.within(t, "make_interp_spline clustered", 16, 1e-14)
	stK.within(t, "make_interp_spline knots", 0, 0)
	stI.within(t, "make_interp_spline integrate", 1<<15, 1e-11)
	stBig.within(t, "make_interp_spline n=1e5", 1<<62, 1e-7)
	stBigI.within(t, "make_interp_spline n=1e5 integrate", 1<<62, 1e-9)
}

func TestLsqFixture(t *testing.T) {
	fx := loadFixture(t, "lsq")
	var st Stats
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		b, err := MakeLsqSpline(x, y, c.v("t"), int(c.n("k")), c.v("w"))
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		st.cmpF(t, c.Name+" c", sampleIf(c, b.C), c.o("c"), 1e-6*maxAbs(y))
		buf := make([]float64, len(xq))
		b.EvalNu(buf, xq, 0)
		st.cmpF(t, c.Name+" y0", sampleIf(c, buf), c.o("y0"), 1e-6*maxAbs(y))
		b.EvalNu(buf, xq, 1)
		st.cmpF(t, c.Name+" y1", sampleIf(c, buf), c.o("y1"), 1e-6*maxAbs(y))
	}
	st.report(t, "make_lsq_spline")
	st.within(t, "make_lsq_spline", 1<<62, 1e-9)
}

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if !math.IsNaN(x) && !math.IsInf(x, 0) {
			m = math.Max(m, math.Abs(x))
		}
	}
	return m
}

func TestSplrepFixture(t *testing.T) {
	fx := loadFixture(t, "splrep")
	var st, stLegacy, stKnots Stats
	var knotsEq, knotsDiff, legEq, legDiff, newOK, legOK int
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		k := int(c.n("k"))
		var o *SplrepOptions
		if w := c.v("w"); w != nil {
			o = &SplrepOptions{W: w}
		}
		sm, err := MakeSplrep(x, y, k, c.n("s"), o)
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if c.o("ok")[0] == 0 {
			if err == nil {
				t.Logf("%s: make_splrep raised, Go accepted", c.Name)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		newOK++
		b := sm.Spline
		want := c.o("t")
		if len(b.T) == int(c.o("nt")[0]) && ulpAll(sampleIf(c, b.T), want) == 0 {
			knotsEq++
		} else {
			knotsDiff++
			t.Errorf("%s: knots differ from make_splrep: %d vs %d", c.Name, len(b.T), len(want))
			continue
		}
		stKnots.cmp(t, c.Name+" t", sampleIf(c, b.T), want)
		scale := 1e-6 * maxAbs(y)
		st.cmpF(t, c.Name+" c", sampleIf(c, b.C), c.o("c"), scale)
		buf := make([]float64, len(xq))
		b.EvalAll(buf, xq)
		st.cmpF(t, c.Name+" y0", sampleIf(c, buf), c.o("y0"), scale)
		if c.o("lok")[0] == 1 {
			legOK++
			lt := c.o("lt")
			if len(b.T) == int(c.o("lnt")[0]) && ulpAll(sampleIf(c, b.T), lt) == 0 {
				legEq++
				stLegacy.cmpF(t, c.Name+" ly0", sampleIf(c, buf), c.o("ly0"), scale)
			} else {
				legDiff++
			}
		}
	}
	st.report(t, "make_splrep")
	stKnots.report(t, "  knots")
	stLegacy.report(t, "  vs splrep (same knots)")
	t.Logf("make_splrep: %d fits, knot vectors identical %d, different %d; legacy splrep: identical knots in %d of %d", newOK, knotsEq, knotsDiff, legEq, legOK)
	st.within(t, "make_splrep", 1<<62, 1e-9)
	stKnots.within(t, "make_splrep knots", 0, 0)
	stLegacy.within(t, "splrep (same knots)", 1<<62, 1e-10)
}

// ulpAll is the largest ULP distance between two equally long slices.
func ulpAll(a, b []float64) uint64 {
	var m uint64
	for i := range a {
		m = max(m, ulpDiff(a[i], b[i]))
	}
	return m
}

func TestRGIFixture(t *testing.T) {
	fx := loadFixture(t, "rgi")
	stats := map[string]*Stats{}
	for i := range fx.Cases {
		c := &fx.Cases[i]
		d := int(c.n("d"))
		axes := make([][]float64, 0, d)
		for j := 0; j < d; j++ {
			axes = append(axes, c.v("ax"+string(rune('0'+j))))
		}
		o := &RGIOptions{Extrapolate: c.n("extrap") == 1}
		r, err := NewRGI(axes, c.v("vals"), c.Op, o)
		if c.Err != "" {
			if err == nil {
				t.Errorf("%s: oracle raised %q, Go accepted", c.Name, c.Err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		st := stats[c.Op]
		if st == nil {
			st = &Stats{}
			stats[c.Op] = st
		}
		xi := c.v("xi")
		got := make([]float64, len(xi)/d)
		if err := r.Eval(got, xi); err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		fl := 1e-6 * maxAbs(c.v("vals"))
		st.cmpF(t, c.Name, got, c.o("y"), fl)
		got2, err := Interpn(axes, c.v("vals"), xi, c.Op, o)
		if err != nil {
			t.Errorf("%s: interpn: %v", c.Name, err)
			continue
		}
		st.cmpF(t, c.Name+" interpn", got2, c.o("yi"), fl)
	}
	budgets := map[string]float64{"linear": 1e-14, "nearest": 0, "slinear": 1e-13, "cubic": 1e-12, "quintic": 1e-11}
	for _, m := range []string{"linear", "nearest", "slinear", "cubic", "quintic"} {
		stats[m].report(t, "RegularGrid "+m)
		stats[m].within(t, "RegularGrid "+m, 1<<62, budgets[m])
	}
}

func TestBicubicFixture(t *testing.T) {
	fx := loadFixture(t, "bicubic")
	var st Stats
	for i := range fx.Cases {
		c := &fx.Cases[i]
		b, err := NewBicubic(c.v("x"), c.v("y"), c.v("z"), int(c.n("kx")), int(c.n("ky")))
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		xi, yi := c.v("xi"), c.v("yi")
		fl := 1e-6 * maxAbs(c.v("z"))
		for _, q := range []struct {
			key    string
			dx, dy int
		}{{"ev", 0, 0}, {"dx", 1, 0}, {"dy", 0, 1}, {"dxy", 1, 1}, {"dyy", 0, 2}} {
			if _, ok := c.Out[q.key]; !ok {
				continue
			}
			got := make([]float64, len(xi))
			for j := range xi {
				got[j] = b.Deriv(xi[j], yi[j], q.dx, q.dy)
			}
			st.cmpF(t, c.Name+" "+q.key, got, c.o(q.key), fl)
		}
		xg, yg := c.v("xg"), c.v("yg")
		grid := make([]float64, len(xg)*len(yg))
		if err := b.EvalGrid(grid, xg, yg, 0, 0); err != nil {
			t.Fatal(err)
		}
		st.cmpF(t, c.Name+" grid", grid, c.o("grid"), fl)
	}
	st.report(t, "RectBivariateSpline")
	st.within(t, "RectBivariateSpline", 1<<62, 1e-10)
}
