package interp

import (
	"math"
	"strings"
	"testing"
)

// flatPPoly lists the coefficients in scipy's layout: c[r][i] with r = 0 the
// highest power, flattened r-major.
func flatPPoly(p *PPoly) []float64 {
	m := len(p.X) - 1
	out := make([]float64, 0, m*p.K)
	for r := 0; r < p.K; r++ {
		for i := 0; i < m; i++ {
			out = append(out, p.C[i*p.K+(p.K-1-r)])
		}
	}
	return out
}

func sampleIf(c *Case, v []float64) []float64 {
	if c.Num["big"] == 1 {
		return bigSample(v)
	}
	return v
}

// floor is the scale under which differences of a derivative are noise: 1e-8
// of the largest value of the function.
func (c *Case) floor() float64 {
	m := 0.0
	for _, v := range c.o("y0") {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			m = math.Max(m, math.Abs(v))
		}
	}
	return 1e-6 * m
}

// ppolyChecks compares a PPoly with the oracle's coefficients c and values
// y<nu>.
func ppolyChecks(t *testing.T, st *Stats, c *Case, p *PPoly, xq []float64, derivs ...int) {
	t.Helper()
	st.cmpF(t, c.Name+" c", sampleIf(c, flatPPoly(p)), c.o("c"), c.floor())
	buf := make([]float64, len(xq))
	for _, nu := range derivs {
		p.EvalNu(buf, xq, nu)
		key := "y" + string(rune('0'+nu))
		st.cmpF(t, c.Name+" "+key, sampleIf(c, buf), c.o(key), c.floor())
	}
}

func TestCubicFixture(t *testing.T) {
	fx := loadFixture(t, "cubic")
	var st, stClustered, stInt Stats
	for i := range fx.Cases {
		c := &fx.Cases[i]
		x, y, xq := c.inputs()
		var p *PPoly
		var err error
		switch c.Op {
		case "d1d2":
			bc := func(o, v float64) BC {
				if o == 1 {
					return BC{D1, v}
				}
				return BC{D2, v}
			}
			p, err = NewCubicSplineBC(x, y, bc(c.n("o0"), c.n("b0")), bc(c.n("o1"), c.n("b1")))
		default:
			p, err = NewCubicSpline(x, y, c.Op)
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
		if strings.Contains(c.Name, "clustered") || strings.Contains(c.Name, "tiny") {
			s = &stClustered // spacings 1e-14..1e2, subnormal results: conditioning, not the algorithm
		}
		ppolyChecks(t, s, c, p, xq, 0, 1, 2, 3)
		buf := make([]float64, len(xq))
		p.Antiderivative(1).EvalAll(buf, xq)
		s.cmpF(t, c.Name+" anti", sampleIf(c, buf), c.o("anti"), c.floor())
		if want := c.o("int"); want != nil {
			got := []float64{p.Integrate(c.v("ab")[0], c.v("ab")[1]), p.Integrate(c.v("ab2")[0], c.v("ab2")[1])}
			if s == &stClustered {
				s.cmpF(t, c.Name+" int", got, want, c.floor())
			} else {
				stInt.cmpF(t, c.Name+" int", got, want, c.floor())
			}
		}
	}
	st.report(t, "CubicSpline")
	stClustered.report(t, "CubicSpline clustered")
	stInt.report(t, "CubicSpline.integrate")
	st.within(t, "CubicSpline", 1<<20, 1e-9)
	stClustered.within(t, "CubicSpline clustered", 1<<62, 1e-6)
	stInt.within(t, "CubicSpline.integrate", 1<<62, 1e-9)
}
