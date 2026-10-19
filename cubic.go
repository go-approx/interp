package interp

import (
	"math"
)

// prepare validates the common input of the cubic interpolators: at least
// two points, equal lengths, finite values and strictly increasing x. It
// returns the spacings dx.
func prepare(x, y []float64) ([]float64, error) {
	n := len(x)
	if n < 2 || len(y) != n {
		return nil, ErrShape
	}
	// One pass: v*0 is NaN exactly when v is NaN or ±Inf, so the sum s is
	// NaN iff some input is non-finite (reported first, as before); a
	// non-increasing step is only remembered.
	dx := make([]float64, n-1)
	s := x[n-1]*0 + y[n-1]*0
	increasing := true
	for i := 0; i < n-1; i++ {
		s += x[i]*0 + y[i]*0
		dx[i] = x[i+1] - x[i]
		if dx[i] <= 0 {
			increasing = false
		}
	}
	if s != 0 {
		return nil, ErrNonFinite
	}
	if !increasing {
		return nil, ErrNotIncreasing
	}
	return dx, nil
}

func finiteAll(v []float64) bool {
	for _, f := range v {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

// hermite builds the PPoly of the cubic Hermite interpolant with
// pre-validated inputs (scipy CubicHermiteSpline).
func hermite(x, y, d, dx []float64) *PPoly {
	m := len(x) - 1
	c := make([]float64, 4*m)
	for i := 0; i < m; i++ {
		h := dx[i]
		slope := (y[i+1] - y[i]) / h
		t := ((d[i] + d[i+1]) - float64(2*slope)) / h
		ci := c[4*i : 4*i+4 : 4*i+4]
		ci[0] = y[i]
		ci[1] = d[i]
		ci[2] = (slope-d[i])/h - t
		ci[3] = t / h
	}
	return &PPoly{X: x, K: 4, C: c, Extrap: Extend}
}

// NewCubicHermite returns the piecewise cubic with values y and first
// derivatives dydx at the strictly increasing nodes x
// (scipy.interpolate.CubicHermiteSpline). The slices are retained.
func NewCubicHermite(x, y, dydx []float64) (*PPoly, error) {
	dx, err := prepare(x, y)
	if err != nil {
		return nil, err
	}
	if len(dydx) != len(x) {
		return nil, ErrShape
	}
	if !finiteAll(dydx) {
		return nil, ErrNonFinite
	}
	return hermite(x, y, dydx, dx), nil
}

func sign(v float64) float64 {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	case v == 0:
		return 0
	}
	return v // NaN
}

// pchipEdge is the one-sided three-point end derivative of Moler's pchiptx.m,
// as in scipy PchipInterpolator._edge_case.
func pchipEdge(h0, h1, m0, m1 float64) float64 {
	d := (float64((2*h0+h1)*m0) - float64(h0*m1)) / (h0 + h1)
	if sign(d) != sign(m0) {
		return 0
	}
	if sign(m0) != sign(m1) && math.Abs(d) > 3*math.Abs(m0) {
		return 3 * m0
	}
	return d
}

// NewPchip returns the monotone piecewise cubic Hermite interpolant of
// Fritsch and Carlson (SIAM J. Numer. Anal. 17, 1980) with the end rules of
// Moler's pchiptx.m: scipy.interpolate.PchipInterpolator, MATLAB pchip.
func NewPchip(x, y []float64) (*PPoly, error) {
	hk, err := prepare(x, y)
	if err != nil {
		return nil, err
	}
	n := len(x)
	mk := make([]float64, n-1)
	for i := range mk {
		mk[i] = (y[i+1] - y[i]) / hk[i]
	}
	dk := make([]float64, n)
	if n == 2 {
		dk[0], dk[1] = mk[0], mk[0]
		return hermite(x, y, dk, hk), nil
	}
	for i := 1; i < n-1; i++ {
		// interior node i sits between segments i-1 and i
		if sign(mk[i]) != sign(mk[i-1]) || mk[i] == 0 || mk[i-1] == 0 {
			continue
		}
		w1 := 2*hk[i] + hk[i-1]
		w2 := hk[i] + 2*hk[i-1]
		whmean := (w1/mk[i-1] + w2/mk[i]) / (w1 + w2)
		dk[i] = 1.0 / whmean
	}
	dk[0] = pchipEdge(hk[0], hk[1], mk[0], mk[1])
	dk[n-1] = pchipEdge(hk[n-2], hk[n-3], mk[n-2], mk[n-3])
	return hermite(x, y, dk, hk), nil
}

// NewAkima returns Akima's 1970 interpolant (scipy Akima1DInterpolator with
// method "akima"), or the modified Akima ("makima", also MATLAB makima) if
// makima is true. As in scipy the result does not extrapolate: it is NaN
// outside [x[0], x[n-1]]; set the Extrap field to change that.
func NewAkima(x, y []float64, makima bool) (*PPoly, error) {
	dx, err := prepare(x, y)
	if err != nil {
		return nil, err
	}
	n := len(x)
	t := make([]float64, n)
	if n == 2 {
		t[0] = (y[1] - y[0]) / dx[0]
		t[1] = t[0]
		p := hermite(x, y, t, dx)
		p.Extrap = NaNOutside
		return p, nil
	}
	m := make([]float64, n+3)
	for i := 0; i < n-1; i++ {
		m[i+2] = (y[i+1] - y[i]) / dx[i]
	}
	m[1] = 2*m[2] - m[3]
	m[0] = 2*m[1] - m[2]
	m[n+1] = 2*m[n] - m[n-1]
	m[n+2] = 2*m[n+1] - m[n]
	f1 := make([]float64, n)
	f2 := make([]float64, n)
	f12 := make([]float64, n)
	mmax := math.Inf(-1)
	for i := 0; i < n; i++ {
		t[i] = 0.5 * (m[i+3] + m[i])
		// dm[j] = |m[j+1]-m[j]|; f1 uses dm[i+2], f2 uses dm[i]
		a := math.Abs(m[i+3] - m[i+2])
		b := math.Abs(m[i+1] - m[i])
		if makima {
			a += float64(0.5 * math.Abs(m[i+3]+m[i+2]))
			b += float64(0.5 * math.Abs(m[i+1]+m[i]))
		}
		f1[i], f2[i] = a, b
		f12[i] = a + b
		if f12[i] > mmax {
			mmax = f12[i]
		}
	}
	for i := 0; i < n; i++ {
		if f12[i] > 1e-9*mmax {
			t[i] = m[i+1] + float64((f2[i]/f12[i])*(m[i+2]-m[i+1]))
		}
	}
	p := hermite(x, y, t, dx)
	p.Extrap = NaNOutside
	return p, nil
}

// BCKind selects a boundary condition of a cubic spline.
type BCKind int

const (
	// NotAKnot makes the first (last) two pieces the same cubic.
	NotAKnot BCKind = iota
	// Natural sets the second derivative to zero.
	Natural
	// Clamped sets the first derivative to zero.
	Clamped
	// Periodic joins the ends with continuous first and second derivatives;
	// it must be given for both ends and needs y[0] == y[n-1].
	Periodic
	// D1 prescribes the first derivative to Value.
	D1
	// D2 prescribes the second derivative to Value.
	D2
)

// BC is one boundary condition: a kind and, for D1 and D2, the prescribed
// value.
type BC struct {
	Kind  BCKind
	Value float64
}

func (b BC) order() (int, float64) {
	switch b.Kind {
	case Natural:
		return 2, 0
	case Clamped:
		return 1, 0
	case D1:
		return 1, b.Value
	}
	return 2, b.Value // D2
}

// NewCubicSpline is scipy.interpolate.CubicSpline with a named boundary
// condition for both ends: "not-a-knot", "natural", "clamped" or
// "periodic" (MATLAB spline is "not-a-knot"; "periodic" sets Extrap to Wrap).
func NewCubicSpline(x, y []float64, bc string) (*PPoly, error) {
	var k BCKind
	switch bc {
	case "not-a-knot":
		k = NotAKnot
	case "natural":
		k = Natural
	case "clamped":
		k = Clamped
	case "periodic":
		k = Periodic
	default:
		return nil, ErrBC
	}
	return NewCubicSplineBC(x, y, BC{Kind: k}, BC{Kind: k})
}

// NewCubicSplineBC is NewCubicSpline with a separate condition per end.
func NewCubicSplineBC(x, y []float64, start, end BC) (p *PPoly, err error) {
	defer guard(&err)
	dx, err := prepare(x, y)
	if err != nil {
		return nil, err
	}
	periodic := start.Kind == Periodic || end.Kind == Periodic
	if periodic {
		if start.Kind != end.Kind {
			return nil, ErrBC
		}
		n := len(y)
		if math.Abs(y[0]-y[n-1]) > 1e-15+1e-15*math.Abs(y[n-1]) {
			return nil, ErrPeriodic
		}
	}
	n := len(x)
	slope := make([]float64, n-1)
	for i := range slope {
		slope[i] = (y[i+1] - y[i]) / dx[i]
	}
	if n == 2 {
		if start.Kind == NotAKnot || start.Kind == Periodic {
			start = BC{D1, slope[0]}
		}
		if end.Kind == NotAKnot || end.Kind == Periodic {
			end = BC{D1, slope[0]}
		}
	}
	var s []float64
	switch {
	case n == 3 && start.Kind == NotAKnot && end.Kind == NotAKnot:
		a := []float64{1, 1, 0, dx[1], 2 * (dx[0] + dx[1]), dx[0], 0, 1, 1}
		s = []float64{2 * slope[0], 3 * (float64(dx[0]*slope[1]) + float64(dx[1]*slope[0])), 2 * slope[1]}
		must(denseSolve(a, s, 3))
	case n == 3 && start.Kind == Periodic:
		t := (slope[0]/dx[0] + slope[1]/dx[1]) / (1/dx[0] + 1/dx[1])
		s = []float64{t, t, t}
	default:
		s = cubicSlopes(x, y, dx, slope, start, end)
	}
	p = hermite(x, y, s, dx)
	if periodic {
		p.Extrap = Wrap
	}
	return p, nil
}

// cubicSlopes solves the tridiagonal system for the derivative at every node.
func cubicSlopes(x, y, dx, slope []float64, start, end BC) []float64 {
	n := len(x)
	// One pass builds the three diagonals and the right-hand side:
	// du[j] = A[j][j+1], di[j] = A[j][j], dl[j] = A[j+1][j].
	du := make([]float64, n-1)
	di := make([]float64, n)
	dl := make([]float64, n-1)
	b := make([]float64, n)
	for j := 1; j <= n-2; j++ {
		di[j] = 2 * (dx[j-1] + dx[j])
		b[j] = 3 * (float64(dx[j]*slope[j-1]) + float64(dx[j-1]*slope[j]))
		du[j] = dx[j-1]
		dl[j-1] = dx[j]
	}
	if start.Kind == Periodic {
		return cubicPeriodic(dx, slope, du, di, dl, b)
	}
	switch start.Kind {
	case NotAKnot:
		di[0] = dx[1]
		du[0] = x[2] - x[0]
		d := x[2] - x[0]
		b[0] = (float64(float64((dx[0]+2*d)*dx[1])*slope[0]) + float64(float64(dx[0]*dx[0])*slope[1])) / d
	default:
		if ord, v := start.order(); ord == 1 {
			di[0] = 1
			du[0] = 0
			b[0] = v
		} else {
			di[0] = 2 * dx[0]
			du[0] = dx[0]
			b[0] = float64(float64(-0.5*v)*float64(dx[0]*dx[0])) + float64(3*(y[1]-y[0]))
		}
	}
	switch end.Kind {
	case NotAKnot:
		di[n-1] = dx[n-3]
		dl[n-2] = x[n-1] - x[n-3]
		d := x[n-1] - x[n-3]
		b[n-1] = (float64(float64(dx[n-2]*dx[n-2])*slope[n-3]) + float64(float64((2*d+dx[n-2])*dx[n-3])*slope[n-2])) / d
	default:
		if ord, v := end.order(); ord == 1 {
			di[n-1] = 1
			dl[n-2] = 0
			b[n-1] = v
		} else {
			di[n-1] = 2 * dx[n-2]
			dl[n-2] = dx[n-2]
			b[n-1] = float64(float64(0.5*v)*float64(dx[n-2]*dx[n-2])) + float64(3*(y[n-1]-y[n-2]))
		}
	}
	must(gtsv(dl, di, du, b))
	return b
}

// cubicPeriodic solves the cyclic system by condensing it to a tridiagonal
// one of order n-2 (scipy CubicSpline, bc_type="periodic").
func cubicPeriodic(dx, slope, du, di, dl, b []float64) []float64 {
	n := len(di)
	nn := n - 2
	di[0] = 2 * (dx[n-2] + dx[0])
	du[0] = dx[n-2]
	aM10 := dx[n-3]
	aM1M2 := dx[n-2]
	aM1M1 := 2 * (dx[n-2] + dx[n-3])
	aM2M1 := dx[n-4]
	a0M1 := dx[0]
	b[0] = 3 * (float64(dx[0]*slope[n-2]) + float64(dx[n-2]*slope[0]))
	bLast := 3 * (float64(dx[n-2]*slope[n-3]) + float64(dx[n-3]*slope[n-2]))
	b1 := b[:nn]
	b2 := make([]float64, nn)
	b2[0] = -a0M1
	b2[nn-1] = -aM2M1
	// Both right-hand sides share one factorization of the order-nn system.
	must(gtsv2(dl[:nn-1], di[:nn], du[:nn-1], b1, b2))
	sm1 := ((bLast - float64(aM10*b1[0])) - float64(aM1M2*b1[nn-1])) /
		((aM1M1 + float64(aM10*b2[0])) + float64(aM1M2*b2[nn-1]))
	s := make([]float64, n)
	for i := 0; i < nn; i++ {
		s[i] = b1[i] + float64(sm1*b2[i])
	}
	s[n-2] = sm1
	s[n-1] = s[0]
	return s
}
