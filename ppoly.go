package interp

import (
	"errors"
	"math"
)

// Errors returned by the constructors. They wrap nothing: compare with
// errors.Is.
var (
	// ErrShape reports inconsistent slice lengths or a degree/size that the
	// method cannot accept.
	ErrShape = errors.New("interp: inconsistent shapes or too few points")
	// ErrNotIncreasing reports x that is not strictly increasing where the
	// method needs it (scipy: "x must be strictly increasing").
	ErrNotIncreasing = errors.New("interp: x must be a strictly increasing sequence")
	// ErrNonFinite reports a NaN or infinity in data that must be finite.
	ErrNonFinite = errors.New("interp: data must be finite")
	// ErrSingular reports a singular linear system (for example a collocation
	// matrix violating the Schoenberg-Whitney conditions).
	ErrSingular = errors.New("interp: singular linear system")
	// ErrBounds reports a query outside the interpolation range when the
	// interpolator was asked to fail instead of filling.
	ErrBounds = errors.New("interp: query point outside the interpolation range")
	// ErrPeriodic reports that y[0] and y[n-1] differ for a periodic spline.
	ErrPeriodic = errors.New("interp: first and last y must be identical for a periodic spline")
	// ErrBC reports an unusable boundary condition.
	ErrBC = errors.New("interp: unusable boundary condition")
)

// Extrapolation says what a piecewise interpolant does outside its base
// interval.
type Extrapolation int

const (
	// Extend continues the first and last polynomial pieces (scipy
	// extrapolate=True).
	Extend Extrapolation = iota
	// NaNOutside returns NaN outside the base interval (extrapolate=False).
	NaNOutside
	// Wrap evaluates periodically (extrapolate="periodic").
	Wrap
)

// PPoly is a piecewise polynomial, scipy.interpolate.PPoly for one
// dependent variable. The pieces are [X[i], X[i+1]); the last one is closed.
//
// Unlike scipy, which stores coefficients highest power first and by
// order-major layout, C is piece-major with ASCENDING powers of (x - X[i]):
// C[i*K+p] multiplies (x-X[i])^p.
type PPoly struct {
	X      []float64 // breakpoints, strictly increasing, length m+1
	K      int       // coefficients per piece (degree K-1)
	C      []float64 // m*K coefficients
	Extrap Extrapolation
}

// NewPPoly builds a PPoly from breakpoints x (m+1 values), k coefficients
// per piece and c (m*k values, piece-major, ascending powers). The slices are
// retained, not copied.
func NewPPoly(x []float64, k int, c []float64, extrap Extrapolation) (*PPoly, error) {
	if len(x) < 2 || k < 1 || len(c) != (len(x)-1)*k {
		return nil, ErrShape
	}
	for i := 1; i < len(x); i++ {
		if !(x[i] > x[i-1]) {
			return nil, ErrNotIncreasing
		}
	}
	return &PPoly{X: x, K: k, C: c, Extrap: extrap}, nil
}

// Coef returns the coefficient of (x-X[i])^p on piece i.
func (p *PPoly) Coef(i, power int) float64 { return p.C[i*p.K+power] }

// findInterval is scipy's find_interval_ascending: the piece holding xv,
// starting the search at hint. It returns -1 for NaN and, without
// extrapolation, for points outside [x[0], x[n]].
func findInterval(x []float64, xv float64, hint int, extrap bool) int {
	nx := len(x)
	a, b := x[0], x[nx-1]
	if !(a <= xv && xv <= b) {
		switch {
		case xv < a && extrap:
			return 0
		case xv > b && extrap:
			return nx - 2
		}
		return -1
	}
	if xv == b {
		return nx - 2
	}
	if xv >= x[hint] && xv < x[hint+1] {
		return hint
	}
	lo, hi := 0, nx-1 // x[lo] <= xv < x[hi]
	for hi-lo > 1 {
		mid := int(uint(lo+hi) >> 1)
		if xv >= x[mid] {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo
}

func floorMod(a, b float64) float64 {
	r := math.Mod(a, b)
	if r != 0 {
		if (b < 0) != (r < 0) {
			r += b
		}
	} else {
		r = math.Copysign(0, b)
	}
	return r
}

// evalPoly1 evaluates the polynomial c (ascending powers) at s: the value
// (dx = 0), the dx-th derivative (dx > 0) or the -dx-th antiderivative
// with zero constants (dx < 0), with scipy's accumulation order.
func evalPoly1(s float64, c []float64, dx int) float64 {
	res, z := 0.0, 1.0
	if dx < 0 {
		for i := 0; i < -dx; i++ {
			z *= s
		}
	}
	last := len(c) - 1
	for kp := 0; kp <= last; kp++ {
		prefactor := 1.0
		switch {
		case dx > 0:
			if kp < dx {
				continue
			}
			for k := kp; k > kp-dx; k-- {
				prefactor *= float64(k)
			}
		case dx < 0:
			for k := kp; k < kp-dx; k++ {
				prefactor /= float64(k + 1)
			}
		}
		res += float64(float64(c[kp]*z) * prefactor)
		if kp < last && kp >= dx {
			z *= s
		}
	}
	return res
}

// evalDirect is evalPoly1 for dx = 0 (the hot path).
func evalDirect(s float64, c []float64) float64 {
	res, z := 0.0, 1.0
	last := len(c) - 1
	for kp := 0; kp < last; kp++ {
		res += float64(c[kp] * z)
		z *= s
	}
	return res + float64(c[last]*z)
}

// wrap maps xv into the base interval for periodic evaluation.
func (p *PPoly) wrap(xv float64) float64 {
	x0 := p.X[0]
	return x0 + floorMod(xv-x0, p.X[len(p.X)-1]-x0)
}

// Eval evaluates the nu-th derivative (nu >= 0) at every point of xs into dst
// (len(dst) == len(xs)). Negative nu evaluates the antiderivative with zero
// constants on each piece (scipy PPoly.__call__(x, nu)).
func (p *PPoly) EvalNu(dst, xs []float64, nu int) {
	hint := 0
	extrap := p.Extrap == Extend
	for j, xv := range xs {
		if p.Extrap == Wrap {
			xv = p.wrap(xv)
		}
		i := findInterval(p.X, xv, hint, extrap || p.Extrap == Wrap)
		if i < 0 {
			dst[j] = math.NaN()
			continue
		}
		hint = i
		c := p.C[i*p.K : (i+1)*p.K]
		if nu == 0 {
			dst[j] = evalDirect(xv-p.X[i], c)
		} else {
			dst[j] = evalPoly1(xv-p.X[i], c, nu)
		}
	}
}

// Eval evaluates the piecewise polynomial at every point of xs into dst.
func (p *PPoly) EvalAll(dst, xs []float64) { p.EvalNu(dst, xs, 0) }

// At evaluates the piecewise polynomial at one point.
func (p *PPoly) At(x float64) float64 {
	var d [1]float64
	p.EvalNu(d[:], []float64{x}, 0)
	return d[0]
}

// Derivative returns the nu-th derivative as a new PPoly; a negative nu is
// the antiderivative.
func (p *PPoly) Derivative(nu int) *PPoly {
	if nu < 0 {
		return p.Antiderivative(-nu)
	}
	k2 := p.K - nu
	m := len(p.X) - 1
	if k2 <= 0 {
		return &PPoly{X: p.X, K: 1, C: make([]float64, m), Extrap: p.Extrap}
	}
	c := make([]float64, m*k2)
	for i := 0; i < m; i++ {
		for q := 0; q < k2; q++ {
			f := 1.0
			for j := 1; j <= nu; j++ {
				f *= float64(q + j)
			}
			c[i*k2+q] = p.C[i*p.K+q+nu] * f
		}
	}
	return &PPoly{X: p.X, K: k2, C: c, Extrap: p.Extrap}
}

// Antiderivative returns the nu-th antiderivative, continuous (with its
// first nu-1 derivatives) across breakpoints and zero at X[0] in the first
// piece. For a periodic PPoly the result does not wrap.
func (p *PPoly) Antiderivative(nu int) *PPoly {
	if nu <= 0 {
		return p.Derivative(-nu)
	}
	m := len(p.X) - 1
	k2 := p.K + nu
	c := make([]float64, m*k2)
	for i := 0; i < m; i++ {
		for q := 0; q < p.K; q++ {
			f := 1.0
			for j := 1; j <= nu; j++ {
				f *= float64(q + j)
			}
			c[i*k2+q+nu] = p.C[i*p.K+q] / f
		}
	}
	for ip := 1; ip < m; ip++ {
		s := p.X[ip] - p.X[ip-1]
		prev := c[(ip-1)*k2 : ip*k2]
		for dx := nu - 1; dx >= 0; dx-- {
			res := evalPoly1(s, prev, dx)
			for kp := 0; kp < dx; kp++ {
				res /= float64(kp + 1)
			}
			c[ip*k2+dx] = res
		}
	}
	e := p.Extrap
	if e == Wrap {
		e = NaNOutside
	}
	return &PPoly{X: p.X, K: k2, C: c, Extrap: e}
}

// Integrate returns the definite integral from a to b (scipy PPoly.integrate),
// extrapolating or not according to the receiver, and handling periodic
// wrapping. If b < a the sign flips.
func (p *PPoly) Integrate(a, b float64) float64 {
	if a == b {
		return 0
	}
	sign := 1.0
	if b < a {
		a, b = b, a
		sign = -1
	}
	if p.Extrap == Wrap {
		return sign * p.integratePeriodic(a, b)
	}
	return sign * p.integrateBase(a, b, p.Extrap == Extend)
}

func (p *PPoly) integrateBase(a, b float64, extrap bool) float64 {
	si := findInterval(p.X, a, 0, extrap)
	ei := findInterval(p.X, b, 0, extrap)
	if si < 0 || ei < 0 {
		return math.NaN()
	}
	tot := 0.0
	for i := si; i <= ei; i++ {
		var vb, va float64
		c := p.C[i*p.K : (i+1)*p.K]
		if i == ei {
			vb = evalPoly1(b-p.X[i], c, -1)
		} else {
			vb = evalPoly1(p.X[i+1]-p.X[i], c, -1)
		}
		if i == si {
			va = evalPoly1(a-p.X[i], c, -1)
		} else {
			va = evalPoly1(0, c, -1)
		}
		tot += vb - va
	}
	return tot
}

func (p *PPoly) integratePeriodic(a, b float64) float64 {
	x0, xn := p.X[0], p.X[len(p.X)-1]
	period := xn - x0
	interval := b - a
	nPeriods := math.Floor(interval / period)
	left := interval - nPeriods*period
	tot := 0.0
	if nPeriods > 0 {
		tot = nPeriods * p.integrateBase(x0, xn, false)
	}
	a = x0 + floorMod(a-x0, period)
	b = a + left
	if b <= xn {
		return tot + p.integrateBase(a, b, false)
	}
	return tot + p.integrateBase(a, xn, false) + p.integrateBase(x0, x0+left+a-xn, false)
}
