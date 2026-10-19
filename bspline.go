package interp

import (
	"errors"
	"math"
)

// ErrKnots reports an unusable knot vector, or a derivative of a B-spline
// whose knots make a divisor vanish.
var ErrKnots = errors.New("interp: unusable knot vector")

// BSpline is a univariate spline in B-spline form, scipy.interpolate.BSpline
// for one dependent variable: sum over j of C[j]*B_{j,K}(x) on the knot
// vector T (len(T) == len(C)+K+1). The base interval is [T[K], T[n]],
// n == len(C).
type BSpline struct {
	T      []float64
	C      []float64
	K      int
	Extrap Extrapolation
}

// NewBSpline checks and wraps knots t, coefficients c and degree k. At least
// n = len(t)-k-1 >= k+1 coefficients are needed; extra ones are ignored. The
// slices are retained.
func NewBSpline(t, c []float64, k int) (*BSpline, error) {
	n := len(t) - k - 1
	if k < 0 || n < k+1 || len(c) < n {
		return nil, ErrShape
	}
	for i := range t {
		if math.IsNaN(t[i]) || math.IsInf(t[i], 0) {
			return nil, ErrNonFinite
		}
		if i > 0 && t[i] < t[i-1] {
			return nil, ErrKnots
		}
	}
	return &BSpline{T: t, C: c[:n], K: k}, nil
}

// N is the number of coefficients.
func (b *BSpline) N() int { return len(b.T) - b.K - 1 }

// findKnotInterval returns l in [k, n-1] with t[l] <= x < t[l+1] (the last
// interval is closed). x outside [t[k], t[n]] gives k or n-1.
func findKnotInterval(t []float64, k int, x float64, hint int) int {
	n := len(t) - k - 1
	lo, hi := k, n
	if hint >= k && hint < n && x >= t[hint] {
		if x < t[hint+1] {
			return hint
		}
		// sorted queries: the next interval is the likely one
		if hint+2 <= n && x < t[hint+2] {
			return hint + 1
		}
		lo = min(hint+1, n-1)
	}
	// The last i in [lo, hi) with x >= t[i] (lo itself when there is none).
	// Note: Go 1.27 still compiles the float comparison to a branch, so
	// random queries mispredict here; a branch-free form would need the
	// knots stored as order-preserving integers.
	for size := hi - lo; size > 1; {
		half := size >> 1
		next := lo + half
		if x >= t[next] {
			lo = next
		}
		size -= half
	}
	// At the right end the interval must have non-zero width (repeated end
	// knots, as an antiderivative of a piecewise constant spline has).
	for k > 0 && lo > k && t[lo] == t[lo+1] {
		lo--
	}
	return lo
}

// basis computes into h[0..k] the nu-th derivatives at x of the k+1 B-splines
// that do not vanish on knot interval ell (scipy _deBoor_D). The recursion
// runs in place, carrying the previous entry of the row it is overwriting.
func basis(t []float64, x float64, k, ell, nu int, h []float64) {
	h[0] = 1
	for j := 1; j <= k-nu; j++ {
		old := h[0]
		h[0] = 0
		for n := 1; n <= j; n++ {
			next := 0.0
			if n < j {
				next = h[n]
			}
			xb, xa := t[ell+n], t[ell+n-j]
			if xb == xa {
				h[n] = 0
			} else {
				w := old / (xb - xa)
				h[n-1] += float64(w * (xb - x))
				h[n] = float64(w * (x - xa))
			}
			old = next
		}
	}
	for j := max(k-nu+1, 1); j <= k; j++ {
		old := h[0]
		h[0] = 0
		for n := 1; n <= j; n++ {
			next := 0.0
			if n < j {
				next = h[n]
			}
			xb, xa := t[ell+n], t[ell+n-j]
			if xb == xa {
				h[n] = 0
			} else {
				w := float64(j) * old / (xb - xa)
				h[n-1] -= w
				h[n] = w
			}
			old = next
		}
	}
}

// EvalNu evaluates the nu-th derivative (nu >= 0) at every point of xs into
// dst. Outside the base interval the result follows b.Extrap.
func (b *BSpline) EvalNu(dst, xs []float64, nu int) {
	k := b.K
	n := b.N()
	var hbuf [16]float64
	h := hbuf[:]
	if k+1 > len(h) {
		h = make([]float64, k+1)
	}
	if nu > k {
		for i := range xs {
			dst[i] = 0
			if math.IsNaN(xs[i]) {
				dst[i] = math.NaN()
			}
		}
		return
	}
	tb, te := b.T[k], b.T[n]
	hint := k
	for i, x := range xs {
		if b.Extrap == Wrap {
			x = tb + floorMod(x-tb, te-tb)
		}
		if math.IsNaN(x) || (b.Extrap == NaNOutside && (x < tb || x > te)) {
			dst[i] = math.NaN()
			continue
		}
		ell := findKnotInterval(b.T, k, x, hint)
		hint = ell
		basis(b.T, x, k, ell, nu, h)
		s := 0.0
		for a := 0; a <= k; a++ {
			s += float64(b.C[ell+a-k] * h[a])
		}
		dst[i] = s
	}
}

// EvalAll evaluates the spline at every point of xs into dst.
func (b *BSpline) EvalAll(dst, xs []float64) { b.EvalNu(dst, xs, 0) }

// At evaluates the spline at one point.
func (b *BSpline) At(x float64) float64 {
	var d [1]float64
	b.EvalNu(d[:], []float64{x}, 0)
	return d[0]
}

// Derivative returns the nu-th derivative (nu <= K) as a B-spline of degree
// K-nu.
func (b *BSpline) Derivative(nu int) (*BSpline, error) {
	if nu < 0 || nu > b.K {
		return nil, ErrShape
	}
	t, c, k := b.T, b.C, b.K
	for j := 0; j < nu; j++ {
		n := len(c)
		nc := make([]float64, n-1)
		for i := range nc {
			dt := t[k+1+i] - t[1+i]
			if dt == 0 {
				return nil, ErrKnots
			}
			nc[i] = float64((c[i+1]-c[i])*float64(k)) / dt
		}
		c = nc
		t = t[1 : len(t)-1]
		k--
	}
	if nu == 0 {
		c = append([]float64(nil), c...)
		t = append([]float64(nil), t...)
	}
	return &BSpline{T: t, C: c, K: k, Extrap: b.Extrap}, nil
}

// Antiderivative returns the nu-th antiderivative, a B-spline of degree
// K+nu vanishing at the left end of the base interval.
func (b *BSpline) Antiderivative(nu int) *BSpline {
	t, c, k := b.T, b.C, b.K
	for j := 0; j < nu; j++ {
		n := len(c)
		nc := make([]float64, n+1)
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += float64(c[i] * (t[i+k+1] - t[i]))
			nc[i+1] = sum / float64(k+1)
		}
		nt := make([]float64, len(t)+2)
		nt[0] = t[0]
		copy(nt[1:], t)
		nt[len(nt)-1] = t[len(t)-1]
		t, c = nt, nc
		k++
	}
	e := b.Extrap
	if e == Wrap {
		e = NaNOutside
	}
	return &BSpline{T: t, C: c, K: k, Extrap: e}
}

// Integrate returns the integral from a to b (scipy BSpline.integrate); the
// bounds may be swapped, which flips the sign. Outside the base interval the
// integrand follows Extrap: continued (Extend), zero (NaNOutside, the
// interval is clipped) or periodic (Wrap).
func (b *BSpline) Integrate(lo, hi float64) float64 {
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

// DesignMatrix holds the banded collocation matrix of a B-spline basis: row i
// has the K+1 values Vals[i*(K+1):(i+1)*(K+1)] in columns Left[i] onwards.
type DesignMatrix struct {
	Left []int
	Vals []float64
	K    int
	N    int // number of columns
}

// NewDesignMatrix evaluates the B-spline basis of degree k on knots t at x
// (scipy BSpline.design_matrix). Points outside the base interval are an
// error unless extrapolate is true.
func NewDesignMatrix(x, t []float64, k int, extrapolate bool) (*DesignMatrix, error) {
	n := len(t) - k - 1
	if k < 0 || n < k+1 {
		return nil, ErrShape
	}
	for i := 1; i < len(t); i++ {
		if t[i] < t[i-1] {
			return nil, ErrKnots
		}
	}
	d := &DesignMatrix{Left: make([]int, len(x)), Vals: make([]float64, len(x)*(k+1)), K: k, N: n}
	hint := k
	for i, xv := range x {
		if math.IsNaN(xv) || (!extrapolate && (xv < t[k] || xv > t[n])) {
			return nil, ErrBounds
		}
		ell := findKnotInterval(t, k, xv, hint)
		hint = ell
		basis(t, xv, k, ell, 0, d.Vals[i*(k+1):(i+1)*(k+1)])
		d.Left[i] = ell - k
	}
	return d, nil
}

// Dense returns the matrix as len(x) rows of N columns, row-major.
func (d *DesignMatrix) Dense() []float64 {
	m := len(d.Left)
	out := make([]float64, m*d.N)
	for i := 0; i < m; i++ {
		copy(out[i*d.N+d.Left[i]:], d.Vals[i*(d.K+1):(i+1)*(d.K+1)])
	}
	return out
}
