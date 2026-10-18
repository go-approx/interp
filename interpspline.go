package interp

import (
	"math"
	"sort"
)

// DerivBC prescribes the Order-th derivative of an interpolating B-spline at
// one end to Value.
type DerivBC struct {
	Order int
	Value float64
}

// ClampedDerivs is the condition "clamped": first derivative zero.
func ClampedDerivs() []DerivBC { return []DerivBC{{1, 0}} }

// NaturalDerivs is the condition "natural": second derivative zero.
func NaturalDerivs() []DerivBC { return []DerivBC{{2, 0}} }

// SplineOptions are the optional arguments of MakeInterpSpline. The zero
// value asks for scipy's defaults: not-a-knot knots, no end conditions.
type SplineOptions struct {
	// T are user knots (len(T) >= len(x)+k+1); with end conditions, the
	// total number of conditions must be len(T)-k-1-len(x).
	T []float64
	// Periodic requests a periodic spline (y[0] == y[n-1], automatic knots).
	Periodic bool
	// Left and Right are derivative conditions at x[0] and x[n-1]; they
	// replace the not-a-knot knot choice by clamped end knots.
	Left, Right []DerivBC
}

func notAKnot(x []float64, k int) []float64 {
	var t []float64
	var k2 int
	if k%2 == 1 {
		k2 = (k + 1) / 2
		t = append(t, x...)
	} else {
		k2 = k / 2
		t = make([]float64, len(x)-1)
		for i := range t {
			t[i] = (x[i+1] + x[i]) / 2
		}
	}
	var mid []float64
	if len(t) > 2*k2 {
		mid = t[k2 : len(t)-k2]
	}
	out := make([]float64, 0, len(mid)+2*(k+1))
	for i := 0; i <= k; i++ {
		out = append(out, x[0])
	}
	out = append(out, mid...)
	for i := 0; i <= k; i++ {
		out = append(out, x[len(x)-1])
	}
	return out
}

func augKnots(x []float64, k int) []float64 {
	out := make([]float64, 0, len(x)+2*k)
	for i := 0; i < k; i++ {
		out = append(out, x[0])
	}
	out = append(out, x...)
	for i := 0; i < k; i++ {
		out = append(out, x[len(x)-1])
	}
	return out
}

func periodicKnots(x []float64, k int) []float64 {
	n := len(x)
	xc := append([]float64(nil), x...)
	if k%2 == 0 {
		for i := 1; i <= n-2; i++ {
			xc[i] -= (x[i] - x[i-1]) / 2
		}
	}
	dx := make([]float64, n-1)
	for i := range dx {
		dx[i] = xc[i+1] - xc[i]
	}
	t := make([]float64, n+2*k)
	copy(t[k:], xc)
	for i := 0; i < k; i++ {
		t[k-i-1] = t[k-i] - dx[n-2-(i%(n-1))]
		t[n+k+i] = t[n+k+i-1] + dx[i%(n-1)]
	}
	return t
}

func checkXY(x, y []float64) error {
	if len(x) == 0 || len(x) != len(y) {
		return ErrShape
	}
	if !finiteAll(x) || !finiteAll(y) {
		return ErrNonFinite
	}
	for i := 1; i < len(x); i++ {
		if !(x[i] > x[i-1]) {
			return ErrNotIncreasing
		}
	}
	return nil
}

// MakeInterpSpline returns the degree-k B-spline interpolating (x, y), as
// scipy.interpolate.make_interp_spline for one dependent variable. x must be
// strictly increasing and finite. The default knots are the not-a-knot ones
// (k = 3 gives the cubic not-a-knot spline).
func MakeInterpSpline(x, y []float64, k int, o *SplineOptions) (_ *BSpline, err error) {
	defer guard(&err)
	if o == nil {
		o = &SplineOptions{}
	}
	if err := checkXY(x, y); err != nil {
		return nil, err
	}
	if k < 0 {
		return nil, ErrShape
	}
	n := len(x)
	hasDeriv := len(o.Left) > 0 || len(o.Right) > 0
	extrap := Extend
	if o.Periodic {
		if math.Abs(y[0]-y[n-1]) > 1e-15+1e-5*math.Abs(y[n-1]) {
			return nil, ErrPeriodic
		}
		if o.T != nil || hasDeriv {
			return nil, ErrBC
		}
		extrap = Wrap
	}
	if k == 0 {
		if o.T != nil || hasDeriv {
			return nil, ErrBC
		}
		t := append(append([]float64(nil), x...), x[n-1])
		return &BSpline{T: t, C: append([]float64(nil), y...), K: 0, Extrap: extrap}, nil
	}
	if k == 1 && o.T == nil {
		if hasDeriv {
			return nil, ErrBC
		}
		t := append(append([]float64{x[0]}, x...), x[n-1])
		return &BSpline{T: t, C: append([]float64(nil), y...), K: 1, Extrap: extrap}, nil
	}
	t := o.T
	if t == nil {
		switch {
		case hasDeriv:
			t = augKnots(x, k)
		case o.Periodic:
			t = periodicKnots(x, k)
		default:
			t = notAKnot(x, k)
		}
	}
	if !finiteAll(t) {
		return nil, ErrNonFinite
	}
	for i := 1; i < len(t); i++ {
		if t[i] < t[i-1] {
			return nil, ErrKnots
		}
	}
	if len(t) < n+k+1 || x[0] < t[k] || x[n-1] > t[len(t)-k] {
		return nil, ErrKnots
	}
	if o.Periodic {
		return &BSpline{T: t, C: makePeriodic(x, y, t, k), K: k, Extrap: Wrap}, nil
	}
	nt := len(t) - k - 1
	nl, nr := len(o.Left), len(o.Right)
	for _, d := range append(append([]DerivBC(nil), o.Left...), o.Right...) {
		if d.Order < 0 || d.Order > k {
			return nil, ErrBC
		}
	}
	if nt-n != nl+nr {
		return nil, ErrBC
	}
	m := newBand(nt, k, k)
	h := make([]float64, k+1)
	hint := k
	put := func(row int, x float64, nu int) bool {
		ell := findKnotInterval(t, k, x, hint)
		hint = ell
		basis(t, x, k, ell, nu, h)
		for a := 0; a <= k; a++ {
			col := ell - k + a
			if col-row > k || row-col > k {
				return false
			}
			m.set(row, col, h[a])
		}
		return true
	}
	rhs := make([]float64, nt)
	inBand := true
	for i, d := range o.Left {
		inBand = put(i, x[0], d.Order) && inBand
		rhs[i] = d.Value
	}
	for i := range x {
		inBand = put(nl+i, x[i], 0) && inBand
		rhs[nl+i] = y[i]
	}
	for i, d := range o.Right {
		inBand = put(nt-nr+i, x[n-1], d.Order) && inBand
		rhs[nt-nr+i] = d.Value
	}
	if !inBand {
		return nil, ErrBC
	}
	must(m.factor())
	m.solve(rhs)
	return &BSpline{T: t, C: rhs, K: k, Extrap: Extend}, nil
}

// makePeriodic solves for the coefficients of the periodic interpolating
// spline on the periodic knots t (scipy _make_periodic_spline).
func makePeriodic(x, y, t []float64, k int) []float64 {
	n := len(x)
	if n <= k {
		return makePeriodicFull(x, y, t, k)
	}
	nn := n - 1
	kul := k / 2
	type entry struct {
		col int
		v   float64
	}
	rows := make([][]entry, nn)
	h := make([]float64, k+1)
	for i := 0; i < nn; i++ {
		ell := findKnotInterval(t, k, x[i], k)
		basis(t, x[i], k, ell, 0, h)
		for a := 0; a <= k; a++ {
			col := ((ell-k+a-kul)%nn + nn) % nn
			rows[i] = append(rows[i], entry{col, h[a]})
		}
	}
	bw := k
	m := newBand(nn, bw, bw)
	type corner struct {
		row, col int
		v        float64
	}
	var cs []corner
	var crow []int
	for i, r := range rows {
		has := false
		for _, e := range r {
			if d := e.col - i; d < -bw || d > bw {
				cs = append(cs, corner{i, e.col, e.v})
				has = true
			} else {
				m.a[m.at(i, e.col)] += e.v
			}
		}
		if has {
			crow = append(crow, i)
		}
	}
	must(m.factor())
	b := append([]float64(nil), y[:nn]...)
	m.solve(b)
	u := b
	if len(crow) > 0 {
		r := len(crow)
		z := make([][]float64, r)
		for q, row := range crow {
			z[q] = make([]float64, nn)
			z[q][row] = 1
			m.solve(z[q])
		}
		dot := func(p int, v []float64) float64 {
			s := 0.0
			for _, c := range cs {
				if c.row == crow[p] {
					s += float64(c.v * v[c.col])
				}
			}
			return s
		}
		s := make([]float64, r*r)
		g := make([]float64, r)
		for p := 0; p < r; p++ {
			g[p] = dot(p, u)
			for q := 0; q < r; q++ {
				s[p*r+q] = dot(p, z[q])
			}
			s[p*r+p]++
		}
		must(denseSolve(s, g, r))
		u = append([]float64(nil), b...)
		for q := 0; q < r; q++ {
			for i := range u {
				u[i] -= float64(g[q] * z[q][i])
			}
		}
	}
	c := make([]float64, n+k-1)
	for j := range c {
		c[j] = u[((j-kul)%nn+nn)%nn]
	}
	return c
}

// makePeriodicFull is the dense solution for n <= k points.
func makePeriodicFull(x, y, t []float64, k int) []float64 {
	n := len(x)
	sz := n + k - 1
	a := make([]float64, sz*sz)
	h := make([]float64, k+1)
	for i := 0; i < k-1; i++ {
		basis(t, x[0], k, k, i+1, h)
		for c := 0; c <= k; c++ {
			a[i*sz+c] += h[c]
		}
		basis(t, x[n-1], k, n+k-1, i+1, h)
		for c := 0; c < k; c++ {
			a[i*sz+sz-k+c] -= h[c]
		}
	}
	for i := 0; i < n; i++ {
		left := k
		if x[i] != t[k] {
			left = sort.SearchFloat64s(t, x[i]) - 1
		}
		basis(t, x[i], k, left, 0, h)
		copy(a[(i+k-1)*sz+left-k:], h[:k+1])
	}
	b := make([]float64, sz)
	copy(b[k-1:], y)
	must(denseSolve(a, b, sz))
	return b
}

// lsq is a banded least-squares problem for B-spline coefficients, reduced
// to triangular form by Givens rotations (FITPACK fpcurf/fpgivs/fprota).
// Row j of r holds the entries of columns j..j+k+1 (the last one is the
// fill-in of the discontinuity rows used by smoothing).
type lsq struct {
	nc, k, w int
	r        []float64
	z        []float64
}

func newLsq(nc, k int) *lsq {
	return &lsq{nc: nc, k: k, w: k + 2, r: make([]float64, nc*(k+2)), z: make([]float64, nc)}
}

func (q *lsq) clone() *lsq {
	return &lsq{nc: q.nc, k: q.k, w: q.w, r: append([]float64(nil), q.r...), z: append([]float64(nil), q.z...)}
}

// addRow rotates the row h (columns j0..j0+len(h)-1, at most w of them) with
// right-hand side yv into the triangle and returns the leftover of yv.
// The row slides down the triangle: after the rotation with row j its
// remaining entries line up with row j+1, and fill-in from the wider rows
// below keeps it alive until it vanishes (FITPACK fpcurf, SciPy qr_reduce).
func (q *lsq) addRow(j0 int, h []float64, yv float64) float64 {
	w := q.w
	var win [16]float64
	hw := win[:0]
	if w > len(win) {
		hw = make([]float64, 0, w)
	}
	hw = append(hw, h...)
	for len(hw) < w {
		hw = append(hw, 0)
	}
	for j := j0; j < q.nc; j++ {
		if piv := hw[0]; piv != 0 {
			ww := q.r[j*w]
			var dd float64
			if store := math.Abs(piv); store >= ww {
				dd = store * math.Sqrt(1+float64((ww/piv)*(ww/piv)))
			} else {
				dd = ww * math.Sqrt(1+float64((piv/ww)*(piv/ww)))
			}
			cos, sin := ww/dd, piv/dd
			q.r[j*w] = dd
			s1, s2 := yv, q.z[j]
			q.z[j] = float64(cos*s2) + float64(sin*s1)
			yv = float64(cos*s1) - float64(sin*s2)
			for l := 1; l < w; l++ {
				a, b := hw[l], q.r[j*w+l]
				q.r[j*w+l] = float64(cos*b) + float64(sin*a)
				hw[l] = float64(cos*a) - float64(sin*b)
			}
		}
		copy(hw, hw[1:])
		hw[w-1] = 0
		live := false
		for _, v := range hw {
			if v != 0 {
				live = true
				break
			}
		}
		if !live {
			break
		}
	}
	return yv
}

// solve back-substitutes for the coefficients; it reports false if a
// diagonal entry vanished.
func (q *lsq) solve() ([]float64, bool) {
	c := make([]float64, q.nc)
	for i := q.nc - 1; i >= 0; i-- {
		d := q.r[i*q.w]
		if d == 0 || math.IsNaN(d) {
			return nil, false
		}
		s := q.z[i]
		for j := 1; j < q.w && i+j < q.nc; j++ {
			s -= float64(c[i+j] * q.r[i*q.w+j])
		}
		c[i] = s / d
	}
	return c, true
}

// solveLsq back-substitutes, panicking (see must) if the problem is singular.
func solveLsq(q *lsq) []float64 {
	c, ok := q.solve()
	must(ok)
	return c
}

// buildLsq forms and reduces the weighted observation matrix for data
// (x, y, w) on knots t.
func buildLsq(x, y, w, t []float64, k int) *lsq {
	nc := len(t) - k - 1
	q := newLsq(nc, k)
	h := make([]float64, k+1)
	hint := k
	for i, xv := range x {
		ell := findKnotInterval(t, k, xv, hint)
		hint = ell
		basis(t, xv, k, ell, 0, h)
		wi := w[i]
		for a := range h {
			h[a] *= wi
		}
		q.addRow(ell-k, h, y[i]*wi)
	}
	return q
}

// residuals evaluates the spline with coefficients c at the data and returns
// the weighted squared residuals and their sum.
func residuals(x, y, w, t []float64, k int, c []float64) ([]float64, float64) {
	res := make([]float64, len(x))
	h := make([]float64, k+1)
	hint := k
	fp := 0.0
	for i, xv := range x {
		ell := findKnotInterval(t, k, xv, hint)
		hint = ell
		basis(t, xv, k, ell, 0, h)
		s := 0.0
		for a := 0; a <= k; a++ {
			s += float64(c[ell-k+a] * h[a])
		}
		d := float64(w[i] * (s - y[i]))
		res[i] = float64(d * d)
		fp += res[i]
	}
	return res, fp
}

// MakeLsqSpline returns the weighted least-squares B-spline of degree k on
// the knots t (scipy.interpolate.make_lsq_spline). w may be nil (unit
// weights); x must be non-decreasing and lie in [t[k], t[n]].
func MakeLsqSpline(x, y, t []float64, k int, w []float64) (_ *BSpline, err error) {
	defer guard(&err)
	m := len(x)
	if m == 0 || len(y) != m || (w != nil && len(w) != m) || k < 0 || len(t) < 2*k+2 {
		return nil, ErrShape
	}
	if !finiteAll(x) || !finiteAll(y) || !finiteAll(t) || !finiteAll(w) {
		return nil, ErrNonFinite
	}
	for i := 1; i < m; i++ {
		if x[i] < x[i-1] {
			return nil, ErrNotIncreasing
		}
	}
	for i := 1; i < len(t); i++ {
		if t[i] < t[i-1] {
			return nil, ErrKnots
		}
	}
	nc := len(t) - k - 1
	if m < nc || x[0] < t[k] || x[m-1] > t[nc] {
		return nil, ErrKnots
	}
	if w == nil {
		w = make([]float64, m)
		for i := range w {
			w[i] = 1
		}
	}
	for _, v := range w {
		if v < 0 {
			return nil, ErrShape
		}
	}
	return &BSpline{T: t, C: solveLsq(buildLsq(x, y, w, t, k)), K: k, Extrap: Extend}, nil
}
