package interp

import (
	"math"
	"sort"
)

// NdBSpline is a tensor-product B-spline in d dimensions
// (scipy.interpolate.NdBSpline): sum over multi-indices of C times the
// product of the univariate B-splines of degree K[d] on knots T[d]. C is
// row-major with shape Shape (Shape[d] == len(T[d])-K[d]-1).
type NdBSpline struct {
	T     [][]float64
	K     []int
	Shape []int
	C     []float64
}

func prod(shape []int) int {
	p := 1
	for _, s := range shape {
		p *= s
	}
	return p
}

func checkGrid(axes [][]float64, values []float64) ([]int, error) {
	if len(axes) == 0 {
		return nil, ErrShape
	}
	shape := make([]int, len(axes))
	for d, a := range axes {
		if len(a) < 2 {
			return nil, ErrShape
		}
		if !finiteAll(a) {
			return nil, ErrNonFinite
		}
		for i := 1; i < len(a); i++ {
			if !(a[i] > a[i-1]) {
				return nil, ErrNotIncreasing
			}
		}
		shape[d] = len(a)
	}
	if prod(shape) != len(values) {
		return nil, ErrShape
	}
	return shape, nil
}

// MakeNdInterp returns the tensor-product spline interpolating values
// (row-major on the grid axes) with degree k[d] along axis d and not-a-knot
// knots (scipy make_ndbspl; scipy solves the one big system iteratively, this
// solves one banded system per axis, which gives the same interpolant). A
// single degree may be given for all axes.
func MakeNdInterp(axes [][]float64, values []float64, k ...int) (_ *NdBSpline, err error) {
	defer guard(&err)
	shape, err := checkGrid(axes, values)
	if err != nil {
		return nil, err
	}
	if !finiteAll(values) {
		return nil, ErrNonFinite
	}
	nd := len(axes)
	ks := make([]int, nd)
	for d := range ks {
		switch {
		case len(k) == 1:
			ks[d] = k[0]
		case len(k) == nd:
			ks[d] = k[d]
		default:
			return nil, ErrShape
		}
		if ks[d] < 1 || len(axes[d]) < ks[d]+1 {
			return nil, ErrShape
		}
	}
	c := append([]float64(nil), values...)
	ts := make([][]float64, nd)
	stride := 1
	strides := make([]int, nd)
	for d := nd - 1; d >= 0; d-- {
		strides[d] = stride
		stride *= shape[d]
	}
	for d := 0; d < nd; d++ {
		k := ks[d]
		n := shape[d]
		t := notAKnot(axes[d], k)
		ts[d] = t
		m := newBand(n, k, k)
		h := make([]float64, k+1)
		for i, xv := range axes[d] {
			ell := findKnotInterval(t, k, xv, k)
			basis(t, xv, k, ell, 0, h)
			for a := 0; a <= k; a++ {
				m.set(i, ell-k+a, h[a])
			}
		}
		must(m.factor())
		line := make([]float64, n)
		outer := len(c) / (n * strides[d])
		if strides[d] > 1 {
			// The lines of this axis are columns of n x stride blocks:
			// solve each block row-wise instead of gathering columns.
			for o := 0; o < outer; o++ {
				m.solveRows(c[o*n*strides[d]:(o+1)*n*strides[d]], strides[d])
			}
			continue
		}
		if outer > 1 {
			// The last axis: its lines are the rows of an outer x n matrix.
			// Transposed, they become columns that solveRows handles
			// together, which hides the latency of each line's division
			// chain behind the others.
			buf := make([]float64, len(c))
			transpose(buf, c, outer, n)
			m.solveRows(buf, outer)
			transpose(c, buf, n, outer)
			continue
		}
		for o := 0; o < outer; o++ {
			for in := 0; in < strides[d]; in++ {
				base := o*n*strides[d] + in
				for i := 0; i < n; i++ {
					line[i] = c[base+i*strides[d]]
				}
				m.solve(line)
				for i := 0; i < n; i++ {
					c[base+i*strides[d]] = line[i]
				}
			}
		}
	}
	return &NdBSpline{T: ts, K: ks, Shape: shape, C: c}, nil
}

// transpose writes the transpose of the r x c row-major matrix src into dst
// (c x r), in tiles so that both sides stay in cache.
func transpose(dst, src []float64, r, c int) {
	const tile = 32
	for i0 := 0; i0 < r; i0 += tile {
		i1 := min(i0+tile, r)
		for j0 := 0; j0 < c; j0 += tile {
			j1 := min(j0+tile, c)
			for i := i0; i < i1; i++ {
				row := src[i*c : (i+1)*c]
				for j := j0; j < j1; j++ {
					dst[j*r+i] = row[j]
				}
			}
		}
	}
}

// EvalNu evaluates the spline at the points xi (len(dst) points of d
// coordinates, row-major) into dst; nu gives the derivative order along each
// axis (nil for none). Points outside the knot range are extrapolated by the
// end polynomial pieces.
func (s *NdBSpline) EvalNu(dst, xi []float64, nu []int) {
	nd := len(s.T)
	strides := make([]int, nd)
	stride := 1
	for d := nd - 1; d >= 0; d-- {
		strides[d] = stride
		stride *= s.Shape[d]
	}
	hs := make([][]float64, nd)
	ells := make([]int, nd)
	for d := range hs {
		hs[d] = make([]float64, s.K[d]+1)
	}
	for p := range dst {
		nan := false
		for d := 0; d < nd; d++ {
			x := xi[p*nd+d]
			if math.IsNaN(x) {
				nan = true
				break
			}
			ell := findKnotInterval(s.T[d], s.K[d], x, s.K[d])
			ells[d] = ell
			o := 0
			if nu != nil {
				o = nu[d]
			}
			if o > s.K[d] {
				for a := range hs[d] {
					hs[d][a] = 0
				}
			} else {
				basis(s.T[d], x, s.K[d], ell, o, hs[d])
			}
		}
		if nan {
			dst[p] = math.NaN()
			continue
		}
		dst[p] = s.contract(hs, ells, strides, 0, 0)
	}
}

// contract sums the tensor product from dimension d on with flat offset off.
func (s *NdBSpline) contract(hs [][]float64, ells, strides []int, d, off int) float64 {
	k := s.K[d]
	sum := 0.0
	if d == len(s.T)-1 {
		for a := 0; a <= k; a++ {
			sum += float64(s.C[off+ells[d]-k+a] * hs[d][a])
		}
		return sum
	}
	for a := 0; a <= k; a++ {
		if hs[d][a] == 0 {
			continue
		}
		sum += float64(hs[d][a] * s.contract(hs, ells, strides, d+1, off+(ells[d]-k+a)*strides[d]))
	}
	return sum
}

// RGIOptions are the optional arguments of NewRGI.
type RGIOptions struct {
	// BoundsError makes Eval fail with ErrBounds for a point outside the
	// grid (scipy's default bounds_error=True; here the zero value is
	// false, so set it to ask for the error).
	BoundsError bool
	// FillValue is returned outside the grid; nil means NaN. Ignored when
	// Extrapolate is set.
	FillValue *float64
	// Extrapolate extrapolates instead of filling (scipy fill_value=None).
	Extrapolate bool
}

// RGI is scipy.interpolate.RegularGridInterpolator on a rectilinear grid.
type RGI struct {
	axes   [][]float64
	shape  []int
	values []float64
	method string
	spline *NdBSpline
	opt    RGIOptions
}

// NewRGI builds the interpolator. Methods: "linear", "nearest", "slinear",
// "cubic" (not-a-knot tensor spline) and "quintic". Axes must be strictly
// increasing; values are row-major (last axis fastest).
func NewRGI(axes [][]float64, values []float64, method string, o *RGIOptions) (*RGI, error) {
	if o == nil {
		o = &RGIOptions{}
	}
	shape, err := checkGrid(axes, values)
	if err != nil {
		return nil, err
	}
	r := &RGI{axes: axes, shape: shape, values: values, method: method, opt: *o}
	k := 0
	switch method {
	case "linear", "nearest":
	case "slinear":
		k = 1
	case "cubic":
		k = 3
	case "quintic":
		k = 5
	default:
		return nil, ErrBC
	}
	if k > 0 {
		r.spline, err = MakeNdInterp(axes, values, k)
		if err != nil {
			return nil, err
		}
	}
	return r, nil
}

// Eval interpolates at the points xi (row-major, len(dst) points of one
// coordinate per axis) into dst.
func (r *RGI) Eval(dst, xi []float64) error {
	nd := len(r.axes)
	if len(xi) != len(dst)*nd {
		return ErrShape
	}
	fill := math.NaN()
	if r.opt.FillValue != nil {
		fill = *r.opt.FillValue
	}
	oob := make([]bool, len(dst))
	for p := range dst {
		for d := 0; d < nd; d++ {
			x := xi[p*nd+d]
			if x < r.axes[d][0] || x > r.axes[d][len(r.axes[d])-1] {
				if r.opt.BoundsError {
					return ErrBounds
				}
				oob[p] = true
			}
		}
	}
	if r.spline != nil {
		r.spline.EvalNu(dst, xi, nil)
	} else if nd == 2 && r.method == "linear" {
		r.evalLinear2D(dst, xi)
	} else {
		r.evalLinearNearest(dst, xi)
	}
	if !r.opt.Extrapolate {
		for p := range dst {
			if oob[p] {
				dst[p] = fill
			}
		}
	}
	return nil
}

// axisGuess is the number of grid intervals per unit length of axis g, or 0
// when that ratio does not fit a float (then no guess is made).
func axisGuess(g []float64) float64 {
	n := len(g)
	s := float64(n-1) / (g[n-1] - g[0])
	if !(s > 0 && s < math.Inf(1)) {
		return 0
	}
	return s
}

// axisInterval is the interval evalLinearNearest uses on axis g for x (not
// NaN): sort.SearchFloat64s(g, x)-1 clamped to [0, len(g)-2], that is the
// i with g[i] < x <= g[i+1] inside the axis. It looks first at the
// proportional position x would have on a uniform axis (s intervals per
// unit length) and bisects only when a few steps do not reach it.
func axisInterval(g []float64, x, s float64) int {
	n := len(g)
	if !(x > g[0]) {
		return 0
	}
	if x > g[n-1] {
		return n - 2
	}
	i := min(max(int((x-g[0])*s), 0), n-2)
	for step := 0; step < 4; step++ {
		switch {
		case !(g[i] < x):
			i--
		case !(x <= g[i+1]):
			i++
		default:
			return i
		}
	}
	return sort.SearchFloat64s(g, x) - 1
}

// evalLinear2D is evalLinearNearest for linear interpolation on two axes,
// unrolled: the same interval, the same normalised distances and the same
// weighted sum in the same order (corners 00, 01, 10, 11; each weight
// built as 1*(1-t)*(1-u) and so on), so the same bits.
func (r *RGI) evalLinear2D(dst, xi []float64) {
	g0, g1 := r.axes[0], r.axes[1]
	s0, s1 := axisGuess(g0), axisGuess(g1)
	n1 := r.shape[1]
	v := r.values
	xi = xi[:2*len(dst)]
	for p := range dst {
		x0, x1 := xi[2*p], xi[2*p+1]
		if x0 != x0 || x1 != x1 {
			dst[p] = math.NaN()
			continue
		}
		i0 := axisInterval(g0, x0, s0)
		i1 := axisInterval(g1, x1, s1)
		t := (x0 - g0[i0]) / (g0[i0+1] - g0[i0])
		u := (x1 - g1[i1]) / (g1[i1+1] - g1[i1])
		lo := v[i0*n1+i1 : i0*n1+i1+2 : i0*n1+i1+2]
		hi := v[(i0+1)*n1+i1 : (i0+1)*n1+i1+2 : (i0+1)*n1+i1+2]
		w0 := float64(1 * (1 - t))
		w1 := float64(1 * t)
		value := 0.0
		value += float64(lo[0] * float64(w0*(1-u)))
		value += float64(lo[1] * float64(w0*u))
		value += float64(hi[0] * float64(w1*(1-u)))
		value += float64(hi[1] * float64(w1*u))
		dst[p] = value
	}
}

func (r *RGI) evalLinearNearest(dst, xi []float64) {
	nd := len(r.axes)
	idx := make([]int, nd)
	nd0 := make([]float64, nd)
	strides := make([]int, nd)
	stride := 1
	for d := nd - 1; d >= 0; d-- {
		strides[d] = stride
		stride *= r.shape[d]
	}
	for p := range dst {
		nan := false
		for d := 0; d < nd; d++ {
			x := xi[p*nd+d]
			if math.IsNaN(x) {
				nan = true
				break
			}
			g := r.axes[d]
			i := sort.SearchFloat64s(g, x) - 1
			i = min(max(i, 0), len(g)-2)
			idx[d] = i
			nd0[d] = (x - g[i]) / (g[i+1] - g[i])
		}
		if nan {
			dst[p] = math.NaN()
			continue
		}
		if r.method == "nearest" {
			off := 0
			for d := 0; d < nd; d++ {
				i := idx[d]
				if !(nd0[d] <= 0.5) {
					i++
				}
				off += i * strides[d]
			}
			dst[p] = r.values[off]
			continue
		}
		value := 0.0
		for c := 0; c < 1<<nd; c++ {
			off := 0
			weight := 1.0
			for d := 0; d < nd; d++ {
				if c>>(nd-1-d)&1 == 0 {
					off += idx[d] * strides[d]
					weight *= 1 - nd0[d]
				} else {
					off += (idx[d] + 1) * strides[d]
					weight *= nd0[d]
				}
			}
			value += float64(r.values[off] * weight)
		}
		dst[p] = value
	}
}

// Interpn is scipy.interpolate.interpn: one-shot interpolation of values on
// the grid axes at the points xi. Methods as NewRGI.
func Interpn(axes [][]float64, values []float64, xi []float64, method string, o *RGIOptions) ([]float64, error) {
	r, err := NewRGI(axes, values, method, o)
	if err != nil {
		return nil, err
	}
	if len(axes) == 0 || len(xi)%len(axes) != 0 {
		return nil, ErrShape
	}
	out := make([]float64, len(xi)/len(axes))
	return out, r.Eval(out, xi)
}

// Bicubic is scipy.interpolate.RectBivariateSpline with s = 0: the
// interpolating tensor-product spline of degrees (kx, ky) of z[i*ny+j] on
// the grid x[i], y[j].
type Bicubic struct {
	s *NdBSpline
	// gx, gy guess the knot interval of a coordinate from its position in
	// the base interval (intervals per unit length); see guessInterval.
	gx, gy float64
}

// guessInterval is findKnotInterval for x inside [t[k], t[n]) found from a
// proportional guess g*(x - t[k]) corrected by a few steps — a handful of
// well-predicted compares on the nearly uniform grids of interpolation,
// where a bisection of scattered queries mispredicts at every level. The
// interval with t[l] <= x < t[l+1] is unique, so the answer is
// findKnotInterval's; anything else (x outside, NaN, the right end, a guess
// too far off) is handed to findKnotInterval itself.
func guessInterval(t []float64, k int, x, g float64) int {
	n := len(t) - k - 1
	if x >= t[k] && x < t[n] {
		l := k + int((x-t[k])*g)
		l = min(max(l, k), n-1)
		for step := 0; step < 4; step++ {
			switch {
			case x < t[l]:
				l--
			case x >= t[l+1]:
				l++
			default:
				return l
			}
		}
	}
	return findKnotInterval(t, k, x, k)
}

// NewBicubic builds the interpolant; kx = ky = 3 is the bicubic spline (also
// MATLAB interp2 "spline").
func NewBicubic(x, y, z []float64, kx, ky int) (*Bicubic, error) {
	s, err := MakeNdInterp([][]float64{x, y}, z, kx, ky)
	if err != nil {
		return nil, err
	}
	g := func(d int) float64 {
		t, k := s.T[d], s.K[d]
		n := len(t) - k - 1
		return float64(n-k) / (t[n] - t[k])
	}
	return &Bicubic{s: s, gx: g(0), gy: g(1)}, nil
}

// clamp moves a coordinate into the knot range, as FITPACK's bispev does:
// outside the grid the interpolant is constant along that axis.
func (b *Bicubic) clamp(x, y float64) (float64, float64) {
	cl := func(v float64, t []float64, k int) float64 {
		if math.IsNaN(v) {
			return v
		}
		return math.Min(math.Max(v, t[k]), t[len(t)-k-1])
	}
	return cl(x, b.s.T[0], b.s.K[0]), cl(y, b.s.T[1], b.s.K[1])
}

// Deriv evaluates the partial derivative of order dx in x and dy in y at
// (x, y); outside the grid the point is first clamped to it.
func (b *Bicubic) Deriv(x, y float64, dx, dy int) float64 {
	x, y = b.clamp(x, y)
	s := b.s
	kx, ky := s.K[0], s.K[1]
	if kx >= maxStackK || ky >= maxStackK {
		var d [1]float64
		s.EvalNu(d[:], []float64{x, y}, []int{dx, dy})
		return d[0]
	}
	if math.IsNaN(x) || math.IsNaN(y) {
		return math.NaN()
	}
	// The two-dimensional case of NdBSpline.EvalNu without its slices and
	// recursion: the same basis values, the same sums in the same order with
	// the same roundings, so the same bits.
	ex := guessInterval(s.T[0], kx, x, b.gx)
	ey := guessInterval(s.T[1], ky, y, b.gy)
	if kx == 3 && ky == 3 && dx == 0 && dy == 0 {
		return bicubicAt(s.T[0], s.T[1], s.C, s.Shape[1], x, y, ex, ey)
	}
	var hx, hy [maxStackK]float64
	if dx <= kx {
		basis(s.T[0], x, kx, ex, dx, hx[:kx+1])
	}
	if dy <= ky {
		basis(s.T[1], y, ky, ey, dy, hy[:ky+1])
	}
	ny := s.Shape[1]
	sum := 0.0
	for a := 0; a <= kx; a++ {
		if hx[a] == 0 {
			continue
		}
		row := s.C[(ex-kx+a)*ny+ey-ky:]
		inner := 0.0
		for c := 0; c <= ky; c++ {
			inner += float64(row[c] * hy[c])
		}
		sum += float64(hx[a] * inner)
	}
	return sum
}

// cubicBasis is basis(t, x, 3, ell, 0, h) unrolled into scalars, with the
// same operations in the same order (see evalUnrolled).
func cubicBasis(t []float64, x float64, ell int) (h0, h1, h2, h3 float64) {
	l1, r1 := deBoorStep(1, t[ell+1], t[ell], x)
	h0, h1 = 0+l1, r1
	la, ra := deBoorStep(h0, t[ell+1], t[ell-1], x)
	lb, rb := deBoorStep(h1, t[ell+2], t[ell], x)
	h0, h1, h2 = 0+la, ra+lb, rb
	la, ra = deBoorStep(h0, t[ell+1], t[ell-2], x)
	lb, rb = deBoorStep(h1, t[ell+2], t[ell-1], x)
	lc, rc := deBoorStep(h2, t[ell+3], t[ell], x)
	return 0 + la, ra + lb, rb + lc, rc
}

// bicubicAt is Deriv's sum for kx = ky = 3 and no derivative, unrolled: the
// same basis values and the same sums in the same order (zero x-weights
// skipped), so the same bits.
func bicubicAt(tx, ty, c []float64, ny int, x, y float64, ex, ey int) float64 {
	x0, x1, x2, x3 := cubicBasis(tx, x, ex)
	y0, y1, y2, y3 := cubicBasis(ty, y, ey)
	row := func(a int, hx float64, sum float64) float64 {
		if hx == 0 {
			return sum
		}
		r := c[(ex-3+a)*ny+ey-3:]
		r = r[:4:4]
		inner := 0.0
		inner += float64(r[0] * y0)
		inner += float64(r[1] * y1)
		inner += float64(r[2] * y2)
		inner += float64(r[3] * y3)
		return sum + float64(hx*inner)
	}
	sum := 0.0
	sum = row(0, x0, sum)
	sum = row(1, x1, sum)
	sum = row(2, x2, sum)
	sum = row(3, x3, sum)
	return sum
}

// maxStackK bounds the degrees evaluated with stack buffers (higher degrees
// take the general path).
const maxStackK = 8

// Ev evaluates the interpolant at the points (xs[i], ys[i]) into dst
// (scipy RectBivariateSpline.ev), clamping coordinates outside the grid.
func (b *Bicubic) Ev(dst, xs, ys []float64) error {
	if len(xs) != len(ys) || len(dst) != len(xs) {
		return ErrShape
	}
	for i := range xs {
		dst[i] = b.Deriv(xs[i], ys[i], 0, 0)
	}
	return nil
}

// EvalGrid evaluates the partial derivatives (dx, dy) on the cartesian
// product of xs and ys into dst (len(xs)*len(ys), row-major), clamping
// coordinates outside the grid.
func (b *Bicubic) EvalGrid(dst, xs, ys []float64, dx, dy int) error {
	if len(dst) != len(xs)*len(ys) {
		return ErrShape
	}
	for i, x := range xs {
		for j, y := range ys {
			dst[i*len(ys)+j] = b.Deriv(x, y, dx, dy)
		}
	}
	return nil
}

// At evaluates the interpolant at one point.
func (b *Bicubic) At(x, y float64) float64 { return b.Deriv(x, y, 0, 0) }
