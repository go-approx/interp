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
func MakeNdInterp(axes [][]float64, values []float64, k ...int) (*NdBSpline, error) {
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
		h, hh := make([]float64, k+1), make([]float64, k+1)
		for i, xv := range axes[d] {
			ell := findKnotInterval(t, k, xv, k)
			basis(t, xv, k, ell, 0, h, hh)
			for a := 0; a <= k; a++ {
				m.set(i, ell-k+a, h[a])
			}
		}
		if !m.factor() {
			return nil, ErrSingular
		}
		line := make([]float64, n)
		outer := len(c) / (n * strides[d])
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
	hh := make([]float64, maxInts(s.K)+1)
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
				basis(s.T[d], x, s.K[d], ell, o, hs[d], hh)
			}
		}
		if nan {
			dst[p] = math.NaN()
			continue
		}
		dst[p] = s.contract(hs, ells, strides, 0, 0)
	}
}

func maxInts(v []int) int {
	m := v[0]
	for _, x := range v {
		m = max(m, x)
	}
	return m
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
}

// NewBicubic builds the interpolant; kx = ky = 3 is the bicubic spline (also
// MATLAB interp2 "spline").
func NewBicubic(x, y, z []float64, kx, ky int) (*Bicubic, error) {
	s, err := MakeNdInterp([][]float64{x, y}, z, kx, ky)
	if err != nil {
		return nil, err
	}
	return &Bicubic{s: s}, nil
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
	var d [1]float64
	b.s.EvalNu(d[:], []float64{x, y}, []int{dx, dy})
	return d[0]
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
