package interp

import (
	"math"
	"sort"
)

// Kind selects the rule of a piecewise-constant or linear interpolant.
type Kind int

const (
	// Linear is piecewise linear interpolation.
	Linear Kind = iota
	// Nearest picks the nearest sample, ties going to the lower one
	// (scipy "nearest").
	Nearest
	// NearestUp picks the nearest sample, ties going to the higher one
	// (scipy "nearest-up"; MATLAB "nearest").
	NearestUp
	// Previous picks the sample at or before the query ("previous").
	Previous
	// Next picks the sample at or after the query ("next").
	Next
)

// Interp1Options are the optional arguments of NewInterp1D.
type Interp1Options struct {
	// AssumeSorted skips the (stable) sort of x.
	AssumeSorted bool
	// BoundsError makes EvalAll fail with ErrBounds for a query outside
	// [x[0], x[n-1]]; it is ignored when Extrapolate is set.
	BoundsError bool
	// Fill gives the value returned below and above the data: nil means NaN,
	// one element serves for both sides, two are (below, above).
	Fill []float64
	// Extrapolate continues the rule beyond the data (linear: the end
	// segments; nearest: the end samples; previous: NaN below, last sample
	// above; next: first sample below, NaN above), like scipy
	// fill_value="extrapolate".
	Extrapolate bool
}

// Interp1D is scipy.interpolate.interp1d for the kinds linear, nearest,
// nearest-up, previous and next, and MATLAB interp1 for the same methods.
type Interp1D struct {
	x, y        []float64
	kind        Kind
	bds         []float64 // nearest: midpoints; previous/next: shifted x
	below       float64
	above       float64
	extrapolate bool
	boundsError bool
}

// NewInterp1D builds the interpolant. x may be unsorted unless AssumeSorted;
// duplicates are kept (queries resolve them as scipy does). x must not hold
// NaN and the Linear kind with Extrapolate needs two points.
func NewInterp1D(x, y []float64, kind Kind, o *Interp1Options) (*Interp1D, error) {
	if o == nil {
		o = &Interp1Options{}
	}
	n := len(x)
	if n < 1 || len(y) != n {
		return nil, ErrShape
	}
	for _, v := range x {
		if math.IsNaN(v) {
			return nil, ErrNonFinite
		}
	}
	xs, ys := x, y
	if !o.AssumeSorted {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return x[idx[a]] < x[idx[b]] })
		xs, ys = make([]float64, n), make([]float64, n)
		for i, j := range idx {
			xs[i], ys[i] = x[j], y[j]
		}
	}
	f := &Interp1D{x: xs, y: ys, kind: kind, below: math.NaN(), above: math.NaN(),
		extrapolate: o.Extrapolate, boundsError: o.BoundsError}
	switch len(o.Fill) {
	case 0:
	case 1:
		f.below, f.above = o.Fill[0], o.Fill[0]
	default:
		f.below, f.above = o.Fill[0], o.Fill[1]
	}
	switch kind {
	case Nearest, NearestUp:
		f.bds = make([]float64, n-1)
		for i := range f.bds {
			f.bds[i] = xs[i+1]/2 + xs[i]/2
		}
	case Previous, Next:
		dir := math.Inf(-1)
		if kind == Next {
			dir = math.Inf(1)
		}
		f.bds = make([]float64, n)
		for i := range f.bds {
			f.bds[i] = math.Nextafter(xs[i], dir)
		}
	}
	if o.Extrapolate {
		switch kind {
		case Previous:
			f.below, f.above = math.NaN(), ys[n-1]
			f.extrapolate = false
			f.boundsError = false
		case Next:
			f.below, f.above = ys[0], math.NaN()
			f.extrapolate = false
			f.boundsError = false
		}
	}
	return f, nil
}

// searchLeft is numpy.searchsorted(a, v, side="left") with NaN sorted last.
func searchLeft(a []float64, v float64) int {
	if math.IsNaN(v) {
		return len(a)
	}
	return sort.SearchFloat64s(a, v)
}

func searchRight(a []float64, v float64) int {
	if math.IsNaN(v) {
		return len(a)
	}
	return sort.Search(len(a), func(i int) bool { return a[i] > v })
}

// npInterp is numpy.interp for one point with the given left/right values.
func npInterp(v float64, xp, fp []float64, left, right float64) float64 {
	n := len(xp)
	if n == 1 { // numpy compares, so a NaN query falls through to the value
		switch {
		case v < xp[0]:
			return left
		case v > xp[0]:
			return right
		}
		return fp[0]
	}
	switch {
	case math.IsNaN(v):
		return v
	case v > xp[n-1]:
		return right
	case v < xp[0]:
		return left
	}
	j := sort.Search(n, func(i int) bool { return xp[i] > v }) - 1
	switch {
	case j == n-1:
		return fp[n-1]
	case xp[j] == v:
		return fp[j]
	}
	slope := (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
	r := math.FMA(slope, v-xp[j], fp[j])
	if math.IsNaN(r) {
		r = math.FMA(slope, v-xp[j+1], fp[j+1])
		if math.IsNaN(r) && fp[j] == fp[j+1] {
			r = fp[j]
		}
	}
	return r
}

// Interp is numpy.interp: the piecewise linear interpolant of (xp, fp), with
// left and right returned outside; xp must be increasing (not checked).
func Interp(dst, x, xp, fp []float64, left, right float64) error {
	if len(xp) == 0 || len(xp) != len(fp) || len(dst) != len(x) {
		return ErrShape
	}
	for i, v := range x {
		dst[i] = npInterp(v, xp, fp, left, right)
	}
	return nil
}

// EvalAll evaluates the interpolant at xq into dst.
func (f *Interp1D) EvalAll(dst, xq []float64) error {
	n := len(f.x)
	if f.boundsError && !f.extrapolate {
		for _, v := range xq {
			if v < f.x[0] || v > f.x[n-1] {
				return ErrBounds
			}
		}
	}
	for i, v := range xq {
		var r float64
		switch f.kind {
		case Linear:
			if f.extrapolate && n == 1 {
				r = math.NaN() // scipy: 0/0 in the single segment formula
			} else if f.extrapolate {
				idx := min(max(searchLeft(f.x, v), 1), n-1)
				lo, hi := idx-1, idx
				r = float64((v-f.x[lo])/(f.x[hi]-f.x[lo])*f.y[hi]) +
					float64((f.x[hi]-v)/(f.x[hi]-f.x[lo])*f.y[lo])
			} else {
				r = npInterp(v, f.x, f.y, f.y[0], f.y[n-1])
			}
		case Nearest:
			r = f.y[min(max(searchLeft(f.bds, v), 0), n-1)]
		case NearestUp:
			r = f.y[min(max(searchRight(f.bds, v), 0), n-1)]
		case Previous:
			r = f.y[min(max(searchLeft(f.bds, v), 1), n)-1]
		case Next:
			r = f.y[min(max(searchRight(f.bds, v), 0), n-1)]
		}
		if !f.extrapolate {
			if v < f.x[0] {
				r = f.below
			} else if v > f.x[n-1] {
				r = f.above
			}
		}
		dst[i] = r
	}
	return nil
}

// At evaluates the interpolant at one point.
func (f *Interp1D) At(x float64) (float64, error) {
	var d [1]float64
	err := f.EvalAll(d[:], []float64{x})
	return d[0], err
}

// ExtrapMode is the extrapolation argument of Interp1.
type ExtrapMode int

const (
	// ExtrapDefault is MATLAB's default: NaN outside the data for linear,
	// nearest, next and previous; extrapolation for pchip, cubic, spline and
	// makima.
	ExtrapDefault ExtrapMode = iota
	// ExtrapOn is MATLAB's "extrap".
	ExtrapOn
	// ExtrapValue returns Value outside the data.
	ExtrapValue
)

// Extrap is the extrapolation argument of Interp1.
type Extrap struct {
	Mode  ExtrapMode
	Value float64
}

// Interp1 is MATLAB's interp1(x, v, xq, method, extrap): method is one of
// "linear", "nearest", "next", "previous", "pchip", "cubic" (a synonym of
// pchip), "spline" and "makima". x is sorted first; the smooth methods need
// strictly increasing x. MATLAB's nearest rounds ties up.
func Interp1(x, v, xq []float64, method string, ex Extrap) ([]float64, error) {
	out := make([]float64, len(xq))
	var kind Kind
	smooth := false
	switch method {
	case "linear":
		kind = Linear
	case "nearest":
		kind = NearestUp
	case "next":
		kind = Next
	case "previous":
		kind = Previous
	case "pchip", "cubic", "spline", "makima":
		smooth = true
	default:
		return nil, ErrBC
	}
	if !smooth {
		o := &Interp1Options{}
		switch ex.Mode {
		case ExtrapOn:
			o.Extrapolate = true
		case ExtrapValue:
			o.Fill = []float64{ex.Value}
		}
		f, err := NewInterp1D(x, v, kind, o)
		if err != nil {
			return nil, err
		}
		return out, f.EvalAll(out, xq)
	}
	n := len(x)
	if len(v) != n {
		return nil, ErrShape
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return x[idx[a]] < x[idx[b]] })
	xs, vs := make([]float64, n), make([]float64, n)
	for i, j := range idx {
		xs[i], vs[i] = x[j], v[j]
	}
	var p *PPoly
	var err error
	switch method {
	case "pchip", "cubic":
		p, err = NewPchip(xs, vs)
	case "spline":
		p, err = NewCubicSpline(xs, vs, "not-a-knot")
	default:
		p, err = NewAkima(xs, vs, true)
		if p != nil {
			p.Extrap = Extend
		}
	}
	if err != nil {
		return nil, err
	}
	if ex.Mode == ExtrapDefault {
		ex.Mode = ExtrapOn
	}
	p.EvalAll(out, xq)
	if ex.Mode == ExtrapValue {
		for i, q := range xq {
			if q < xs[0] || q > xs[n-1] {
				out[i] = ex.Value
			}
		}
	}
	return out, nil
}
