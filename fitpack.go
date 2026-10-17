package interp

import (
	"math"
	"sort"
)

// The smoothing spline of this file follows Dierckx's algorithm for
// smoothing splines with automatic knot placement (P. Dierckx, "An algorithm
// for smoothing, differentiation and integration of experimental data using
// spline functions", J. Comput. Appl. Math. 1(3), 1975, 165-184; "Curve and
// surface fitting with splines", OUP 1993, chapter 4; the periodic and
// parametric variants are in Comput. Graphics Image Process. 20, 1982,
// 171-184). The structure -- knot growth driven by the residual sums per knot
// interval, then a rational iteration on the smoothing parameter p -- was
// checked against SciPy's pure Python replica in
// scipy/interpolate/_fitpack_repro.py (BSD-3-Clause, Copyright (c) 2001-2002
// Enthought, Inc., 2003-2026 SciPy Developers), which this Go code ports.

const (
	fitTol   = 0.001 // hard-coded in curfit.f
	fitMaxIt = 20
)

// SplrepOptions are the optional arguments of MakeSplrep.
type SplrepOptions struct {
	// W are the weights (1/standard deviation); nil means all ones.
	W []float64
	// Bounds is the fitting interval [xb, xe]; nil means [x[0], x[n-1]].
	Bounds *[2]float64
	// T are user knots; mutually exclusive with Nest.
	T []float64
	// Nest caps the number of knots; 0 means len(x)+k+1.
	Nest int
}

// Smoothing is the result of MakeSplrep.
type Smoothing struct {
	// Spline is the fitted spline.
	Spline *BSpline
	// Fp is the weighted sum of squared residuals sum((w*(y-spl(x)))^2).
	Fp float64
	// Ier is FITPACK's status: 0 success, 1 NaN residual, 2 the iteration
	// left the theoretical regime (fp does not meet |fp-s|/s < 0.001), 3 the
	// iteration limit (20) was reached.
	Ier int
	// P is the smoothing parameter of the final iteration (0 when no
	// iteration was needed).
	P float64
}

func getResiduals(x, y, w, t []float64, k int) (res []float64, fp float64, c []float64, ok bool) {
	c, ok = buildLsq(x, y, w, t, k).solve()
	if !ok {
		return nil, 0, nil, false
	}
	res, fp = residuals(x, y, w, t, k, c)
	return res, fp, c, !math.IsNaN(fp)
}

// fpknot is the position for a new knot: the data site in the middle of the
// knot interval with the largest sum of squared residuals.
func fpknot(x, t []float64, k int, res []float64) (float64, bool) {
	interval := k + 1
	nc := len(t) - k - 1
	ix := []int{0}
	var fparts []float64
	fpart := 0.0
	m := len(x)
	for i := 0; i < m; i++ {
		fpart += res[i]
		if x[i] >= t[interval] && interval < nc {
			carry := res[i] / 2
			fpart -= carry
			fparts = append(fparts, fpart)
			fpart = carry
			interval++
			ix = append(ix, i)
		}
	}
	ix = append(ix, m-1)
	fparts = append(fparts, fpart)
	imax, fmax := -1, -1e100
	for i := range fparts {
		if ix[i+1]-ix[i] > 1 && fparts[i] > fmax {
			imax, fmax = i, fparts[i]
		}
	}
	if imax < 0 {
		return 0, false
	}
	return x[(ix[imax]+ix[imax+1]+1)/2], true
}

// generateKnots grows the knot vector until the least-squares spline meets
// the smoothing condition and returns the final vector (the last one
// scipy.interpolate.generate_knots yields).
func generateKnots(x, y, w []float64, xb, xe float64, k int, s float64, nest int) ([]float64, bool) {
	acc := s * fitTol
	m := len(x)
	if nest <= 0 {
		nest = max(m+k+1, 2*k+3)
	}
	nmin, nmax := 2*(k+1), m+k+1
	t := make([]float64, 0, nmin)
	for i := 0; i <= k; i++ {
		t = append(t, xb)
	}
	for i := 0; i <= k; i++ {
		t = append(t, xe)
	}
	n := len(t)
	var fp, fpold float64
	nplus := 0
	last := t
	for iter := 0; iter < m; iter++ {
		last = append([]float64(nil), t...)
		fpold = fp
		res, f, _, ok := getResiduals(x, y, w, t, k)
		if !ok {
			return nil, false
		}
		fp = f
		fpms := fp - s
		if math.Abs(fpms) < acc || fpms < 0 {
			return last, true
		}
		if n == nmin {
			nplus = 1
		} else {
			npl1 := nplus * 2
			if delta := fpold - fp; delta > acc {
				npl1 = int(float64(nplus) * fpms / delta)
			}
			nplus = min(nplus*2, max(npl1, nplus/2, 1))
		}
		for j := 0; j < nplus; j++ {
			nk, ok := fpknot(x, t, k, res)
			if !ok {
				return nil, false
			}
			at := sort.SearchFloat64s(t, nk)
			t = append(t, 0)
			copy(t[at+1:], t[at:])
			t[at] = nk
			n = len(t)
			if n >= nmax {
				return notAKnot(x, k), true
			}
			if n >= nest {
				return t, true
			}
			if j < nplus-1 {
				res, _, _, ok = getResiduals(x, y, w, t, k)
				if !ok {
					return nil, false
				}
			}
		}
	}
	return last, true
}

// discMatrix returns the jumps of the k-th derivative of the B-splines at the
// interior knots (nrint-1 rows of k+2 entries), normalised as FITPACK does.
func discMatrix(t []float64, k int) [][]float64 {
	n := len(t)
	delta := t[n-k-1] - t[k]
	nrint := n - 2*k - 1
	prodd := func(i, j int) float64 {
		res := 1.0
		for s := 0; s < k+2; s++ {
			if i+s != j {
				res *= t[j] - t[i+s]
			}
		}
		return res
	}
	scale := math.Pow(delta/float64(nrint), float64(k))
	out := make([][]float64, nrint-1)
	for jj := range out {
		j := jj + k + 1
		row := make([]float64, k+2)
		for ii := range row {
			i := jj + ii
			row[ii] = float64((t[i+k+1]-t[i])/prodd(i, j)) * scale
		}
		out[jj] = row
	}
	return out
}

func fprati(p1, f1, p2, f2, p3, f3 float64) float64 {
	h1 := f1 * (f2 - f3)
	h2 := f2 * (f3 - f1)
	h3 := f3 * (f1 - f2)
	if math.IsInf(p3, 1) {
		return -(p2*h1 + p1*h2) / h3
	}
	return -(p1*p2*h3 + p2*p3*h1 + p1*p3*h2) / (p1*h1 + p2*h2 + p3*h3)
}

// MakeSplrep returns the smoothing spline of degree k (>= 1, odd k advised)
// for data (x, y), the FITPACK curfit/splrep problem as replicated by
// scipy.interpolate.make_splrep: the spline with the fewest knots that
// satisfies sum((w*(y-g(x)))^2) <= s, minimising the jumps of its k-th
// derivative. s == 0 interpolates (MakeInterpSpline, not-a-knot) and then
// no W, T or Nest may be given. UnivariateSpline's default smoothing is
// s = len(x).
func MakeSplrep(x, y []float64, k int, s float64, o *SplrepOptions) (*Smoothing, error) {
	if o == nil {
		o = &SplrepOptions{}
	}
	if s == 0 {
		if o.T != nil || o.W != nil || o.Nest != 0 {
			return nil, ErrBC
		}
		spl, err := MakeInterpSpline(x, y, k, nil)
		if err != nil {
			return nil, err
		}
		_, fp := residuals(x, y, ones(len(x)), spl.T, k, spl.C)
		return &Smoothing{Spline: spl, Fp: fp}, nil
	}
	m := len(x)
	if m < k+1 || len(y) != m || k < 1 || s < 0 || (o.W != nil && len(o.W) != m) || (o.T != nil && o.Nest != 0) {
		return nil, ErrShape
	}
	if !finiteAll(x) || !finiteAll(y) || !finiteAll(o.W) || math.IsNaN(s) {
		return nil, ErrNonFinite
	}
	for i := 1; i < m; i++ {
		if x[i] < x[i-1] {
			return nil, ErrNotIncreasing
		}
	}
	w := o.W
	if w == nil {
		w = ones(m)
	} else {
		sum := 0.0
		for _, v := range w {
			if v < 0 {
				return nil, ErrShape
			}
			sum += v
		}
		if sum == 0 {
			return nil, ErrShape
		}
	}
	xb, xe := x[0], x[m-1]
	if o.Bounds != nil {
		xb, xe = o.Bounds[0], o.Bounds[1]
	}
	if o.Nest != 0 && o.Nest < 2*(k+1) {
		return nil, ErrShape
	}
	t := o.T
	if t == nil {
		var ok bool
		t, ok = generateKnots(x, y, w, xb, xe, k, s, o.Nest)
		if !ok {
			return nil, ErrSingular
		}
	} else if len(t) < 2*(k+1) || !finiteAll(t) || !sort.Float64sAreSorted(t) {
		return nil, ErrKnots
	}
	nc := len(t) - k - 1
	q := buildLsq(x, y, w, t, k)
	if len(t) == 2*(k+1) {
		c, ok := q.solve()
		if !ok {
			return nil, ErrSingular
		}
		_, fp := residuals(x, y, w, t, k, c)
		return &Smoothing{Spline: &BSpline{T: t, C: c, K: k}, Fp: fp}, nil
	}
	diag := 0.0
	for j := 0; j < nc; j++ {
		diag += q.r[j*q.w]
	}
	p := float64(nc) / diag
	_, fp, _, ok := getResiduals(x, y, w, t, k)
	if !ok {
		return nil, ErrSingular
	}
	fpinf := fp - s
	t0 := make([]float64, 0, 2*(k+1))
	for i := 0; i <= k; i++ {
		t0 = append(t0, xb)
	}
	for i := 0; i <= k; i++ {
		t0 = append(t0, xe)
	}
	_, fp0, _, ok := getResiduals(x, y, w, t0, k)
	if !ok {
		return nil, ErrSingular
	}
	fp0 -= s
	b := discMatrix(t, k)
	var cur []float64
	var curFp float64
	f := func(p float64) (float64, bool) {
		qq := q.clone()
		for jj, row := range b {
			r := make([]float64, len(row))
			for i, v := range row {
				r[i] = v / p
			}
			qq.addRow(jj, r, 0)
		}
		c, ok := qq.solve()
		if !ok {
			return 0, false
		}
		_, fp := residuals(x, y, w, t, k, c)
		cur, curFp = c, fp
		return fp - s, true
	}
	acc := s * fitTol
	ier, pf, ok := rootRati(f, p, [2]float64{0, fp0}, [2]float64{math.Inf(1), fpinf}, acc)
	if !ok {
		return nil, ErrSingular
	}
	return &Smoothing{Spline: &BSpline{T: t, C: cur, K: k}, Fp: curFp, Ier: ier, P: pf}, nil
}

func ones(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 1
	}
	return w
}

// rootRati finds the root of the decreasing function f by FITPACK's rational
// interpolation r(p) = (u*p+v)/(p+w), starting from p0 inside the bracket
// (p1, f1), (p3, f3). It returns FITPACK's status, the last p and false if f
// itself failed.
func rootRati(f func(float64) (float64, bool), p0 float64, lo, hi [2]float64, acc float64) (int, float64, bool) {
	const con1, con9, con4 = 0.1, 0.9, 0.04
	ich1, ich3 := 0, 0
	p1, f1 := lo[0], lo[1]
	p3, f3 := hi[0], hi[1]
	p := p0
	for it := 0; it < fitMaxIt; it++ {
		p2 := p
		f2, ok := f(p)
		if !ok {
			return 0, p, false
		}
		if math.IsNaN(f2) {
			return 1, p, true
		}
		if math.Abs(f2) < acc {
			return 0, p, true
		}
		if ich3 == 0 {
			if f2-f3 <= acc {
				p3, f3 = p2, f2
				p *= con4
				if p <= p1 {
					p = p1*con9 + p2*con1
				}
				continue
			}
			if f2 < 0 {
				ich3 = 1
			}
		}
		if ich1 == 0 {
			if f1-f2 <= acc {
				p1, f1 = p2, f2
				p /= con4
				if !math.IsInf(p3, 1) && p <= p3 {
					p = p2*con1 + p3*con9
				}
				continue
			}
			if f2 > 0 {
				ich1 = 1
			}
		}
		if f1 <= f2 || f2 <= f3 {
			return 2, p, true
		}
		p = fprati(p1, f1, p2, f2, p3, f3)
		if f2 < 0 {
			p3, f3 = p2, f2
		} else {
			p1, f1 = p2, f2
		}
	}
	return 3, p, true
}
