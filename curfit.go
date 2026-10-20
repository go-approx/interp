package interp

import "math"

// This file ports FITPACK's curfit/fpcurf (P. Dierckx), the routine behind
// scipy.interpolate.splrep and UnivariateSpline, from SciPy's C translation
// scipy/interpolate/src/dfitpack.c (SciPy 1.18.1, BSD-3-Clause, Copyright (c)
// 2001-2002 Enthought, Inc., 2003-2026 SciPy Developers; FITPACK by Paul
// Dierckx, distributed with SciPy under the same licence). The routines
// fpcurf, fpknot, fpback, fpgivs, fprota, fpdisc and fprati keep their order
// of operations, so the knots are placed exactly where the legacy routine
// places them: within a pass every new knot splits the residual sum of its
// interval in proportion to the number of data points (fpknot), and the
// least-squares problem is solved once per pass by Givens rotations, row by
// row. Arrays are stored row-major (a row of the band holds its k+1 or k+2
// entries next to each other) where FITPACK stores them column-major.

// Splrep returns the smoothing spline of degree k (1 <= k <= 5) for data
// (x, y) exactly as FITPACK's curfit computes it for
// scipy.interpolate.splrep(x, y, w, xb, xe, k, s=s) (task 0): the knots are
// placed by fpknot, one least-squares solve per pass, and the smoothing
// parameter p is then found by rational interpolation. s == 0 interpolates.
// x must be non-decreasing; o may give weights W, the interval Bounds and
// Nest (default max(len(x)+k+1, 2k+3), the value splrep uses); user knots
// (T) are not accepted, use MakeLsqSpline or MakeSplrep for them.
//
// Ier carries FITPACK's status: 0 a smoothing spline with fp within 0.1% of
// s, -1 the interpolating spline, -2 the least-squares polynomial, 1 nest
// was too small for s, 2 and 3 as for MakeSplrep. Splrep differs from
// MakeSplrep (make_splrep) in where it places the knots, which is why both
// exist: MakeSplrep re-solves after every knot, Splrep is the faster legacy
// algorithm.
func Splrep(x, y []float64, k int, s float64, o *SplrepOptions) (*Smoothing, error) {
	if o == nil {
		o = &SplrepOptions{}
	}
	m := len(x)
	if k < 1 || k > 5 || m < k+1 || len(y) != m || (o.W != nil && len(o.W) != m) || o.T != nil {
		return nil, ErrShape
	}
	if !finiteAll(x) || !finiteAll(y) || !finiteAll(o.W) || math.IsNaN(s) || math.IsInf(s, 0) {
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
	}
	xb, xe := x[0], x[m-1]
	if o.Bounds != nil {
		xb, xe = o.Bounds[0], o.Bounds[1]
	}
	k1 := k + 1
	nest := o.Nest
	if nest == 0 {
		nest = max(m+k1, 2*k+3)
	}
	if s < 0 || nest < 2*k1 || (s == 0 && nest < m+k1) || !(xb <= x[0]) || !(xe >= x[m-1]) || !(xb < xe) {
		return nil, ErrShape
	}
	return fpcurf(x, y, w, xb, xe, k, s, nest), nil
}

// fpgivs is FITPACK's Givens rotation: it returns (cos, sin) and the new
// pivot that replaces ww.
func fpgivs(piv, ww float64) (cs, sn, dd float64) {
	store := math.Abs(piv)
	if store >= ww {
		r := ww / piv
		dd = store * math.Sqrt(1+float64(r*r))
	} else {
		r := piv / ww
		dd = ww * math.Sqrt(1+float64(r*r))
	}
	return ww / dd, piv / dd, dd
}

// givens is fpgivs with a shorter dependency chain where squaring cannot
// overflow or lose the pivot: dd = sqrt(piv^2 + ww^2) and one reciprocal,
// instead of a division, a square root and two divisions. ww >= 0 is the
// diagonal entry of the triangle. The result differs from fpgivs by
// rounding only.
func givens(piv, ww float64) (cs, sn, dd float64) {
	if ap := math.Abs(piv); ap > 1e-100 && ap < 1e100 && ww < 1e100 {
		dd = math.Sqrt(float64(piv*piv) + float64(ww*ww))
		inv := 1 / dd
		return ww * inv, piv * inv, dd
	}
	return fpgivs(piv, ww)
}

// fpback solves the upper triangular band system a*c = z of n rows and
// bandwidth bw (row-major, stride bw); c may alias z.
func fpback(a, z []float64, n, bw int, c []float64) {
	c[n-1] = z[n-1] / a[(n-1)*bw]
	for i := n - 2; i >= 0; i-- {
		store := z[i]
		i1 := min(n-1-i, bw-1)
		row := a[i*bw : i*bw+bw]
		for l := 1; l <= i1; l++ {
			store -= float64(c[i+l] * row[l])
		}
		c[i] = store / row[0]
	}
}

// fpknotAdd is FITPACK's fpknot (istart = 1): it adds one knot inside the
// interval with the largest residual sum that holds data points, splitting
// that sum in proportion to the points on each side, and returns the new n
// and nrint.
func fpknotAdd(x, t []float64, n int, fpint []float64, nrdata []int, nrint int) (int, int) {
	k := (n - nrint - 1) / 2
	fpmax := 0.0
	jbegin := 1
	number, maxpt, maxbeg := 0, 0, 0
	for j := 1; j <= nrint; j++ {
		jpoint := nrdata[j-1]
		if !(fpmax >= fpint[j-1] || jpoint == 0) {
			fpmax = fpint[j-1]
			number, maxpt, maxbeg = j, jpoint, jbegin
		}
		jbegin += jpoint + 1
	}
	if number == 0 { // FITPACK's error exit: n and nrint grow, nothing moves
		return n + 1, nrint + 1
	}
	ihalf := maxpt/2 + 1
	nrx := maxbeg + ihalf
	next := number + 1
	for j := next; j <= nrint; j++ {
		jj := next + nrint - j
		fpint[jj] = fpint[jj-1]
		nrdata[jj] = nrdata[jj-1]
		t[jj+k] = t[jj+k-1]
	}
	nrdata[number-1] = ihalf - 1
	nrdata[next-1] = maxpt - ihalf
	am := float64(maxpt)
	fpint[number-1] = fpmax * float64(nrdata[number-1]) / am
	fpint[next-1] = fpmax * float64(nrdata[next-1]) / am
	t[next+k-1] = x[nrx-1]
	return n + 1, nrint + 1
}

// fpdisc fills b (row-major, stride k+2) with the jumps of the k-th
// derivative of the B-splines at the interior knots t[k+1..n-k-2].
func fpdisc(t []float64, n, k int, b []float64) {
	k1, k2 := k+1, k+2
	nk1 := n - k1
	fac := float64(nk1-k) / (t[nk1] - t[k1-1])
	var h [12]float64
	for l := k2; l <= nk1; l++ {
		lmk := l - k1
		for j := 1; j <= k1; j++ {
			h[j-1] = t[l-1] - t[l+j-k2-1]
			h[j+k1-1] = t[l-1] - t[l+j-1]
		}
		lp := lmk
		for j := 1; j <= k2; j++ {
			prod := h[j-1]
			for i := 1; i <= k; i++ {
				prod = float64(prod*h[j+i-1]) * fac
			}
			b[(lmk-1)*k2+j-1] = (t[lp+k1-1] - t[lp-1]) / prod
			lp++
		}
	}
}

// interpKnots places the interior knots of the interpolating spline as
// fpcurf does (data points for odd k, midpoints for even k).
func interpKnots(x, t []float64, k int) {
	m := len(x)
	k1 := k + 1
	j := k/2 + 1 // 0-based index of x(k3+2)
	for i := 0; i < m-k1; i++ {
		if k%2 == 0 {
			t[k1+i] = float64(x[j+i]+x[j+i-1]) * 0.5
		} else {
			t[k1+i] = x[j+i]
		}
	}
}

// recips stores the reciprocals of the knot differences t[ell+i]-t[ell+i-j]
// (j = 1..k, i = 1..j, in that order) that the de Boor-Cox recursion
// divides by on knot interval ell, 0 for a zero difference. All the points
// of one interval share them, so the recursion multiplies instead of
// dividing: one rounding more per step than fpbspl, and no division chain.
func recips(t []float64, k, ell int, r []float64) {
	idx := 0
	for j := 1; j <= k; j++ {
		for i := 1; i <= j; i++ {
			d := t[ell+i] - t[ell+i-j]
			r[idx] = 0
			if d != 0 {
				r[idx] = 1 / d
			}
			idx++
		}
	}
}

// basisRecip is fpbspl (basis with nu = 0) with the reciprocals of recips.
func basisRecip(t []float64, x float64, k, ell int, r, h []float64) {
	switch k {
	case 3:
		t0, t1, t2, t3 := t[ell-2], t[ell-1], t[ell], t[ell+1]
		t4, t5 := t[ell+2], t[ell+3]
		// j = 1
		f := r[0]
		h0 := float64(f * (t3 - x))
		h1 := float64(f * (x - t2))
		// j = 2
		f = h0 * r[1]
		g0 := float64(f * (t3 - x))
		g1 := float64(f * (x - t1))
		f = h1 * r[2]
		g1 += float64(f * (t4 - x))
		g2 := float64(f * (x - t2))
		// j = 3
		f = g0 * r[3]
		h[0] = float64(f * (t3 - x))
		e1 := float64(f * (x - t0))
		f = g1 * r[4]
		h[1] = e1 + float64(f*(t4-x))
		e2 := float64(f * (x - t1))
		f = g2 * r[5]
		h[2] = e2 + float64(f*(t5-x))
		h[3] = float64(f * (x - t2))
	default:
		h[0] = 1
		idx := 0
		for j := 1; j <= k; j++ {
			old := h[0]
			h[0] = 0
			for n := 1; n <= j; n++ {
				next := 0.0
				if n < j {
					next = h[n]
				}
				w := old * r[idx]
				idx++
				h[n-1] += float64(w * (t[ell+n] - x))
				h[n] = float64(w * (x - t[ell+n-j]))
				old = next
			}
		}
	}
}

// givensLsq is the first half of an fpcurf pass: the least-squares spline
// on knots t (len n) reduced row by row by Givens rotations into the band a
// (row-major, stride k+1) and right-hand side z, which it clears first. It
// stores the unweighted B-spline values of every point in q (row-major,
// stride k+1) and returns fp, the sum of squared residuals.
func givensLsq(x, y, w, t []float64, k int, a, z, q []float64) float64 {
	k1 := k + 1
	nk1 := len(t) - k1
	clear(z[:nk1])
	clear(a[:nk1*k1])
	var h [8]float64
	var rc [15]float64
	fp := 0.0
	ell := k
	recips(t, k, ell, rc[:])
	for it := range x {
		xi, wi := x[it], w[it]
		yi := y[it] * wi
		if !(xi < t[ell+1] || ell == nk1-1) {
			for !(xi < t[ell+1] || ell == nk1-1) {
				ell++
			}
			recips(t, k, ell, rc[:])
		}
		qr := q[it*k1 : it*k1+k1]
		basisRecip(t, xi, k, ell, rc[:], qr)
		for i := 0; i < k1; i++ {
			h[i] = qr[i] * wi
		}
		for i := 0; i < k1; i++ {
			piv := h[i]
			if piv == 0 {
				continue
			}
			row := ell - k + i
			ar := a[row*k1 : row*k1+k1]
			cs, sn, dd := givens(piv, ar[0])
			ar[0] = dd
			yi, z[row] = float64(cs*yi)-float64(sn*z[row]), float64(cs*z[row])+float64(sn*yi)
			for i1, i2 := i+1, 1; i1 < k1; i1, i2 = i1+1, i2+1 {
				h[i1], ar[i2] = float64(cs*h[i1])-float64(sn*ar[i2]), float64(cs*ar[i2])+float64(sn*h[i1])
			}
		}
		fp += float64(yi * yi)
	}
	return fp
}

// fpcurf is FITPACK's fpcurf for iopt = 0 on validated input.
func fpcurf(x, y, w []float64, xb, xe float64, k int, s float64, nest int) *Smoothing {
	const tol, maxit = 1e-3, 20
	const con1, con9, con4 = 0.1, 0.9, 0.04
	m := len(x)
	k1, k2 := k+1, k+2
	nmin, nmax := 2*k1, m+k1
	acc := tol * s
	t := make([]float64, nest)
	c := make([]float64, nest)
	z := make([]float64, nest)
	fpint := make([]float64, nest)
	nrdata := make([]int, nest)
	a := make([]float64, nest*k1)
	q := make([]float64, m*k1)
	var n, nplus, ier int
	var fp, fpold, fp0, fpms float64
	if s > 0 {
		n = nmin
		nrdata[0] = m - 2
	} else {
		n = nmax
		interpKnots(x, t, k)
	}
	var h [8]float64
	result :=func(p float64) *Smoothing {
		nk1 := n - k1
		tt := append([]float64(nil), t[:n]...)
		cc := append([]float64(nil), c[:nk1]...)
		return &Smoothing{Spline: &BSpline{T: tt, C: cc, K: k}, Fp: fp, Ier: ier, P: p}
	}
	var nk1 int
	for iter := 1; iter <= m; iter++ {
		if n == nmin {
			ier = -2
		}
		nrint := n - nmin + 1
		nk1 = n - k1
		for j := 0; j < k1; j++ {
			t[j] = xb
			t[n-1-j] = xe
		}
		fp = givensLsq(x, y, w, t[:n], k, a, z, q)
		if ier == -2 {
			fp0 = fp
		}
		fpback(a, z, nk1, k1, c)
		fpms = fp - s
		if math.Abs(fpms) < acc {
			return result(0)
		}
		if fpms < 0 {
			break
		}
		if n == nmax {
			ier = -1
			return result(0)
		}
		if n == nest {
			ier = 1
			return result(0)
		}
		if ier == 0 {
			npl1 := nplus * 2
			if fpold-fp > acc {
				val := float64(nplus) * fpms / (fpold - fp)
				npl1 = math.MaxInt32
				if val <= math.MaxInt32 {
					npl1 = int(val)
				}
			}
			nplus = min(nplus*2, max(npl1, nplus/2, 1))
		} else {
			nplus = 1
			ier = 0
		}
		fpold = fp
		// Residual sums per knot interval; a point on a knot counts half on
		// each side.
		fpart := 0.0
		i := 0
		lk := k + 1
		for it := 0; it < m; it++ {
			nw := false
			if !(x[it] < t[lk] || lk >= nk1) {
				nw = true
				lk++
			}
			cr := c[lk-k1 : lk]
			qr := q[it*k1 : it*k1+k1]
			term := 0.0
			for j := range qr {
				term += float64(cr[j] * qr[j])
			}
			term = w[it] * (term - y[it])
			term *= term
			fpart += term
			if nw {
				store := term * 0.5
				fpint[i] = fpart - store
				i++
				fpart = store
			}
		}
		fpint[nrint-1] = fpart
		for ll := 1; ll <= nplus; ll++ {
			n, nrint = fpknotAdd(x, t, n, fpint, nrdata, nrint)
			if n == nmax {
				interpKnots(x, t, k)
				iter = 0 // FITPACK restarts the loop for the new knots
				break
			}
			if n == nest {
				break
			}
		}
	}
	if ier == -2 {
		return result(0)
	}
	// Part 2: the smoothing spline, rows of the discontinuity matrix with
	// weight 1/p rotated into the triangle, p found by rational interpolation.
	b := make([]float64, nest*k2)
	g := make([]float64, nest*k2)
	fpdisc(t, n, k, b)
	p1, f1 := 0.0, fp0-s
	p3, f3 := -1.0, fpms
	p := 0.0
	for i := 0; i < nk1; i++ {
		p += a[i*k1]
	}
	p = float64(nk1) / p
	ich1, ich3 := 0, 0
	n8 := n - nmin
	for iter := 1; ; iter++ {
		pinv := 1 / p
		for i := 0; i < nk1; i++ {
			c[i] = z[i]
			gr := g[i*k2 : i*k2+k2]
			copy(gr, a[i*k1:i*k1+k1])
			gr[k2-1] = 0
		}
		for it := 1; it <= n8; it++ {
			br := b[(it-1)*k2 : (it-1)*k2+k2]
			for i := range br {
				h[i] = br[i] * pinv
			}
			yi := 0.0
			for j := it; j <= nk1; j++ {
				gr := g[(j-1)*k2 : (j-1)*k2+k2]
				cs, sn, dd := fpgivs(h[0], gr[0])
				gr[0] = dd
				yi, c[j-1] = float64(cs*yi)-float64(sn*c[j-1]), float64(cs*c[j-1])+float64(sn*yi)
				if j == nk1 {
					break
				}
				i2 := k1
				if j > n8 {
					i2 = nk1 - j
				}
				for i := 1; i <= i2; i++ {
					h[i], gr[i] = float64(cs*h[i])-float64(sn*gr[i]), float64(cs*gr[i])+float64(sn*h[i])
					h[i-1] = h[i]
				}
				h[i2] = 0
			}
		}
		fpback(g, c, nk1, k2, c)
		fp = 0
		lk := k + 1
		for it := 0; it < m; it++ {
			if !(x[it] < t[lk] || lk >= nk1) {
				lk++
			}
			cr := c[lk-k1 : lk]
			qr := q[it*k1 : it*k1+k1]
			term := 0.0
			for j := range qr {
				term += float64(cr[j] * qr[j])
			}
			term = w[it] * (term - y[it])
			fp += float64(term * term)
		}
		fpms = fp - s
		if math.Abs(fpms) < acc {
			return result(p)
		}
		if iter == maxit {
			ier = 3
			return result(p)
		}
		p2, f2 := p, fpms
		if ich3 == 0 {
			if f2-f3 > acc {
				if f2 < 0 {
					ich3 = 1
				}
			} else {
				p3, f3 = p2, f2
				p *= con4
				if p <= p1 {
					p = float64(p1*con9) + float64(p2*con1)
				}
				continue
			}
		}
		if ich1 == 0 {
			if f1-f2 > acc {
				if f2 > 0 {
					ich1 = 1
				}
			} else {
				p1, f1 = p2, f2
				p /= con4
				if p3 >= 0 && p >= p3 {
					p = float64(p2*con1) + float64(p3*con9)
				}
				continue
			}
		}
		if f2 >= f1 || f2 <= f3 {
			ier = 2
			return result(p)
		}
		p = fpratiFitpack(p1, f1, p2, f2, p3, f3)
		if f2 < 0 {
			p3, f3 = p2, f2
		} else {
			p1, f1 = p2, f2
		}
	}
}

// fpratiFitpack is FITPACK's fprati with p3 <= 0 standing for infinity.
func fpratiFitpack(p1, f1, p2, f2, p3, f3 float64) float64 {
	if p3 > 0 {
		h1 := f1 * (f2 - f3)
		h2 := f2 * (f3 - f1)
		h3 := f3 * (f1 - f2)
		return -(float64(float64(p1*p2)*h3) + float64(float64(p2*p3)*h1) + float64(float64(p3*p1)*h2)) /
			(float64(p1*h1) + float64(p2*h2) + float64(p3*h3))
	}
	return (float64(float64(p1*(f1-f3))*f2) - float64(float64(p2*(f2-f3))*f1)) / ((f1 - f2) * f3)
}
