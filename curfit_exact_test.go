package interp

import (
	"math"
	"math/big"
	"testing"
)

// The least-squares kernel of Splrep differs from FITPACK's fpcurf in two
// roundings: the de Boor-Cox recursion multiplies by per-interval
// reciprocals (basisRecip) instead of dividing (fpbspl), and the Givens
// rotation takes sqrt(piv^2+ww^2) (givens) instead of fpgivs's scaled form.
// These tests show that the result is no further from the exact
// least-squares solution than the faithful FITPACK arithmetic.

// givensLsqFitpack is givensLsq with FITPACK's own arithmetic (fpbspl via
// basis, fpgivs).
func givensLsqFitpack(x, y, w, t []float64, k int, a, z []float64) {
	k1 := k + 1
	nk1 := len(t) - k1
	clear(z[:nk1])
	clear(a[:nk1*k1])
	var h [8]float64
	ell := k
	for it := range x {
		xi, wi := x[it], w[it]
		yi := y[it] * wi
		for !(xi < t[ell+1] || ell == nk1-1) {
			ell++
		}
		basis(t, xi, k, ell, 0, h[:])
		for i := 0; i < k1; i++ {
			h[i] *= wi
		}
		for i := 0; i < k1; i++ {
			piv := h[i]
			if piv == 0 {
				continue
			}
			row := ell - k + i
			ar := a[row*k1 : row*k1+k1]
			cs, sn, dd := fpgivs(piv, ar[0])
			ar[0] = dd
			yi, z[row] = float64(cs*yi)-float64(sn*z[row]), float64(cs*z[row])+float64(sn*yi)
			for i1, i2 := i+1, 1; i1 < k1; i1, i2 = i1+1, i2+1 {
				h[i1], ar[i2] = float64(cs*h[i1])-float64(sn*ar[i2]), float64(cs*ar[i2])+float64(sn*h[i1])
			}
		}
	}
}

// exactLsq is the weighted least-squares spline on knots t computed in
// 256-bit arithmetic: exact B-spline values of the float64 inputs, banded
// normal equations, Cholesky.
func exactLsq(x, y, w, t []float64, k int) []float64 {
	const prec = 256
	nf := func(v float64) *big.Float { return new(big.Float).SetPrec(prec).SetFloat64(v) }
	zero := func() *big.Float { return new(big.Float).SetPrec(prec) }
	k1 := k + 1
	nc := len(t) - k1
	nm := make([][]*big.Float, nc) // nm[i][d] = N[i][i+d], d <= k
	rhs := make([]*big.Float, nc)
	for i := range nm {
		nm[i] = make([]*big.Float, k1)
		for d := range nm[i] {
			nm[i][d] = zero()
		}
		rhs[i] = zero()
	}
	h := make([]*big.Float, k1)
	hh := make([]*big.Float, k1)
	tmp := zero()
	ell := k
	for it := range x {
		for !(x[it] < t[ell+1] || ell == nc-1) {
			ell++
		}
		xv := nf(x[it])
		h[0] = nf(1)
		for j := 1; j <= k; j++ {
			for i := 0; i < j; i++ {
				hh[i] = h[i]
			}
			h[0] = zero()
			for i := 1; i <= j; i++ {
				tl, tr := t[ell+i], t[ell+i-j]
				if tl == tr {
					h[i] = zero()
					continue
				}
				f := zero().Quo(hh[i-1], zero().Sub(nf(tl), nf(tr)))
				h[i-1] = zero().Add(h[i-1], zero().Mul(f, zero().Sub(nf(tl), xv)))
				h[i] = zero().Mul(f, zero().Sub(xv, nf(tr)))
			}
		}
		w2 := nf(w[it])
		w2.Mul(w2, w2)
		for a := 0; a <= k; a++ {
			wa := zero().Mul(w2, h[a])
			r := ell - k + a
			rhs[r].Add(rhs[r], tmp.Mul(wa, nf(y[it])))
			for b := a; b <= k; b++ {
				nm[r][b-a].Add(nm[r][b-a], tmp.Mul(wa, h[b]))
			}
		}
	}
	// Banded Cholesky N = L L^T, L[i][i-d] stored in lo[i][d].
	lo := make([][]*big.Float, nc)
	for i := 0; i < nc; i++ {
		lo[i] = make([]*big.Float, k1)
		for d := k; d >= 0; d-- {
			j := i - d
			if j < 0 {
				lo[i][d] = zero()
				continue
			}
			s := zero().Set(nm[j][d])
			for e := 1; e <= k-d && j-e >= 0; e++ {
				s.Sub(s, tmp.Mul(lo[i][d+e], lo[j][e]))
			}
			if d == 0 {
				lo[i][0] = s.Sqrt(s)
			} else {
				lo[i][d] = s.Quo(s, lo[j][0])
			}
		}
	}
	v := make([]*big.Float, nc)
	for i := 0; i < nc; i++ {
		s := zero().Set(rhs[i])
		for d := 1; d <= k && i-d >= 0; d++ {
			s.Sub(s, tmp.Mul(lo[i][d], v[i-d]))
		}
		v[i] = s.Quo(s, lo[i][0])
	}
	c := make([]float64, nc)
	for i := nc - 1; i >= 0; i-- {
		s := zero().Set(v[i])
		for d := 1; d <= k && i+d < nc; d++ {
			s.Sub(s, tmp.Mul(lo[i+d][d], v[i+d]))
		}
		v[i] = s.Quo(s, lo[i][0])
		c[i], _ = v[i].Float64()
	}
	return c
}

func maxErr(c, ref []float64) float64 {
	e, m := 0.0, 0.0
	for i := range ref {
		e = math.Max(e, math.Abs(c[i]-ref[i]))
		m = math.Max(m, math.Abs(ref[i]))
	}
	return e / m
}

// TestGivensLsq3MatchesGeneric: the k = 3 unrolled kernel does the same
// operations as the generic one, so the results are bit-identical.
func TestGivensLsq3MatchesGeneric(t *testing.T) {
	x, y, _ := perfDataPlain(3000)
	w := ones(len(x))
	for i := range w {
		w[i] = 0.5 + float64(i%7)/5
	}
	sm, err := Splrep(x, y, 3, 1, &SplrepOptions{W: w})
	if err != nil {
		t.Fatal(err)
	}
	knots := sm.Spline.T
	nc := len(knots) - 4
	a1, z1, q1 := make([]float64, nc*4), make([]float64, nc), make([]float64, len(x)*4)
	a2, z2, q2 := make([]float64, nc*4), make([]float64, nc), make([]float64, len(x)*4)
	f1 := givensLsq3(x, y, w, knots, a1, z1, q1)
	f2 := givensLsqK(x, y, w, knots, 3, a2, z2, q2)
	if f1 != f2 || ulpAll(a1, a2) != 0 || ulpAll(z1, z2) != 0 || ulpAll(q1, q2) != 0 {
		t.Errorf("k=3 kernel differs from the generic one: fp %v vs %v", f1, f2)
	}
	// Weights of 1e160 push every rotation outside the fast range: both
	// kernels fall back to fpgivs and still agree with FITPACK's arithmetic.
	big := make([]float64, len(x))
	for i := range big {
		big[i] = 1e160
	}
	for _, k := range []int{3, 2} {
		kn := knots
		if k == 2 {
			kn = knots[1 : len(knots)-1]
		}
		nck := len(kn) - k - 1
		a3, z3 := make([]float64, nck*(k+1)), make([]float64, nck)
		givensLsq(x, y, big, kn, k, a3, z3, make([]float64, len(x)*(k+1)))
		af, zf := make([]float64, nck*(k+1)), make([]float64, nck)
		givensLsqFitpack(x, y, big, kn, k, af, zf)
		c3, cf := make([]float64, nck), make([]float64, nck)
		fpback(a3, z3, nck, k+1, c3)
		fpback(af, zf, nck, k+1, cf)
		if e := maxErr(c3, cf); !(e < 1e-12) {
			t.Errorf("k=%d large weights: %g", k, e)
		}
	}
	// The fallback rotation (fpgivs) for a pivot outside the fast range.
	if cs, sn, dd := fpgivsSlow(3e200, 4e200); !givensOK(1) || givensOK(dd) || math.Abs(cs-0.8) > 1e-15 || math.Abs(sn-0.6) > 1e-15 {
		t.Errorf("fpgivsSlow: %v %v %v", cs, sn, dd)
	}
}

func TestSplrepKernelAgainstExact(t *testing.T) {
	cases := []struct {
		n, k int
		s    float64
	}{{4000, 3, 4}, {4000, 5, 4}, {20000, 5, 20}}
	if testing.Short() {
		cases = cases[:2]
	}
	for _, tc := range cases {
		x, y, _ := perfDataPlain(tc.n)
		w := ones(tc.n)
		sm, err := Splrep(x, y, tc.k, tc.s, nil)
		if err != nil {
			t.Fatal(err)
		}
		knots := sm.Spline.T
		k1 := tc.k + 1
		nc := len(knots) - k1
		a := make([]float64, nc*k1)
		z := make([]float64, nc)
		q := make([]float64, tc.n*k1)
		ours := make([]float64, nc)
		givensLsq(x, y, w, knots, tc.k, a, z, q)
		fpback(a, z, nc, k1, ours)
		fitpack := make([]float64, nc)
		givensLsqFitpack(x, y, w, knots, tc.k, a, z)
		fpback(a, z, nc, k1, fitpack)
		ref := exactLsq(x, y, w, knots, tc.k)
		eo, ef := maxErr(ours, ref), maxErr(fitpack, ref)
		t.Logf("n=%d k=%d, %d coefficients: error vs exact: Splrep %.3g, FITPACK arithmetic %.3g", tc.n, tc.k, nc, eo, ef)
		if eo > 2*ef+1e-15 {
			t.Errorf("n=%d k=%d: Splrep kernel error %.3g exceeds twice FITPACK's %.3g", tc.n, tc.k, eo, ef)
		}
	}
}
