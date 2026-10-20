package interp

import (
	"math"
	"testing"
)

// npInterpLoop is the reference: npInterp point by point, as Interp was
// before interpKernel.
func npInterpLoop(dst, x, xp, fp []float64, left, right float64, sl []float64) {
	hint := 0
	for i, v := range x {
		dst[i], hint = npInterp(v, xp, fp, left, right, hint, sl)
	}
}

func sameBits(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (a != a && b != b)
}

// TestInterpKernelMatchesReference: the inline fast path of interpKernel
// gives npInterp's results bit for bit on sorted, shuffled, clustered and
// sparse queries, duplicate nodes, NaN and infinite values, queries on the
// nodes and outside.
func TestInterpKernelMatchesReference(t *testing.T) {
	r := newRNG(11)
	nan, inf := math.NaN(), math.Inf(1)
	for trial := 0; trial < 300; trial++ {
		n := 1 + r.IntN(60)
		xp := sortedX(r, n, -5, 5, false)
		if trial%5 == 0 && n > 3 { // duplicate nodes
			xp[n/2] = xp[n/2-1]
		}
		fp := make([]float64, n)
		for i := range fp {
			fp[i] = r.Float64()*4 - 2
		}
		if trial%7 == 0 {
			fp[r.IntN(n)] = inf
		}
		if trial%11 == 0 {
			fp[r.IntN(n)] = nan
		}
		m := r.IntN(200)
		x := make([]float64, m)
		for i := range x {
			switch r.IntN(10) {
			case 0:
				x[i] = xp[r.IntN(n)] // on a node
			case 1:
				x[i] = r.Float64()*30 - 15 // often outside
			case 2:
				x[i] = nan
			default:
				x[i] = r.Float64()*10 - 5
			}
		}
		if trial%2 == 0 { // sorted queries: the walking and galloping path
			sortFloats(x)
		}
		var sl []float64
		if trial%3 != 0 {
			sl = slopes(xp, fp)
		}
		got, want := make([]float64, m), make([]float64, m)
		interpKernel(got, x, xp, fp, -7, 9, sl)
		npInterpLoop(want, x, xp, fp, -7, 9, sl)
		for i := range got {
			if !sameBits(got[i], want[i]) {
				t.Fatalf("trial %d, x[%d]=%v: kernel %v, reference %v", trial, i, x[i], got[i], want[i])
			}
		}
	}
}

// TestInterpKernelLongJumps: sparse sorted queries over many nodes jump
// far ahead (gallop) and back (a restart).
func TestInterpKernelLongJumps(t *testing.T) {
	x, y, _ := perfDataPlain(5000)
	q := []float64{0.001, 0.002, 3, 3.0001, 9.99, 0.5, 7, 10, 10.5, -1, 4}
	got, want := make([]float64, len(q)), make([]float64, len(q))
	for _, sl := range [][]float64{nil, slopes(x, y)} {
		interpKernel(got, q, x, y, 0, 0, sl)
		npInterpLoop(want, q, x, y, 0, 0, sl)
		for i := range got {
			if !sameBits(got[i], want[i]) {
				t.Fatalf("q=%v: %v vs %v", q[i], got[i], want[i])
			}
		}
	}
	// gallop itself: every start and target on a small grid.
	xp := []float64{0, 1, 1, 2, 3, 5, 8, 13}
	for lo := 0; lo < len(xp)-1; lo++ {
		for v := xp[lo]; v < xp[len(xp)-1]; v += 0.25 {
			j := gallop(xp, v, lo)
			if !(xp[j] <= v && v < xp[j+1]) {
				t.Fatalf("gallop(%v from %d) = %d", v, lo, j)
			}
		}
	}
}

// TestInterp1DLinearKernel: interp1d linear without extrapolation now runs
// through interpKernel with the fill values; it must agree with the
// general loop's rules (fill outside, NaN propagated, exact ends).
func TestInterp1DLinearKernel(t *testing.T) {
	x := []float64{0, 1, 2, 4}
	y := []float64{1, 3, 2, 6}
	f, err := NewInterp1D(x, y, Linear, &Interp1Options{Fill: []float64{-1, -2}})
	if err != nil {
		t.Fatal(err)
	}
	q := []float64{-0.5, 0, 0.5, 1, 3, 4, 4.5, math.NaN()}
	want := []float64{-1, 1, 2, 3, 4, 6, -2, math.NaN()}
	got := make([]float64, len(q))
	if err := f.EvalAll(got, q); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if !sameBits(got[i], want[i]) {
			t.Errorf("x=%v: %v, want %v", q[i], got[i], want[i])
		}
	}
}

func sortFloats(v []float64) {
	for i := 1; i < len(v); i++ { // insertion sort, NaN kept in place order
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}
