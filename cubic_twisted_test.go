package interp

import (
	"math"
	"math/big"
	"testing"
)

// exactTridiag solves the slope system of cubicSlopes in 256-bit
// arithmetic (Thomas algorithm; the systems are diagonally dominant).
func exactTridiag(x, y, dx, slope []float64, start, end BC) []float64 {
	n := len(x)
	const prec = 256
	f := func(v float64) *big.Float { return new(big.Float).SetPrec(prec).SetFloat64(v) }
	d0, du0, b0 := startRow(x, y, dx, slope, start)
	dn, dln, bn := endRow(x, y, dx, slope, end)
	d := make([]*big.Float, n)
	b := make([]*big.Float, n)
	dl := make([]*big.Float, n-1)
	du := make([]*big.Float, n-1)
	d[0], b[0], du[0] = f(d0), f(b0), f(du0)
	for j := 1; j <= n-2; j++ {
		d[j] = f(2 * (dx[j-1] + dx[j]))
		b[j] = f(3 * (float64(dx[j]*slope[j-1]) + float64(dx[j-1]*slope[j])))
		du[j] = f(dx[j-1])
		dl[j-1] = f(dx[j])
	}
	d[n-1], b[n-1], dl[n-2] = f(dn), f(bn), f(dln)
	t := new(big.Float).SetPrec(prec)
	for i := 0; i < n-1; i++ {
		fact := new(big.Float).SetPrec(prec).Quo(dl[i], d[i])
		d[i+1].Sub(d[i+1], t.Mul(fact, du[i]))
		b[i+1].Sub(b[i+1], t.Mul(fact, b[i]))
	}
	out := make([]float64, n)
	b[n-1].Quo(b[n-1], d[n-1])
	out[n-1], _ = b[n-1].Float64()
	for i := n - 2; i >= 0; i-- {
		b[i].Sub(b[i], t.Mul(du[i], b[i+1]))
		b[i].Quo(b[i], d[i])
		out[i], _ = b[i].Float64()
	}
	return out
}

func slopeInputs(n int) (x, y, dx, slope []float64) {
	x, y, _ = perfDataPlain(n)
	dx = make([]float64, n-1)
	slope = make([]float64, n-1)
	for i := range dx {
		dx[i] = x[i+1] - x[i]
		slope[i] = (y[i+1] - y[i]) / dx[i]
	}
	return
}

// TestCubicTwistedAccuracy: the twisted elimination is as close to the
// exact solution as dgtsv's (the error of either is a few ulps).
func TestCubicTwistedAccuracy(t *testing.T) {
	bcs := []struct {
		name       string
		start, end BC
	}{
		{"not-a-knot", BC{Kind: NotAKnot}, BC{Kind: NotAKnot}},
		{"natural", BC{Kind: D2}, BC{Kind: D2}},
		{"mixed", BC{Kind: NotAKnot}, BC{Kind: D2, Value: 0.5}},
	}
	for _, n := range []int{1500, 1501, 20000} {
		x, y, dx, slope := slopeInputs(n)
		for _, bc := range bcs {
			tw, ok := cubicSlopesTwisted(x, y, dx, slope, bc.start, bc.end)
			if !ok {
				t.Fatalf("%s n=%d: twisted refused", bc.name, n)
			}
			ref, ok := cubicSlopesFast(x, y, dx, slope, bc.start, bc.end)
			if !ok {
				t.Fatalf("%s n=%d: dgtsv path refused", bc.name, n)
			}
			ex := exactTridiag(x, y, dx, slope, bc.start, bc.end)
			et, eg := maxErr(tw, ex), maxErr(ref, ex)
			t.Logf("%-10s n=%5d: error vs exact (scaled): twisted %.3g, dgtsv %.3g", bc.name, n, et, eg)
			if et > 2*eg+4e-16 {
				t.Errorf("%s n=%d: twisted error %.3g exceeds dgtsv's %.3g", bc.name, n, et, eg)
			}
		}
	}
}

// gtsvSlopes is the reference for cubicSlopesFast: the three diagonals
// assembled, then gtsv.
func gtsvSlopes(x, y, dx, slope []float64, start, end BC) ([]float64, bool) {
	n := len(x)
	du := make([]float64, n-1)
	di := make([]float64, n)
	dl := make([]float64, n-1)
	b := make([]float64, n)
	for j := 1; j <= n-2; j++ {
		di[j] = 2 * (dx[j-1] + dx[j])
		b[j] = 3 * (float64(dx[j]*slope[j-1]) + float64(dx[j-1]*slope[j]))
		du[j] = dx[j-1]
		dl[j-1] = dx[j]
	}
	di[0], du[0], b[0] = startRow(x, y, dx, slope, start)
	di[n-1], dl[n-2], b[n-1] = endRow(x, y, dx, slope, end)
	return b, gtsv(dl, di, du, b)
}

// TestCubicSlopesFastMatchesGtsv: below twistMin the slopes are dgtsv's
// bit for bit (SciPy's arithmetic), for every boundary condition, random
// spacings and sizes from 2 nodes.
func TestCubicSlopesFastMatchesGtsv(t *testing.T) {
	r := newRNG(14)
	bcs := []BC{{Kind: NotAKnot}, {Kind: D1, Value: 0.7}, {Kind: D2, Value: -1.5}, {Kind: D2}}
	for trial := 0; trial < 300; trial++ {
		n := 2 + r.IntN(40)
		start, end := bcs[r.IntN(4)], bcs[r.IntN(4)]
		if n < 4 && (start.Kind == NotAKnot || end.Kind == NotAKnot) {
			continue // n = 2, 3 with not-a-knot are special-cased by the caller
		}
		x := sortedX(r, n, 0, float64(1+r.IntN(30)), false)
		y := make([]float64, n)
		dx := make([]float64, n-1)
		slope := make([]float64, n-1)
		for i := range y {
			y[i] = r.Float64()*2 - 1
		}
		for i := range dx {
			dx[i] = x[i+1] - x[i]
			slope[i] = (y[i+1] - y[i]) / dx[i]
		}
		want, okw := gtsvSlopes(x, y, dx, slope, start, end)
		got, ok := cubicSlopesFast(x, y, dx, slope, start, end)
		if !ok {
			continue // a pivot is needed: cubicSlopes uses gtsv itself
		}
		if !okw {
			t.Fatalf("trial %d: fast path solved a system gtsv calls singular", trial)
		}
		for i := range got {
			if !sameBits(got[i], want[i]) {
				t.Fatalf("trial %d (n=%d %v/%v) slope %d: %v vs %v", trial, n, start.Kind, end.Kind, i, got[i], want[i])
			}
		}
	}
}

// TestCubicTwistedRefuses: the twisted solve hands over to dgtsv when a
// pivot is both smaller than the entry it eliminates and than its own
// row's other entry (spacings of both signs make such a reduced row; the
// guard is internal, so the abscissae need not be sorted here), and for
// the singular not-a-knot system on three nodes.
func TestCubicTwistedRefuses(t *testing.T) {
	try := func(name string, dx []float64, start, end BC, want bool) {
		t.Helper()
		n := len(dx) + 1
		x := make([]float64, n)
		y := make([]float64, n)
		for i := 1; i < n; i++ {
			x[i] = x[i-1] + dx[i-1]
			y[i] = float64(i % 3)
		}
		slope := make([]float64, n-1)
		for i := range slope {
			slope[i] = (y[i+1] - y[i]) / dx[i]
		}
		if _, ok := cubicSlopesTwisted(x, y, dx, slope, start, end); ok != want {
			t.Errorf("%s: accepted %v, want %v", name, ok, want)
		}
	}
	nat := BC{Kind: D2}
	try("small pivot from the top", []float64{1, -1.3, 1, 1, 1, 1}, nat, nat, false)
	try("small pivot from the bottom", []float64{1, 1, 1, 1, -1.3, 1}, nat, nat, false)
	try("well spaced", []float64{1, 1, 1, 1, 1, 1}, nat, nat, true)
	try("three nodes, not-a-knot", []float64{1, 2}, BC{Kind: NotAKnot}, BC{Kind: NotAKnot}, false)
	if stablePivot(0, 0, 0) || !stablePivot(1, 2, 1) || !stablePivot(1, 1, 2) || stablePivot(1, 2, 2) {
		t.Error("stablePivot")
	}
}

// TestCubicTwistedPath:NewCubicSpline uses the twisted solve from twistMin
// nodes and still interpolates with continuous second derivatives; a
// clamped condition whose first pivot is smaller than the entry below it
// makes the twisted solve refuse (dgtsv pivots) and the result is the
// dgtsv one.
func TestCubicTwistedPath(t *testing.T) {
	for _, n := range []int{twistMin - 1, twistMin, twistMin + 1} {
		x, y, _ := perfDataPlain(n)
		p, err := NewCubicSpline(x, y, "not-a-knot")
		if err != nil {
			t.Fatal(err)
		}
		v := make([]float64, n)
		p.EvalAll(v, x)
		for i := range v {
			if math.Abs(v[i]-y[i]) > 1e-12 {
				t.Fatalf("n=%d: not interpolating at %d", n, i)
			}
		}
	}
	// Spacing 10 > the clamped first row's diagonal 1: a pivot is needed
	// at the top for dgtsv; the twisted solve must refuse rather than
	// eliminate without it.
	n := twistMin + 10
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = 10 * float64(i)
		y[i] = math.Sin(float64(i) * 0.1)
	}
	dx := make([]float64, n-1)
	slope := make([]float64, n-1)
	for i := range dx {
		dx[i] = x[i+1] - x[i]
		slope[i] = (y[i+1] - y[i]) / dx[i]
	}
	// The clamped row (1, 0) is diagonally dominant: accepted although its
	// multiplier is 10, and as accurate as dgtsv (which pivots here).
	cl := BC{Kind: D1, Value: 0.3}
	tw, ok := cubicSlopesTwisted(x, y, dx, slope, cl, cl)
	if !ok {
		t.Fatal("twisted solve refused a diagonally dominant clamped system")
	}
	if e := maxErr(tw, exactTridiag(x, y, dx, slope, cl, cl)); e > 1e-15 {
		t.Errorf("clamped, large spacing: error %.3g", e)
	}
	p, err := NewCubicSplineBC(x, y, cl, cl)
	if err != nil {
		t.Fatal(err)
	}
	if d := p.Derivative(1).At(x[0]); math.Abs(d-0.3) > 1e-12 {
		t.Errorf("clamped start derivative %v", d)
	}
}
