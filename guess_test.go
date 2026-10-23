package interp

import (
	"math"
	"testing"
)

func sameBitsG(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (a != a && b != b)
}

// TestEvalLinear2DMatchesGeneral: the unrolled 2-D linear path gives the
// general N-D loop's bits on uniform and strongly non-uniform axes, at the
// nodes, the ends, outside and for NaN.
func TestEvalLinear2DMatchesGeneral(t *testing.T) {
	r := newRNG(21)
	for trial := 0; trial < 80; trial++ {
		n0, n1 := 2+r.IntN(40), 2+r.IntN(40)
		a0 := sortedX(r, n0, -2, 7, false)
		a1 := sortedX(r, n1, 0, 1, false)
		if trial%3 == 0 { // clustered: the guess misses, bisection takes over
			for i := range a0 {
				a0[i] = math.Pow(float64(i)/float64(n0-1), 5) * 9
			}
		}
		vals := make([]float64, n0*n1)
		for i := range vals {
			vals[i] = r.Float64()*4 - 2
			if i%7 == 0 {
				vals[i] = math.Copysign(0, -1)
			}
		}
		rg, err := NewRGI([][]float64{a0, a1}, vals, "linear", &RGIOptions{Extrapolate: true})
		if err != nil {
			t.Fatal(err)
		}
		var xi []float64
		for i := 0; i < 300; i++ {
			x0, x1 := r.Float64()*14-4, r.Float64()*2-0.5
			switch i % 9 {
			case 0:
				x0 = a0[r.IntN(n0)]
			case 1:
				x1 = a1[r.IntN(n1)]
			case 2:
				x0 = math.NaN()
			case 3:
				x1 = math.NaN()
			case 4:
				x0, x1 = a0[n0-1], a1[n1-1]
			case 5:
				x0, x1 = a0[0], a1[0]
			}
			xi = append(xi, x0, x1)
		}
		got := make([]float64, len(xi)/2)
		want := make([]float64, len(xi)/2)
		if err := rg.Eval(got, xi); err != nil {
			t.Fatal(err)
		}
		rg.evalLinearNearest(want, xi)
		for i := range got {
			if !sameBitsG(got[i], want[i]) {
				t.Fatalf("trial %d point (%v, %v): %v, general %v", trial, xi[2*i], xi[2*i+1], got[i], want[i])
			}
		}
	}
}

func TestAxisInterval(t *testing.T) {
	g := []float64{0, 1, 1.5, 2, 6, 7, 7.25, 10}
	s := axisGuess(g)
	for x := -1.0; x <= 11; x += 0.0625 {
		want := min(max(searchFloat(g, x)-1, 0), len(g)-2)
		if got := axisInterval(g, x, s); got != want {
			t.Fatalf("x=%v: %d, want %d", x, got, want)
		}
	}
	// A ratio that overflows, a span that overflows: no guess; a plain axis.
	if axisGuess([]float64{0, 1e-320}) != 0 || axisGuess([]float64{-1e308, 1e308}) != 0 || axisGuess([]float64{0, 0.5, 2}) != 1 {
		t.Error("axisGuess range")
	}
}

func searchFloat(g []float64, x float64) int {
	i := 0
	for i < len(g) && g[i] < x {
		i++
	}
	return i
}

// TestFindIntervalGuess: the proportional guess returns the same piece as
// the bisection for every query, on uniform and clustered breakpoints, from
// any hint.
func TestFindIntervalGuess(t *testing.T) {
	r := newRNG(22)
	for trial := 0; trial < 50; trial++ {
		n := 2 + r.IntN(60)
		x := sortedX(r, n, -3, 3, false)
		if trial%2 == 0 {
			for i := range x {
				x[i] = math.Pow(float64(i)/float64(n-1), 4)*6 - 3
			}
		}
		g := float64(n-1) / (x[n-1] - x[0])
		for q := 0; q < 200; q++ {
			xv := r.Float64()*8 - 4
			if q%5 == 0 {
				xv = x[r.IntN(n)]
			}
			hint := r.IntN(n - 1)
			for _, ex := range []bool{false, true} {
				if a, b := findInterval(x, xv, hint, ex, g), findInterval(x, xv, hint, ex, 0); a != b {
					t.Fatalf("trial %d x=%v hint %d: guess %d, bisection %d", trial, xv, hint, a, b)
				}
			}
		}
	}
}

// TestPPolyShuffledMatchesSorted: shuffled queries (the guess path) give the
// same bits as the same queries one by one; a breakpoint span too wide for
// the ratio falls back to bisection.
func TestPPolyShuffledMatchesSorted(t *testing.T) {
	x, y, xq := perfDataPlain(3000)
	p, err := NewCubicSpline(x, y, "not-a-knot")
	if err != nil {
		t.Fatal(err)
	}
	r := newRNG(23)
	r.Shuffle(len(xq), func(i, j int) { xq[i], xq[j] = xq[j], xq[i] })
	got := make([]float64, len(xq))
	p.EvalAll(got, xq)
	for i, q := range xq {
		if w := p.At(q); !sameBitsG(got[i], w) {
			t.Fatalf("x=%v: %v vs %v", q, got[i], w)
		}
	}
	wide, err := NewPPoly([]float64{-1e308, 0, 1e308}, 1, []float64{1, 2}, Extend)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, 3)
	wide.EvalAll(out, []float64{5, -5, 1e300})
	if out[0] != 2 || out[1] != 1 || out[2] != 2 {
		t.Errorf("wide span: %v", out)
	}
}
