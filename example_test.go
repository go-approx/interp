package interp_test

import (
	"fmt"
	"math"

	"github.com/go-approx/interp"
)

func ExampleNewCubicSpline() {
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{0, 1, 8, 27, 64} // x^3
	s, err := interp.NewCubicSpline(x, y, "not-a-knot")
	if err != nil {
		panic(err)
	}
	// A not-a-knot spline reproduces a cubic exactly.
	fmt.Printf("%.6f %.6f\n", s.At(2.5), s.Derivative(1).At(2.5))
	// Output: 15.625000 18.750000
}

func ExampleNewPchip() {
	x := []float64{0, 1, 2, 3}
	y := []float64{0, 0, 1, 1} // a step: PCHIP does not overshoot
	p, _ := interp.NewPchip(x, y)
	min, max := math.Inf(1), math.Inf(-1)
	for v := 0.0; v <= 3; v += 0.01 {
		min, max = math.Min(min, p.At(v)), math.Max(max, p.At(v))
	}
	fmt.Println(min >= 0, max <= 1)
	// Output: true true
}

func ExampleMakeInterpSpline() {
	x := []float64{0, 1, 2, 3, 4, 5}
	y := []float64{0, 1, 0, 1, 0, 1}
	b, _ := interp.MakeInterpSpline(x, y, 3, nil)
	d, _ := b.Derivative(1)
	fmt.Printf("%.1f %.4f\n", b.At(3), d.At(3))
	// Output: 1.0 0.2222
}

func ExampleMakeSplrep() {
	x := make([]float64, 50)
	y := make([]float64, 50)
	for i := range x {
		x[i] = float64(i) / 49 * 6
		y[i] = math.Sin(x[i]) + 0.01*math.Cos(97*x[i]) // a tiny fast wiggle
	}
	sm, _ := interp.MakeSplrep(x, y, 3, 0.01, nil)
	fmt.Println(len(sm.Spline.T) < 50+4, sm.Fp <= 0.01*1.001)
	// Output: true true
}

func ExampleInterp1() {
	x := []float64{1, 2, 3}
	v := []float64{10, 20, 15}
	q, _ := interp.Interp1(x, v, []float64{0, 1.5, 4}, "linear", interp.Extrap{})
	fmt.Println(q[1], math.IsNaN(q[0]), math.IsNaN(q[2]))
	// Output: 15 true true
}

func ExampleNewRGI() {
	ax := [][]float64{{0, 1, 2}, {0, 1}}
	vals := []float64{0, 1, 10, 11, 20, 21} // f(i, j) = 10 i + j
	r, _ := interp.NewRGI(ax, vals, "linear", nil)
	out := make([]float64, 1)
	_ = r.Eval(out, []float64{1.5, 0.5})
	fmt.Println(out[0])
	// Output: 15.5
}
