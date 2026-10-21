package interp

import (
	"math"
	"testing"
)

// TestBicubicAtMatchesNdBSpline: the unrolled bicubic evaluation with the
// guessed knot interval gives NdBSpline.EvalNu's bits on uniform and
// strongly non-uniform grids (where the guess misses and the bisection
// takes over), at the grid nodes, the right ends, outside (clamped) and NaN.
func TestBicubicAtMatchesNdBSpline(t *testing.T) {
	r := newRNG(15)
	for trial := 0; trial < 60; trial++ {
		nx, ny := 4+r.IntN(30), 4+r.IntN(30)
		xs := sortedX(r, nx, 0, 10, false)
		ys := sortedX(r, ny, -3, 3, false)
		if trial%3 == 0 { // clustered: the proportional guess is far off
			for i := range xs {
				xs[i] = math.Pow(float64(i)/float64(nx-1), 4) * 10
			}
		}
		z := make([]float64, nx*ny)
		for i := range z {
			z[i] = r.Float64()*2 - 1
		}
		b, err := NewBicubic(xs, ys, z, 3, 3)
		if err != nil {
			t.Fatal(err)
		}
		pts := [][2]float64{{xs[nx-1], ys[ny-1]}, {xs[0], ys[0]}, {math.NaN(), 1}, {1, math.NaN()}, {-5, 7}, {15, -9}}
		for i := 0; i < 200; i++ {
			pts = append(pts, [2]float64{r.Float64()*12 - 1, r.Float64()*8 - 4})
		}
		for i := 0; i < nx; i++ {
			pts = append(pts, [2]float64{xs[i], ys[i%ny]})
		}
		var d [1]float64
		for _, p := range pts {
			got := b.At(p[0], p[1])
			cx, cy := b.clamp(p[0], p[1])
			b.s.EvalNu(d[:], []float64{cx, cy}, []int{0, 0})
			if !sameBits(got, d[0]) {
				t.Fatalf("trial %d at %v: %v, NdBSpline %v", trial, p, got, d[0])
			}
		}
	}
}

func TestGuessInterval(t *testing.T) {
	knots := []float64{0, 0, 0, 0, 1, 2, 3, 9, 9.5, 10, 10, 10, 10}
	g := 6.0 / 10
	for x := -1.0; x <= 11; x += 0.125 {
		if got, want := guessInterval(knots, 3, x, g), findKnotInterval(knots, 3, x, 3); got != want {
			t.Fatalf("x=%v: %d, want %d", x, got, want)
		}
	}
}
