package interp

import (
	"math"
	"testing"
)

// TestBicubicDerivBitIdentical checks the allocation-free two-dimensional
// path against the general NdBSpline evaluation, bit for bit, for several
// degrees, derivative orders, points outside the grid and NaN.
func TestBicubicDerivBitIdentical(t *testing.T) {
	r := newRNG(11)
	for _, k := range [][2]int{{1, 1}, {3, 3}, {3, 1}, {5, 3}} {
		xs := sortedX(r, 12, 0, 10, true)
		ys := sortedX(r, 9, -2, 3, true)
		z := make([]float64, len(xs)*len(ys))
		for i := range z {
			z[i] = r.NormFloat64()
		}
		b, err := NewBicubic(xs, ys, z, k[0], k[1])
		if err != nil {
			t.Fatal(err)
		}
		pts := [][2]float64{{-1, 0}, {11, 4}, {math.NaN(), 1}, {5, math.NaN()}}
		for i := 0; i < 200; i++ {
			pts = append(pts, [2]float64{r.Float64() * 10, r.Float64()*5 - 2})
		}
		for _, p := range pts {
			for dx := 0; dx <= k[0]+1; dx++ {
				for dy := 0; dy <= k[1]+1; dy++ {
					got := b.Deriv(p[0], p[1], dx, dy)
					cx, cy := b.clamp(p[0], p[1])
					var want [1]float64
					b.s.EvalNu(want[:], []float64{cx, cy}, []int{dx, dy})
					if math.Float64bits(got) != math.Float64bits(want[0]) && !(math.IsNaN(got) && math.IsNaN(want[0])) {
						t.Fatalf("k=%v (%v, %v) d=(%d,%d): %v, general path %v", k, p[0], p[1], dx, dy, got, want[0])
					}
				}
			}
		}
		dst := make([]float64, 3)
		if err := b.Ev(dst, []float64{1, 2, 3}, []float64{0, 1, 2}); err != nil {
			t.Fatal(err)
		}
		if dst[1] != b.At(2, 1) {
			t.Fatal("Ev disagrees with At")
		}
		if err := b.Ev(dst, []float64{1}, []float64{0, 1}); err == nil {
			t.Fatal("Ev accepted mismatched lengths")
		}
	}
	// Degrees past the stack buffers take the general path.
	xs := sortedX(r, 12, 0, 10, false)
	z := make([]float64, len(xs)*len(xs))
	for i := range z {
		z[i] = float64(i % 7)
	}
	b, err := NewBicubic(xs, xs, z, maxStackK+1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if v := b.At(5, 5); math.IsNaN(v) {
		t.Fatal("high-degree evaluation gave NaN")
	}
}
