package interp

import (
	"math"
	"testing"
)

// solveColumns is the reference for solveRows: solve applied to each
// gathered column.
func solveColumns(m *bandLU, c []float64, s int) {
	n := m.n
	line := make([]float64, n)
	for k := 0; k < s; k++ {
		for i := 0; i < n; i++ {
			line[i] = c[i*s+k]
		}
		m.solve(line)
		for i := 0; i < n; i++ {
			c[i*s+k] = line[i]
		}
	}
}

// TestSolveRowsMatchesSolve: random band matrices that need pivoting, and
// right-hand sides with zeros, signed zeros and tiny values, give the same
// bits by rows as by columns.
func TestSolveRowsMatchesSolve(t *testing.T) {
	r := newRNG(13)
	for trial := 0; trial < 200; trial++ {
		n := 1 + r.IntN(25)
		kl, ku := r.IntN(4), r.IntN(4)
		m := newBand(n, kl, ku)
		for i := 0; i < n; i++ {
			for j := max(0, i-kl); j <= min(n-1, i+ku); j++ {
				v := r.Float64()*2 - 1
				if i == j {
					v += 0.1
				}
				m.set(i, j, v)
			}
		}
		if !m.factor() {
			continue
		}
		s := 1 + r.IntN(9)
		c := make([]float64, n*s)
		for i := range c {
			switch r.IntN(7) {
			case 0:
				c[i] = 0
			case 1:
				c[i] = math.Copysign(0, -1)
			case 2:
				c[i] = 1e-310
			default:
				c[i] = r.Float64()*10 - 5
			}
		}
		want := append([]float64(nil), c...)
		m.solveRows(c, s)
		solveColumns(m, want, s)
		for i := range c {
			if !sameBits(c[i], want[i]) {
				t.Fatalf("trial %d (n=%d kl=%d ku=%d s=%d) entry %d: rows %v, columns %v", trial, n, kl, ku, s, i, c[i], want[i])
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	for _, rc := range [][2]int{{1, 1}, {1, 70}, {70, 1}, {33, 65}, {64, 64}, {5, 3}} {
		r, c := rc[0], rc[1]
		src := make([]float64, r*c)
		for i := range src {
			src[i] = float64(i)
		}
		dst := make([]float64, r*c)
		transpose(dst, src, r, c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if dst[j*r+i] != src[i*c+j] {
					t.Fatalf("%dx%d: (%d,%d)", r, c, i, j)
				}
			}
		}
	}
}

// TestMakeNdInterpRowsSolve: a 3-D grid exercises solveRows on the two
// outer axes (block strides ny*nz and nz); the result still interpolates.
func TestMakeNdInterpRowsSolve(t *testing.T) {
	ax := [][]float64{{0, 1, 2, 3, 4, 5}, {0, 0.5, 1, 2, 3}, {1, 2, 4, 8}}
	vals := make([]float64, 6*5*4)
	for i := range vals {
		vals[i] = math.Sin(float64(i) * 0.37)
	}
	s, err := MakeNdInterp(ax, vals, 3, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, 1)
	idx := 0
	for _, a := range ax[0] {
		for _, b := range ax[1] {
			for _, c := range ax[2] {
				s.EvalNu(out, []float64{a, b, c}, nil)
				if math.Abs(out[0]-vals[idx]) > 1e-12 {
					t.Fatalf("(%v,%v,%v): %v vs %v", a, b, c, out[0], vals[idx])
				}
				idx++
			}
		}
	}
}
