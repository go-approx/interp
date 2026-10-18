package interp

import "testing"

func benchData(n int) (x, y, xq []float64) { return perfDataPlain(n) }

func BenchmarkCubicConstruct(b *testing.B) {
	x, y, _ := benchData(100000)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := NewCubicSpline(x, y, "not-a-knot"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBSplineConstruct(b *testing.B) {
	x, y, _ := benchData(100000)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := MakeInterpSpline(x, y, 3, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBSplineEval(b *testing.B) {
	x, y, xq := benchData(100000)
	s, _ := MakeInterpSpline(x, y, 3, nil)
	out := make([]float64, len(xq))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.EvalAll(out, xq)
	}
}

func BenchmarkPPolyEval(b *testing.B) {
	x, y, xq := benchData(100000)
	p, _ := NewCubicSpline(x, y, "not-a-knot")
	out := make([]float64, len(xq))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.EvalAll(out, xq)
	}
}

func BenchmarkInterpLinear(b *testing.B) {
	x, y, xq := benchData(100000)
	out := make([]float64, len(xq))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Interp(out, xq, x, y, 0, 0)
	}
}
