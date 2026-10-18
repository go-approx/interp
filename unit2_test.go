package interp

import (
	"errors"
	"math"
	"testing"
)

func TestInterp1DOptions(t *testing.T) {
	x := []float64{0, 1, 2, 3}
	y := []float64{10, 20, 30, 40}
	q := []float64{-1, 0, 0.5, 1.5, 3, 4, math.NaN()}
	eval := func(kind Kind, o *Interp1Options) []float64 {
		t.Helper()
		f, err := NewInterp1D(x, y, kind, o)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, len(q))
		if err := f.EvalAll(out, q); err != nil {
			t.Fatal(err)
		}
		return out
	}
	same := func(name string, got, want []float64) {
		t.Helper()
		for i := range want {
			eq(t, name, got[i], want[i], 1e-12)
		}
	}
	nan := math.NaN()
	same("linear fill NaN", eval(Linear, nil), []float64{nan, 10, 15, 25, 40, nan, nan})
	same("linear one fill", eval(Linear, &Interp1Options{Fill: []float64{0}}), []float64{0, 10, 15, 25, 40, 0, nan})
	same("linear two fills", eval(Linear, &Interp1Options{Fill: []float64{-1, -2}}), []float64{-1, 10, 15, 25, 40, -2, nan})
	same("linear extrapolate", eval(Linear, &Interp1Options{Extrapolate: true}), []float64{0, 10, 15, 25, 40, 50, nan})
	same("nearest", eval(Nearest, &Interp1Options{Extrapolate: true}), []float64{10, 10, 10, 20, 40, 40, 40})
	same("nearest-up", eval(NearestUp, &Interp1Options{Extrapolate: true}), []float64{10, 10, 20, 30, 40, 40, 40})
	same("previous", eval(Previous, nil), []float64{nan, 10, 10, 20, 40, nan, 40})
	same("previous extrapolating", eval(Previous, &Interp1Options{Extrapolate: true}), []float64{nan, 10, 10, 20, 40, 40, 40})
	same("next", eval(Next, nil), []float64{nan, 10, 20, 30, 40, nan, 40})
	same("next extrapolating", eval(Next, &Interp1Options{Extrapolate: true, BoundsError: true}), []float64{10, 10, 20, 30, 40, nan, 40})
	f, err := NewInterp1D([]float64{3, 0, 2, 1}, []float64{40, 10, 30, 20}, Linear, nil)
	if err != nil {
		t.Fatal(err)
	}
	v, err := f.At(1.5)
	eq(t, "unsorted", v, 25, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	f, _ = NewInterp1D(x, y, Linear, &Interp1Options{BoundsError: true, AssumeSorted: true})
	if _, err := f.At(5); !errors.Is(err, ErrBounds) {
		t.Error("BoundsError")
	}
	if _, err := f.At(3); err != nil {
		t.Error("the right end is inside")
	}
	if _, err := NewInterp1D(nil, nil, Linear, nil); !errors.Is(err, ErrShape) {
		t.Error("empty")
	}
	if _, err := NewInterp1D(x, y[:2], Linear, nil); !errors.Is(err, ErrShape) {
		t.Error("mismatch")
	}
	if _, err := NewInterp1D([]float64{0, math.NaN()}, []float64{1, 2}, Linear, nil); !errors.Is(err, ErrNonFinite) {
		t.Error("NaN x")
	}
	f, _ = NewInterp1D([]float64{1}, []float64{5}, Linear, &Interp1Options{Extrapolate: true})
	if v, _ := f.At(1); !math.IsNaN(v) {
		t.Error("single point extrapolating linear is NaN like scipy")
	}
	out := make([]float64, 3)
	if err := Interp(out, []float64{-1, 0.5, 9}, x, y, -5, 5); err != nil {
		t.Fatal(err)
	}
	same("Interp", out, []float64{-5, 15, 5})
	if err := Interp(out, []float64{1}, x, y[:2], 0, 0); !errors.Is(err, ErrShape) {
		t.Error("Interp lengths")
	}
	one := make([]float64, 4)
	if err := Interp(one, []float64{0, 1, 2, math.NaN()}, []float64{1}, []float64{7}, -1, 9); err != nil {
		t.Fatal(err)
	}
	same("Interp one point", one, []float64{-1, 7, 9, 7})
	inf := math.Inf(1)
	r := []float64{0}
	_ = Interp(r, []float64{0.5}, []float64{0, 1}, []float64{inf, inf}, 0, 0)
	if !math.IsInf(r[0], 1) {
		t.Errorf("flat infinite segment: %v", r[0])
	}
	_ = Interp(r, []float64{0.5}, []float64{0, 1}, []float64{inf, 0}, 0, 0)
	if !math.IsInf(r[0], 1) {
		t.Errorf("inf to finite segment, numpy's second attempt: %v", r[0])
	}
}

func TestInterp1MATLAB(t *testing.T) {
	x := []float64{3, 1, 2, 4}
	v := []float64{9, 1, 4, 16}
	xq := []float64{0, 1.5, 2.5, 5}
	nan := math.NaN()
	cases := []struct {
		method string
		ex     Extrap
		want   []float64
	}{
		{"linear", Extrap{}, []float64{nan, 2.5, 6.5, nan}},
		{"linear", Extrap{Mode: ExtrapOn}, []float64{-2, 2.5, 6.5, 23}},
		{"linear", Extrap{Mode: ExtrapValue, Value: 0}, []float64{0, 2.5, 6.5, 0}},
		{"nearest", Extrap{}, []float64{nan, 4, 9, nan}},
		{"next", Extrap{}, []float64{nan, 4, 9, nan}},
		{"previous", Extrap{}, []float64{nan, 1, 4, nan}},
		{"spline", Extrap{}, nil},
		{"pchip", Extrap{}, nil},
		{"cubic", Extrap{Mode: ExtrapValue, Value: -1}, nil},
		{"makima", Extrap{}, nil},
	}
	for _, c := range cases {
		got, err := Interp1(x, v, xq, c.method, c.ex)
		if err != nil {
			t.Fatalf("%s: %v", c.method, err)
		}
		if c.want == nil {
			if c.ex.Mode == ExtrapValue {
				eq(t, c.method+" outside", got[0], -1, 0)
			} else if math.IsNaN(got[0]) || math.IsNaN(got[3]) {
				t.Errorf("%s extrapolates by default: %v", c.method, got)
			}
			continue
		}
		for i := range c.want {
			eq(t, c.method, got[i], c.want[i], 1e-12)
		}
	}
	if _, err := Interp1(x, v, xq, "v5cubic", Extrap{}); !errors.Is(err, ErrBC) {
		t.Error("unknown method")
	}
	if _, err := Interp1(x, v[:2], xq, "pchip", Extrap{}); !errors.Is(err, ErrShape) {
		t.Error("length mismatch")
	}
	if _, err := Interp1([]float64{1, 1, 2}, []float64{1, 2, 3}, xq, "spline", Extrap{}); !errors.Is(err, ErrNotIncreasing) {
		t.Error("duplicate x with spline")
	}
	if _, err := Interp1(nil, nil, xq, "linear", Extrap{}); !errors.Is(err, ErrShape) {
		t.Error("empty")
	}
}

func TestGridErrorsAndOptions(t *testing.T) {
	ax0 := []float64{0, 1, 2, 3, 4, 5, 6}
	ax1 := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	vals := make([]float64, 56)
	for i, a := range ax0 {
		for j, b := range ax1 {
			vals[i*8+j] = 2*a - 3*b + 1
		}
	}
	axes := [][]float64{ax0, ax1}
	for _, m := range []string{"linear", "nearest", "slinear", "cubic", "quintic"} {
		r, err := NewRGI(axes, vals, m, nil)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, 2)
		if err := r.Eval(out, []float64{1.5, 2.5, -1, 3}); err != nil {
			t.Fatal(err)
		}
		if m != "nearest" {
			eq(t, m+" inside", out[0], 2*1.5-3*2.5+1, 1e-10)
		}
		if !math.IsNaN(out[1]) {
			t.Errorf("%s: outside should be NaN, got %v", m, out[1])
		}
	}
	fill := -9.0
	r, _ := NewRGI(axes, vals, "linear", &RGIOptions{FillValue: &fill})
	out := make([]float64, 1)
	_ = r.Eval(out, []float64{-1, 3})
	eq(t, "fill value", out[0], -9, 0)
	r, _ = NewRGI(axes, vals, "linear", &RGIOptions{Extrapolate: true})
	_ = r.Eval(out, []float64{-1, 3})
	eq(t, "extrapolation", out[0], 2*(-1)-3*3+1, 1e-12)
	r, _ = NewRGI(axes, vals, "linear", &RGIOptions{BoundsError: true})
	if err := r.Eval(out, []float64{-1, 3}); !errors.Is(err, ErrBounds) {
		t.Error("BoundsError")
	}
	if err := r.Eval(out, []float64{1}); !errors.Is(err, ErrShape) {
		t.Error("wrong coordinate count")
	}
	_ = r.Eval(out, []float64{math.NaN(), 3})
	if !math.IsNaN(out[0]) {
		t.Error("NaN coordinate")
	}
	bad := func(name string, axes [][]float64, vals []float64, m string, want error) {
		t.Helper()
		_, err := NewRGI(axes, vals, m, nil)
		isErr(t, name, err, want)
	}
	bad("no axes", nil, nil, "linear", ErrShape)
	bad("one-point axis", [][]float64{{1}, ax1}, vals[:8], "linear", ErrShape)
	bad("NaN axis", [][]float64{{0, math.NaN()}, {0, 1}}, vals[:4], "linear", ErrNonFinite)
	bad("unsorted axis", [][]float64{{0, 2, 1}, {0, 1}}, vals[:6], "linear", ErrNotIncreasing)
	bad("value count", axes, vals[:50], "linear", ErrShape)
	bad("method", axes, vals, "spline", ErrBC)
	bad("too few points for cubic", [][]float64{{0, 1}, {0, 1}}, vals[:4], "cubic", ErrShape)
	nanVals := append([]float64{math.NaN()}, vals[1:]...)
	bad("NaN values for a spline", axes, nanVals, "cubic", ErrNonFinite)
	if _, err := MakeNdInterp(axes, vals, 3, 3, 3); !errors.Is(err, ErrShape) {
		t.Error("degree count")
	}
	if _, err := MakeNdInterp(axes, vals, 1, 3); err != nil {
		t.Error(err)
	}
	if _, err := Interpn(axes, vals, []float64{1, 2, 3}, "linear", nil); !errors.Is(err, ErrShape) {
		t.Error("Interpn odd coordinate count")
	}
	if _, err := Interpn(axes, vals, nil, "spline", nil); !errors.Is(err, ErrBC) {
		t.Error("Interpn method")
	}
	if _, err := Interpn(nil, nil, nil, "linear", nil); !errors.Is(err, ErrShape) {
		t.Error("Interpn no axes")
	}
	v, err := Interpn(axes, vals, []float64{1.5, 2.5}, "cubic", nil)
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "Interpn", v[0], 2*1.5-3*2.5+1, 1e-10)
	s, _ := MakeNdInterp(axes, vals, 3)
	d := make([]float64, 1)
	s.EvalNu(d, []float64{2.5, 3.5}, []int{4, 0})
	eq(t, "derivative above the degree", d[0], 0, 0)
	s.EvalNu(d, []float64{2.5, 3.5}, []int{1, 0})
	eq(t, "d/dx", d[0], 2, 1e-9)
	if _, err := NewBicubic(ax0, ax1, vals[:5], 3, 3); err == nil {
		t.Error("bicubic size")
	}
	bc, err := NewBicubic(ax0, ax1, vals, 3, 3)
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "bicubic At", bc.At(1.5, 2.5), 2*1.5-3*2.5+1, 1e-10)
	eq(t, "bicubic clamps", bc.At(-4, 2), bc.At(0, 2), 1e-12)
	if !math.IsNaN(bc.At(math.NaN(), 2)) {
		t.Error("bicubic NaN")
	}
	g := make([]float64, 6)
	if err := bc.EvalGrid(g, []float64{1, 2}, []float64{1, 2, 3}, 1, 0); err != nil {
		t.Fatal(err)
	}
	eq(t, "bicubic grid", g[4], 2, 1e-9)
	if err := bc.EvalGrid(g[:5], []float64{1, 2}, []float64{1, 2, 3}, 0, 0); !errors.Is(err, ErrShape) {
		t.Error("bicubic grid size")
	}
}

func TestRemainingBranches(t *testing.T) {
	// derivative recursion over a triple knot (zero-width spans in the second pass)
	h := make([]float64, 3)
	basis([]float64{0, 0, 1, 1, 1, 2, 2}, 1, 2, 3, 1, h)
	if h[0] != 0 {
		t.Errorf("basis over a triple knot: %v", h)
	}
	// s = 0 with unusable data
	_, err := MakeSplrep([]float64{0, 2, 1, 3}, []float64{0, 1, 2, 3}, 3, 0, nil)
	isErr(t, "interpolating smoothing with unsorted x", err, ErrNotIncreasing)
	// a degree above 15 uses heap scratch in the rotations
	r := newRNG(5)
	x := sortedX(r, 60, 0, 1, false)
	y := smoothY(r, x, 0.1)
	k := 16
	var tk []float64
	for i := 0; i <= k; i++ {
		tk = append(tk, 0)
	}
	tk = append(tk, 0.3, 0.6)
	for i := 0; i <= k; i++ {
		tk = append(tk, 1)
	}
	b, err := MakeLsqSpline(x, y, tk, k, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v := b.At(0.5); math.Abs(v-math.Sin(0.85)) > 0.5 {
		t.Errorf("high-degree least squares: %v", v)
	}
}
