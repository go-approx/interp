package interp

import (
	"errors"
	"math"
	"testing"
)

func eq(t *testing.T, name string, got, want float64, tol float64) {
	t.Helper()
	if !(math.Abs(got-want) <= tol) && !(math.IsNaN(got) && math.IsNaN(want)) {
		t.Errorf("%s: got %.17g, want %.17g", name, got, want)
	}
}

func isErr(t *testing.T, name string, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Errorf("%s: err = %v, want %v", name, err, want)
	}
}

func TestLinsolveSingular(t *testing.T) {
	// gtsv: a zero pivot with nothing to swap in, at each place it can occur
	if gtsv([]float64{0, 0}, []float64{0, 1, 1}, []float64{1, 1}, []float64{1, 1, 1}) {
		t.Error("gtsv: zero first pivot should fail")
	}
	if gtsv([]float64{1, 0}, []float64{1, 1, 0}, []float64{1, 1}, []float64{1, 1, 1}) {
		// row elimination leaves a zero last pivot
		t.Error("gtsv: zero last pivot should fail")
	}
	if gtsv([]float64{0}, []float64{0, 1}, []float64{1}, []float64{1, 1}) {
		t.Error("gtsv: zero pivot in the n-2 step should fail")
	}
	// the pivoting branches: small diagonal, large sub-diagonal
	d := []float64{1e-3, 1e-3, 1e-3, 1e-3}
	dl := []float64{1, 1, 1}
	du := []float64{1, 1, 1}
	b := []float64{1, 2, 3, 4}
	if !gtsv(dl, d, du, b) {
		t.Fatal("pivoting gtsv failed")
	}
	// check A x = b for the original matrix
	x := b
	for i, want := range []float64{1, 2, 3, 4} {
		got := 1e-3 * x[i]
		if i > 0 {
			got += x[i-1]
		}
		if i < 3 {
			got += x[i+1]
		}
		eq(t, "pivot solve", got, want, 1e-9)
	}
	// single and double unknowns
	if !gtsv(nil, []float64{2}, nil, []float64{4}) {
		t.Error("1x1 failed")
	}
	if gtsv(nil, []float64{0}, nil, []float64{4}) {
		t.Error("singular 1x1 should fail")
	}
	b2 := []float64{3, 5}
	if !gtsv([]float64{1}, []float64{1e-3}, []float64{1}, b2) { // swap branch at n == 2
		t.Error("2x2 failed")
	}
	if gtsv([]float64{0}, []float64{0}, []float64{1}, []float64{1, 1}) {
		t.Error("2x2 zero pivot should fail")
	}
}

func TestDenseAndBandSolve(t *testing.T) {
	a := []float64{0, 1, 1, 1}
	b := []float64{2, 3}
	if !denseSolve(a, b, 2) {
		t.Fatal("dense with swap failed")
	}
	eq(t, "x0", b[0], 1, 1e-14)
	eq(t, "x1", b[1], 2, 1e-14)
	if denseSolve([]float64{1, 1, 1, 1}, []float64{1, 2}, 2) {
		t.Error("singular dense should fail")
	}
	m := newBand(3, 1, 1)
	m.set(0, 0, 0)
	m.set(0, 1, 1)
	m.set(1, 0, 1)
	m.set(1, 1, 0)
	m.set(1, 2, 1)
	m.set(2, 1, 1)
	m.set(2, 2, 0)
	if m.factor() {
		t.Error("singular band should fail")
	}
	m = newBand(3, 1, 1)
	for i := 0; i < 3; i++ {
		m.set(i, i, 4)
		if i > 0 {
			m.set(i, i-1, 1)
		}
		if i < 2 {
			m.set(i, i+1, 1)
		}
	}
	m.set(1, 0, 5) // forces a row swap in column 0
	if !m.factor() {
		t.Fatal("band factor failed")
	}
	r := []float64{1, 0, 2} // a zero entry exercises the b[j] == 0 shortcut
	m.solve(r)
	eq(t, "band row 0", 4*r[0]+r[1], 1, 1e-13)
	eq(t, "band row 1", 5*r[0]+4*r[1]+r[2], 0, 1e-13)
}

func TestPPolyBasics(t *testing.T) {
	x := []float64{0, 1, 3}
	// p0 = 1 + 2s, p1 = 3 + s^2 (ascending powers)
	p, err := NewPPoly(x, 3, []float64{1, 2, 0, 3, 0, 1}, Extend)
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "coef", p.Coef(1, 2), 1, 0)
	eq(t, "At", p.At(0.5), 2, 1e-15)
	eq(t, "At interval 2", p.At(2), 3+1, 1e-15)
	eq(t, "extend left", p.At(-1), -1, 1e-15)
	eq(t, "extend right", p.At(4), 3+9, 1e-15)
	if !math.IsNaN(p.At(math.NaN())) {
		t.Error("NaN should give NaN")
	}
	q := *p
	q.Extrap = NaNOutside
	if !math.IsNaN(q.At(-1)) || !math.IsNaN(q.At(3.5)) {
		t.Error("NaNOutside should give NaN")
	}
	eq(t, "right end inside", q.At(3), 3+4, 1e-15)
	w := *p
	w.Extrap = Wrap
	eq(t, "wrap", w.At(3.5), p.At(0.5), 1e-15)
	eq(t, "wrap left", w.At(-0.5), p.At(2.5), 1e-15)
	eq(t, "wrap zero", w.At(3), p.At(0), 1e-15)
	// derivatives: evalPoly1 with dx > 0 and the dx < 0 antiderivative path
	out := make([]float64, 1)
	p.EvalNu(out, []float64{2}, 1)
	eq(t, "d/dx", out[0], 2, 1e-15)
	p.EvalNu(out, []float64{2}, 2)
	eq(t, "d2/dx2", out[0], 2, 1e-15)
	p.EvalNu(out, []float64{2}, 3)
	eq(t, "d3/dx3", out[0], 0, 1e-15)
	p.EvalNu(out, []float64{2}, -1)
	eq(t, "antiderivative from zero", out[0], 3*1+1.0/3, 1e-15)
	// Derivative / Antiderivative objects
	d := p.Derivative(1)
	eq(t, "Derivative", d.At(2), 2, 1e-15)
	if z := p.Derivative(5); z.K != 1 || z.At(2) != 0 {
		t.Errorf("derivative beyond the degree should be zero: %+v", z)
	}
	if n := p.Derivative(-1); n.K != 4 {
		t.Errorf("negative derivative is the antiderivative: K=%d", n.K)
	}
	if n := p.Antiderivative(0); n.K != 3 {
		t.Errorf("zero-order antiderivative is the function: K=%d", n.K)
	}
	a := p.Antiderivative(2)
	eq(t, "second antiderivative, second derivative", a.Derivative(2).At(2), p.At(2), 1e-13)
	eq(t, "continuity of the antiderivative", a.At(1-1e-12), a.At(1+1e-12), 1e-9)
	pa := w.Antiderivative(1)
	if pa.Extrap != NaNOutside {
		t.Error("antiderivative of a periodic PPoly must not wrap")
	}
	if _, err := NewPPoly([]float64{0}, 1, nil, Extend); !errors.Is(err, ErrShape) {
		t.Error("one breakpoint")
	}
	if _, err := NewPPoly([]float64{0, 0}, 1, []float64{1}, Extend); !errors.Is(err, ErrNotIncreasing) {
		t.Error("repeated breakpoint")
	}
	if floorMod(-0.0, 5) != 0 || math.Signbit(floorMod(-0.0, 5)) {
		t.Error("floorMod of zero keeps the sign of the divisor")
	}
}

func TestPPolyIntegrate(t *testing.T) {
	x := []float64{0, 1, 3}
	p, _ := NewPPoly(x, 2, []float64{1, 1, 2, -1}, Extend) // 1+s on [0,1], 2-s on [1,3]
	eq(t, "empty", p.Integrate(1, 1), 0, 0)
	eq(t, "inside", p.Integrate(0, 1), 1.5, 1e-15)
	eq(t, "across", p.Integrate(0.5, 2), 1.5-(0.5+0.125)+(2-0.5), 1e-14)
	eq(t, "reversed", p.Integrate(2, 0.5), -p.Integrate(0.5, 2), 1e-15)
	eq(t, "extended", p.Integrate(-1, 0), 0.5, 1e-14)
	q := *p
	q.Extrap = NaNOutside
	if !math.IsNaN(q.Integrate(-1, 2)) {
		t.Error("integral reaching outside a NaNOutside PPoly is NaN")
	}
	w := *p
	w.Extrap = Wrap
	period := w.Integrate(0, 3)
	eq(t, "periods", w.Integrate(0, 6.5), 2*period+w.Integrate(0, 0.5), 1e-13)
	eq(t, "wrap across the end", w.Integrate(2.5, 3.5), w.Integrate(2.5, 3)+w.Integrate(0, 0.5), 1e-14)
	eq(t, "wrap inside", w.Integrate(0.5, 0.9), p.Integrate(0.5, 0.9), 1e-15)
}

func TestCubicErrorsAndOptions(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{0, 1, 0, 1, 0}
	if _, err := NewCubicSpline(x, y, "wobbly"); !errors.Is(err, ErrBC) {
		t.Error("unknown bc name")
	}
	if _, err := NewCubicSplineBC(x, y, BC{Kind: Periodic}, BC{Kind: Natural}); !errors.Is(err, ErrBC) {
		t.Error("periodic on one end")
	}
	if _, err := NewCubicSplineBC(x, []float64{0, 1, 0, 1, 2}, BC{Kind: Periodic}, BC{Kind: Periodic}); !errors.Is(err, ErrPeriodic) {
		t.Error("periodic mismatch")
	}
	if _, err := NewCubicSpline(x[:1], y[:1], "natural"); !errors.Is(err, ErrShape) {
		t.Error("one point")
	}
	if _, err := NewCubicSpline(x, y[:3], "natural"); !errors.Is(err, ErrShape) {
		t.Error("length mismatch")
	}
	if _, err := NewCubicSpline([]float64{0, 1, math.Inf(1)}, []float64{1, 2, 3}, "natural"); !errors.Is(err, ErrNonFinite) {
		t.Error("infinite x")
	}
	if _, err := NewCubicSpline([]float64{0, 2, 1}, []float64{1, 2, 3}, "natural"); !errors.Is(err, ErrNotIncreasing) {
		t.Error("unsorted x")
	}
	// all four boundary conditions of a straight line reproduce it
	line := []float64{1, 3, 5, 7, 9}
	for _, bc := range []string{"not-a-knot", "natural", "clamped"} {
		p, err := NewCubicSpline(x, line, bc)
		if err != nil {
			t.Fatal(err)
		}
		if bc == "clamped" {
			continue // zero end slopes bend the line
		}
		eq(t, bc, p.At(2.5), 6, 1e-13)
	}
	// prescribed end derivatives
	p, err := NewCubicSplineBC(x, line, BC{D1, 2}, BC{D1, 2})
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "D1 line", p.At(2.5), 6, 1e-13)
	p, err = NewCubicSplineBC(x, line, BC{D2, 0}, BC{D2, 0})
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "D2 line", p.At(2.5), 6, 1e-13)
	// two points: a straight line, whatever the condition
	p, err = NewCubicSpline([]float64{0, 2}, []float64{1, 5}, "periodic")
	if err == nil {
		t.Error("two points with y[0] != y[1] cannot be periodic")
	}
	p, err = NewCubicSpline([]float64{0, 2}, []float64{1, 1}, "periodic")
	if err != nil || p.At(1) != 1 {
		t.Errorf("two-point periodic: %v", err)
	}
	p, err = NewCubicSpline([]float64{0, 2}, []float64{1, 5}, "not-a-knot")
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "two points", p.At(1), 3, 1e-15)
	// three points: not-a-knot is a parabola, periodic has its own rule
	p, err = NewCubicSpline([]float64{0, 1, 2}, []float64{0, 1, 4}, "not-a-knot")
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "parabola", p.At(1.5), 2.25, 1e-14)
	p, err = NewCubicSpline([]float64{0, 1, 2}, []float64{0, 1, 0}, "periodic")
	if err != nil {
		t.Fatal(err)
	}
	if p.Extrap != Wrap {
		t.Error("periodic spline wraps")
	}
	// NewCubicHermite
	h, err := NewCubicHermite([]float64{0, 1}, []float64{0, 1}, []float64{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "hermite line", h.At(0.25), 0.25, 1e-15)
	if _, err := NewCubicHermite([]float64{0, 1}, []float64{0, 1}, []float64{1}); !errors.Is(err, ErrShape) {
		t.Error("hermite derivative length")
	}
	if _, err := NewCubicHermite([]float64{0, 1}, []float64{0, 1}, []float64{1, math.NaN()}); !errors.Is(err, ErrNonFinite) {
		t.Error("hermite NaN derivative")
	}
	if _, err := NewCubicHermite([]float64{1, 0}, []float64{0, 1}, []float64{1, 1}); !errors.Is(err, ErrNotIncreasing) {
		t.Error("hermite order")
	}
	if sign(math.NaN()) == 0 || !math.IsNaN(sign(math.NaN())) {
		t.Error("sign of NaN")
	}
}

func TestPchipAkimaErrors(t *testing.T) {
	if _, err := NewPchip([]float64{0}, []float64{0}); !errors.Is(err, ErrShape) {
		t.Error("pchip one point")
	}
	if _, err := NewAkima([]float64{0, 0}, []float64{0, 1}, false); !errors.Is(err, ErrNotIncreasing) {
		t.Error("akima duplicate")
	}
	p, err := NewAkima([]float64{0, 1}, []float64{0, 2}, true)
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "akima two points", p.At(0.5), 1, 1e-15)
	if !math.IsNaN(p.At(2)) {
		t.Error("akima does not extrapolate by default")
	}
}

func TestBSplineBasics(t *testing.T) {
	// a cubic with the boundary knots repeated four times
	knots := []float64{0, 0, 0, 0, 1, 2, 2, 2, 2}
	coef := []float64{0, 1, 3, 2, 1}
	b, err := NewBSpline(knots, coef, 3)
	if err != nil {
		t.Fatal(err)
	}
	if b.N() != 5 {
		t.Errorf("N = %d", b.N())
	}
	v := b.At(0.7)
	eq(t, "At matches EvalAll", v, func() float64 { d := make([]float64, 1); b.EvalAll(d, []float64{0.7}); return d[0] }(), 0)
	// the derivative of the derivative against finite differences
	d1, err := b.Derivative(1)
	if err != nil {
		t.Fatal(err)
	}
	h := 1e-6
	eq(t, "derivative", d1.At(0.7), (b.At(0.7+h)-b.At(0.7-h))/(2*h), 1e-8)
	d0, err := b.Derivative(0)
	if err != nil || d0.At(0.7) != v {
		t.Error("zeroth derivative is a copy")
	}
	if _, err := b.Derivative(4); !errors.Is(err, ErrShape) {
		t.Error("derivative beyond the degree")
	}
	if _, err := b.Derivative(-1); !errors.Is(err, ErrShape) {
		t.Error("negative derivative")
	}
	out := make([]float64, 2)
	b.EvalNu(out, []float64{0.5, math.NaN()}, 4)
	if out[0] != 0 || !math.IsNaN(out[1]) {
		t.Errorf("derivative above the degree is zero: %v", out)
	}
	// antiderivative and integrate
	a := b.Antiderivative(1)
	eq(t, "anti starts at zero", a.At(0), 0, 1e-15)
	eq(t, "integrate", b.Integrate(0.2, 1.5), a.At(1.5)-a.At(0.2), 1e-14)
	eq(t, "integrate swapped", b.Integrate(1.5, 0.2), -b.Integrate(0.2, 1.5), 1e-15)
	nb := *b
	nb.Extrap = NaNOutside
	eq(t, "clipped integral", nb.Integrate(-1, 5), b.Integrate(0, 2), 1e-14)
	eq(t, "empty clipped integral", nb.Integrate(5, 6), 0, 0)
	if !math.IsNaN(nb.At(-1)) || !math.IsNaN(nb.At(3)) {
		t.Error("NaNOutside")
	}
	wb := *b
	wb.Extrap = Wrap
	eq(t, "wrap eval", wb.At(2.7), b.At(0.7), 1e-14)
	eq(t, "wrap integral", wb.Integrate(0.5, 4.5), 2*b.Integrate(0, 2)+b.Integrate(0.5, 0.5+0.0)+wb.Integrate(0.5, 0.5), 1e-12)
	eq(t, "wrap integral across", wb.Integrate(1.5, 2.5), b.Integrate(1.5, 2)+b.Integrate(0, 0.5), 1e-14)
	eq(t, "wrap integral inside", wb.Integrate(0.5, 1), b.Integrate(0.5, 1), 1e-15)
	if wa := wb.Antiderivative(1); wa.Extrap != NaNOutside {
		t.Error("antiderivative of a wrapping spline does not wrap")
	}
	// constructor checks
	if _, err := NewBSpline(knots, coef[:3], 3); !errors.Is(err, ErrShape) {
		t.Error("short coefficients")
	}
	if _, err := NewBSpline([]float64{0, 1, 0.5, 2, 3, 4, 5, 6, 7}, coef, 3); !errors.Is(err, ErrKnots) {
		t.Error("unsorted knots")
	}
	if _, err := NewBSpline([]float64{0, 0, 0, 0, math.NaN(), 2, 2, 2, 2}, coef, 3); !errors.Is(err, ErrNonFinite) {
		t.Error("NaN knot")
	}
	// a derivative whose divisor vanishes: k+1 equal interior knots
	z := &BSpline{T: []float64{0, 0, 1, 1, 1, 2, 2}, C: []float64{1, 2, 3, 4, 5}[:4], K: 1}
	z.T = []float64{0, 0, 1, 1, 2, 2}
	z.C = []float64{1, 2, 3, 4}
	if _, err := z.Derivative(1); !errors.Is(err, ErrKnots) {
		t.Errorf("a jump between double knots has no first derivative: %v", err)
	}
	z2 := &BSpline{T: []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}, C: []float64{1, 2, 3, 4, 5, 6}[:6], K: 2}
	z2.T = []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}
	z2.C = []float64{1, 2, 3, 4, 5, 6}
	if _, err := z2.Derivative(1); !errors.Is(err, ErrKnots) {
		t.Errorf("triple interior knot with k = 2: %v", err)
	}
}

func TestBSplineDegenerateKnots(t *testing.T) {
	// k > 15 uses heap scratch
	n := 20
	tk := make([]float64, 0, 2*n)
	for i := 0; i < 18; i++ {
		tk = append(tk, 0)
	}
	for i := 0; i < 18; i++ {
		tk = append(tk, 1)
	}
	c := make([]float64, 18)
	c[0] = 1
	b, err := NewBSpline(tk, c, 17)
	if err != nil {
		t.Fatal(err)
	}
	eq(t, "Bernstein B_0 at 0.5", b.At(0.5), math.Pow(0.5, 17), 1e-18)
	_ = n
	// repeated knots inside the basis recursion (zero-width spans) and in a derivative
	b2, _ := NewBSpline([]float64{0, 0, 0, 1, 1, 2, 2, 2}, []float64{1, 2, 3, 4, 5}, 2)
	d := make([]float64, 3)
	b2.EvalNu(d, []float64{0.5, 1, 1.5}, 1)
	if math.IsNaN(d[0]) || math.IsNaN(d[1]) || math.IsNaN(d[2]) {
		t.Errorf("derivative with a double interior knot: %v", d)
	}
}

func TestDesignMatrix(t *testing.T) {
	tk := []float64{0, 0, 0, 0, 1, 2, 2, 2, 2}
	x := []float64{0, 0.3, 1, 2, 1.7}
	dm, err := NewDesignMatrix(x, tk, 3, false)
	if err != nil {
		t.Fatal(err)
	}
	dense := dm.Dense()
	coef := []float64{0, 1, 3, 2, 1}
	b, _ := NewBSpline(tk, coef, 3)
	for i, xv := range x {
		s := 0.0
		for j, cj := range coef {
			s += dense[i*dm.N+j] * cj
		}
		eq(t, "design row", s, b.At(xv), 1e-14)
	}
	if _, err := NewDesignMatrix([]float64{3}, tk, 3, false); !errors.Is(err, ErrBounds) {
		t.Error("outside without extrapolation")
	}
	if _, err := NewDesignMatrix([]float64{math.NaN()}, tk, 3, true); !errors.Is(err, ErrBounds) {
		t.Error("NaN")
	}
	if _, err := NewDesignMatrix([]float64{3}, tk, 3, true); err != nil {
		t.Errorf("extrapolating: %v", err)
	}
	if _, err := NewDesignMatrix(x, tk[:5], 3, false); !errors.Is(err, ErrShape) {
		t.Error("too few knots")
	}
	if _, err := NewDesignMatrix(x, []float64{0, 0, 0, 0, 2, 1, 2, 2, 2}, 3, false); !errors.Is(err, ErrKnots) {
		t.Error("unsorted knots")
	}
}

func TestMakeInterpSplineOptions(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5}
	y := []float64{0, 1, 0, 1, 0, 1}
	bad := func(name string, x, y []float64, k int, o *SplineOptions, want error) {
		t.Helper()
		_, err := MakeInterpSpline(x, y, k, o)
		isErr(t, name, err, want)
	}
	bad("empty", nil, nil, 3, nil, ErrShape)
	bad("mismatch", x, y[:3], 3, nil, ErrShape)
	bad("NaN", x, []float64{0, 1, math.NaN(), 1, 0, 1}, 3, nil, ErrNonFinite)
	bad("duplicate", []float64{0, 1, 1, 2, 3, 4}, y, 3, nil, ErrNotIncreasing)
	bad("negative k", x, y, -1, nil, ErrShape)
	bad("periodic mismatch", x, y, 3, &SplineOptions{Periodic: true}, ErrPeriodic)
	yp := []float64{0, 1, 0, 1, 0, 0}
	bad("periodic with knots", x, yp, 3, &SplineOptions{Periodic: true, T: []float64{0, 0, 0, 0, 3, 5, 5, 5, 5}}, ErrBC)
	bad("periodic with derivatives", x, yp, 3, &SplineOptions{Periodic: true, Left: ClampedDerivs()}, ErrBC)
	bad("k=0 with knots", x, y, 0, &SplineOptions{T: x}, ErrBC)
	bad("k=1 with derivatives", x, y, 1, &SplineOptions{Left: ClampedDerivs()}, ErrBC)
	bad("non-finite knots", x, y, 3, &SplineOptions{T: []float64{0, 0, 0, 0, math.Inf(1), 5, 5, 5, 5, 5}}, ErrNonFinite)
	bad("unsorted knots", x, y, 3, &SplineOptions{T: []float64{0, 0, 0, 0, 3, 2, 5, 5, 5, 5}}, ErrKnots)
	bad("short knots", x, y, 3, &SplineOptions{T: []float64{0, 0, 0, 0, 5, 5, 5, 5}}, ErrKnots)
	bad("knots miss the data", x, y, 3, &SplineOptions{T: []float64{1, 1, 1, 1, 3, 4, 5, 5, 5, 5}}, ErrKnots)
	bad("derivative order", x, y, 3, &SplineOptions{Left: []DerivBC{{4, 0}}, Right: ClampedDerivs()}, ErrBC)
	bad("derivative count", x, y, 3, &SplineOptions{Left: ClampedDerivs()}, ErrBC)
	// knots whose collocation matrix falls outside the band
	bad("band violation", []float64{0, 1, 2, 3, 4, 5}, y, 1, &SplineOptions{T: []float64{0, 0, 0.1, 0.2, 0.3, 0.4, 5, 5}}, ErrBC)
	// Schoenberg-Whitney violation: two data points between three knots of degree 0 span
	bad("singular collocation", []float64{0, 1, 2, 3, 4}, []float64{0, 1, 2, 3, 4}, 1, &SplineOptions{T: []float64{0, 0, 2, 2.5, 3, 4, 4}}, ErrSingular)
	// k = 0 and k = 1 shortcuts, with and without periodic
	b0, err := MakeInterpSpline(x, y, 0, nil)
	if err != nil || b0.At(2.5) != 0 || b0.At(3) != 1 || b0.At(5) != 1 {
		t.Errorf("k=0: %v", err)
	}
	b1, err := MakeInterpSpline(x, yp, 1, &SplineOptions{Periodic: true})
	if err != nil || b1.Extrap != Wrap {
		t.Errorf("k=1 periodic: %v", err)
	}
	eq(t, "k=1", b1.At(1.5), 0.5, 1e-15)
	// derivative conditions
	b, err := MakeInterpSpline(x, y, 3, &SplineOptions{Left: []DerivBC{{1, 2}}, Right: []DerivBC{{2, -1}}})
	if err != nil {
		t.Fatal(err)
	}
	d1 := make([]float64, 1)
	b.EvalNu(d1, []float64{0}, 1)
	eq(t, "left slope", d1[0], 2, 1e-12)
	b.EvalNu(d1, []float64{5}, 2)
	eq(t, "right curvature", d1[0], -1, 1e-12)
	// even degree knots (midpoints) and periodic variants of both parities
	for _, k := range []int{2, 4} {
		if _, err := MakeInterpSpline(x, y, k, nil); err != nil {
			t.Errorf("k=%d: %v", k, err)
		}
	}
	for _, k := range []int{2, 3, 4, 5} {
		for _, n := range []int{3, 4, 5, 9, 40} {
			xs := make([]float64, n)
			ys := make([]float64, n)
			for i := range xs {
				xs[i] = float64(i) + 0.2*math.Sin(float64(3*i))
				ys[i] = math.Sin(2 * math.Pi * float64(i) / float64(n-1))
			}
			ys[n-1] = ys[0]
			if n > k {
				xs[n-1] = float64(n - 1) // keep the period equal to the knot span
			}
			sp, err := MakeInterpSpline(xs, ys, k, &SplineOptions{Periodic: true})
			if err != nil {
				t.Errorf("periodic k=%d n=%d: %v", k, n, err)
				continue
			}
			for i := range xs {
				eq(t, "periodic interpolates", sp.At(xs[i]), ys[i], 1e-9)
			}
			eq(t, "periodic wraps", sp.At(xs[0]+(xs[n-1]-xs[0])), sp.At(xs[0]), 1e-9)
		}
	}
}

func TestLsqAndSplrepErrors(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	y := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	tk := []float64{0, 0, 0, 0, 3, 7, 7, 7, 7}
	if _, err := MakeLsqSpline(x, y, tk, 3, nil); err != nil {
		t.Fatal(err)
	}
	w := []float64{1, 2, 1, 2, 1, 2, 1, 2}
	if _, err := MakeLsqSpline(x, y, tk, 3, w); err != nil {
		t.Fatal(err)
	}
	bad := func(name string, x, y, tk []float64, k int, w []float64, want error) {
		t.Helper()
		_, err := MakeLsqSpline(x, y, tk, k, w)
		isErr(t, name, err, want)
	}
	bad("empty", nil, nil, tk, 3, nil, ErrShape)
	bad("weights length", x, y, tk, 3, w[:3], ErrShape)
	bad("few knots", x, y, tk[:4], 3, nil, ErrShape)
	bad("negative k", x, y, tk, -1, nil, ErrShape)
	bad("NaN", x, []float64{math.NaN(), 1, 0, 1, 0, 1, 0, 1}, tk, 3, nil, ErrNonFinite)
	bad("unsorted x", []float64{0, 2, 1, 3, 4, 5, 6, 7}, y, tk, 3, nil, ErrNotIncreasing)
	bad("unsorted t", x, y, []float64{0, 0, 0, 0, 5, 3, 7, 7, 7, 7}, 3, nil, ErrKnots)
	bad("t does not cover x", x, y, []float64{1, 1, 1, 1, 3, 7, 7, 7, 7}, 3, nil, ErrKnots)
	bad("too many knots", x[:5], y[:5], []float64{0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4}, 3, nil, ErrKnots)
	bad("negative weight", x, y, tk, 3, []float64{1, 1, 1, -1, 1, 1, 1, 1}, ErrShape)
	// no data in the span of an interior B-spline: the normal equations are singular
	bad("singular", x[:6], y[:6], []float64{0, 0, 0, 0, 5, 5, 5}, 2, nil, ErrSingular)
}

func TestSplrepOptionsAndErrors(t *testing.T) {
	r := newRNG(9)
	x := sortedX(r, 40, 0, 10, true)
	y := smoothY(r, x, 0.2)
	sm, err := MakeSplrep(x, y, 3, 0, nil)
	if err != nil || sm.Ier != 0 {
		t.Fatalf("interpolation: %v", err)
	}
	bad := func(name string, x, y []float64, k int, s float64, o *SplrepOptions, want error) {
		t.Helper()
		_, err := MakeSplrep(x, y, k, s, o)
		isErr(t, name, err, want)
	}
	bad("s=0 with weights", x, y, 3, 0, &SplrepOptions{W: ones(40)}, ErrBC)
	bad("too few points", x[:3], y[:3], 3, 1, nil, ErrShape)
	bad("negative s", x, y, 3, -1, nil, ErrShape)
	bad("mismatch", x, y[:5], 3, 1, nil, ErrShape)
	bad("weights length", x, y, 3, 1, &SplrepOptions{W: ones(3)}, ErrShape)
	bad("knots and nest", x, y, 3, 1, &SplrepOptions{T: []float64{0, 0, 0, 0, 10, 10, 10, 10}, Nest: 20}, ErrShape)
	bad("nest too small", x, y, 3, 1, &SplrepOptions{Nest: 5}, ErrShape)
	bad("NaN y", x, append([]float64{math.NaN()}, y[1:]...), 3, 1, nil, ErrNonFinite)
	bad("NaN s", x, y, 3, math.NaN(), nil, ErrNonFinite)
	bad("NaN weight", x, y, 3, 1, &SplrepOptions{W: append([]float64{math.NaN()}, ones(39)...)}, ErrNonFinite)
	bad("unsorted", append([]float64{20}, x[1:]...), y, 3, 1, nil, ErrNotIncreasing)
	bad("negative weight", x, y, 3, 1, &SplrepOptions{W: append([]float64{-1}, ones(39)...)}, ErrShape)
	bad("zero weights", x, y, 3, 1, &SplrepOptions{W: make([]float64, 40)}, ErrShape)
	bad("bad user knots", x, y, 3, 1, &SplrepOptions{T: []float64{0, 1}}, ErrKnots)
	bad("unsorted user knots", x, y, 3, 1, &SplrepOptions{T: []float64{0, 0, 0, 0, 7, 5, 10, 10, 10, 10}}, ErrKnots)
	// a polynomial of low degree needs no interior knot: the fit is the LSQ polynomial
	poly := make([]float64, len(x))
	for i, v := range x {
		poly[i] = 1 + v - 0.1*v*v
	}
	sm, err = MakeSplrep(x, poly, 3, 10, nil)
	if err != nil || len(sm.Spline.T) != 8 {
		t.Fatalf("polynomial data: %v knots=%d", err, len(sm.Spline.T))
	}
	// user knots, with weights and bounds, nest cap
	sm, err = MakeSplrep(x, y, 3, 1, &SplrepOptions{T: []float64{0, 0, 0, 0, 3, 6, 10, 10, 10, 10}, W: ones(40)})
	if err != nil {
		t.Fatal(err)
	}
	if sm.Fp <= 0 {
		t.Error("fp")
	}
	sm, err = MakeSplrep(x, y, 3, 1e-3, &SplrepOptions{Nest: 12, Bounds: &[2]float64{x[0], x[39]}})
	if err != nil || len(sm.Spline.T) > 14 {
		t.Errorf("nest cap: %v %d", err, len(sm.Spline.T))
	}
	// the iteration that stalls: a huge weight on one point
	w := ones(40)
	w[7] = 1e6
	if _, err := MakeSplrep(x, y, 3, 0.01, &SplrepOptions{W: w}); err != nil {
		t.Errorf("heavy weight: %v", err)
	}
}

func TestFitpackInternals(t *testing.T) {
	// fpknot with no interval holding two points falls back to the middle point
	if got := fpknot([]float64{0, 3}, []float64{0, 0, 0, 0, 3, 3, 3, 3}, 3, []float64{1, 1}); got != 0 {
		t.Errorf("fpknot fallback: %v", got)
	}
	if got := fpknot([]float64{0, 1, 2, 3}, []float64{0, 0, 0, 0, 1.5, 3, 3, 3, 3}, 3, []float64{0, 5, 0, 0}); got != 1 {
		t.Errorf("fpknot: %v", got)
	}
	inf := math.Inf(1)
	// the status codes of the rational iteration
	if ier, _ := rootRati(func(p float64) float64 { return math.NaN() }, 1, [2]float64{0, 1}, [2]float64{inf, -1}, 1e-3); ier != 1 {
		t.Errorf("NaN gives ier 1: %d", ier)
	}
	if ier, _ := rootRati(func(p float64) float64 { return 1 }, 1, [2]float64{0, 1}, [2]float64{inf, -1}, 1e-3); ier != 3 {
		t.Errorf("a constant never converges: ier %d", ier)
	}
	if ier, p := rootRati(func(p float64) float64 { return 2 - p }, 1, [2]float64{0, 3}, [2]float64{inf, -1}, 1e-12); ier != 0 || math.Abs(p-2) > 1e-9 {
		t.Errorf("a decreasing line converges: ier %d p %v", ier, p)
	}
	// f not decreasing: 2
	vals := []float64{5, 4, 6}
	idx := 0
	if ier, _ := rootRati(func(p float64) float64 { idx++; return vals[(idx-1)%3] }, 1, [2]float64{0, 10}, [2]float64{inf, -10}, 1e-12); ier != 2 {
		t.Errorf("a function that increases again gives ier 2: %d", ier)
	}
	// p too small with a finite upper end of the bracket
	if ier, _ := rootRati(func(p float64) float64 { return 2 }, 1, [2]float64{0, 0.5}, [2]float64{100, -3}, 1e-12); ier != 3 {
		t.Errorf("p too small: %d", ier)
	}
	// p too large
	if ier, _ := rootRati(func(p float64) float64 { return -2 + 0*p }, 1, [2]float64{0, 5}, [2]float64{inf, -1}, 1e-12); ier != 3 {
		t.Errorf("p too large: %d", ier)
	}
	// p too large with a lower bound above the shrunk p
	if ier, _ := rootRati(func(p float64) float64 { return -2 }, 1, [2]float64{0.9, 5}, [2]float64{inf, -1}, 1e-12); ier != 3 {
		t.Errorf("p below p1: %d", ier)
	}
	// repeated abscissae leave the least-squares problem singular
	_, err := MakeSplrep([]float64{0, 0, 0, 0, 1}, []float64{0, 1, 2, 3, 4}, 3, 0.1, nil)
	isErr(t, "singular smoothing", err, ErrSingular)
	// the helper's recover must not swallow foreign panics
	defer func() {
		if recover() == nil {
			t.Error("a foreign panic must propagate")
		}
	}()
	func() {
		var err error
		defer guard(&err)
		panic("other")
	}()
}
