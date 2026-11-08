package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/go-approx/interp"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with scipy.interpolate (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, inventory
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataParity -perf.parity .       both sides get the same bytes (needs uv)
//	go test -run TestWorkParity -perf.parity .       both sides do the same work: same knots (needs uv)
//	go test -run TestRefSmoke -perf.smoke .          every SciPy call runs once (needs uv)
//
// What is compared. Every row times one call on both sides, on the same data:
// the Go body is the public function or method, the SciPy body is the
// scipy.interpolate call it ports (SciPy 1.18.1 and NumPy 2.5.4, pinned with
// their hashes in refs.txt). For a method row the object is built outside the
// timed loop on both sides. There is no callback in this package: the
// interpolants take arrays, so no Python function is on the SciPy side of any
// row. The arrays have n = 10 000 points (the design matrix 1 000 points, which
// is 8 MB dense), the 2-D rows a 100 x 80 grid and 10 000 scattered queries.

const n = 10_000

const (
	gold    = 0.6180339887498949
	plastic = 0.7548776662466927
)

func frac(i int, m float64) float64 { return math.Mod(float64(i)*m, 1) }

// The 1-D data: jittered nodes on [0, 10], a smooth signal with a little noise,
// sorted queries inside the nodes. Every product is wrapped in float64 so that
// arm64 does not fuse it with the addition: the bytes are NumPy's.
func nodes(m int) []float64 {
	x := make([]float64, m)
	h := 10 / float64(m-1)
	for i := range x {
		x[i] = float64(float64(i)*10/float64(m-1)) + float64(float64(0.4*(frac(i, gold)-0.5))*h)
	}
	return x
}

func signal(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		t := v / 10
		y[i] = float64(float64(float64(4*t)*(1-t))*(1+t)) + float64(0.01*(frac(i, plastic)-0.5))
	}
	return y
}

func queries(m int) []float64 {
	q := make([]float64, m)
	for i := range q {
		q[i] = float64(10*(float64(i)+0.5)) / float64(m)
	}
	return q
}

// clamped knots on [0, 10] with m coefficients (cubic), and the LSQ knots on [-1, 11].
func knots(m int) []float64 {
	t := make([]float64, 0, m+4)
	t = append(t, 0, 0, 0, 0)
	for i := 1; i < m-3; i++ {
		t = append(t, float64(10*float64(i))/float64(m-3))
	}
	return append(t, 10, 10, 10, 10)
}

func lsqKnots() []float64 {
	t := []float64{-1, -1, -1, -1}
	for i := 1; i < 199; i++ {
		t = append(t, -1+float64(12*float64(i))/199)
	}
	return append(t, 11, 11, 11, 11)
}

func coefs(m int) []float64 {
	c := make([]float64, m)
	for i := range c {
		c[i] = frac(i, plastic) - 0.5
	}
	return c
}

// The PPoly coefficients in SciPy's layout: c[r*m+i], r = 0 the highest power.
func ppolyC(k, m int) []float64 {
	c := make([]float64, k*m)
	for i := range c {
		c[i] = frac(i, gold) - 0.5
	}
	return c
}

// The 2-D data: a 100 x 80 grid, z = a_i b_j, queries in (0.1, 9.9).
const mx, my = 100, 80

func axis(m int) []float64 {
	a := make([]float64, m)
	for j := range a {
		a[j] = float64(10*float64(j)) / float64(m-1)
	}
	return a
}

func gridZ(ax, ay []float64) []float64 {
	z := make([]float64, len(ax)*len(ay))
	for i, a := range ax {
		ai := float64(a*(10-a)) / 25
		for j, b := range ay {
			z[i*len(ay)+j] = float64(ai * (1 + b/10))
		}
	}
	return z
}

func scatter(m int, mult float64) []float64 {
	q := make([]float64, m)
	for i := range q {
		q[i] = float64(frac(i, mult)*9.8) + 0.1
	}
	return q
}

// Python expressions of the same data (n is the size the worker passes).
const (
	pyX    = "i*10.0/(n-1) + 0.4*(np.mod(i*%v,1.0)-0.5)*(10.0/(n-1))"
	pyY    = "4.0*(x/10.0)*(1.0-x/10.0)*(1.0+x/10.0) + 0.01*(np.mod(i*%v,1.0)-0.5)"
	pyQ    = "10.0*(i+0.5)/n"
	pyKnot = "np.concatenate([[0.0]*4, 10.0*np.arange(1,n-3)/(n-3), [10.0]*4])"
	pyLsq  = "np.concatenate([[-1.0]*4, -1.0 + 12.0*np.arange(1,199)/199, [11.0]*4])"
)

var (
	pyG = fmt.Sprintf("%v", gold)
	pyP = fmt.Sprintf("%v", plastic)
)

func p1(extra string) string {
	return "(lambda i: (lambda x: [x, " + fmt.Sprintf(pyY, plastic) + ", " + pyQ + extra + "])(" +
		fmt.Sprintf(pyX, gold) + "))(np.arange(n, dtype=np.float64))"
}

var (
	pyD1   = p1("")                 // x, y, q
	pyD1t  = p1(", " + pyLsq)       // x, y, q, t
	pyDppo = "(lambda d: [d[0], (np.mod(np.arange(4*(n-1), dtype=np.float64)*" + pyG + ", 1.0)-0.5).reshape(4, n-1)])(" + pyD1 + ")"
	pyDB   = "(lambda i: [" + pyKnot + ", np.mod(i*" + pyP + ",1.0)-0.5, " + pyQ + "])(np.arange(n, dtype=np.float64))"
	pyDDM  = "(lambda i: [" + pyQ + ", " + pyKnot + "])(np.arange(n, dtype=np.float64))"
	pyAX   = "10.0*np.arange(100.0)/99"
	pyAY   = "10.0*np.arange(80.0)/79"
	pyQX   = "np.mod(np.arange(n, dtype=np.float64)*" + pyG + ",1.0)*9.8+0.1"
	pyQY   = "np.mod(np.arange(n, dtype=np.float64)*" + pyP + ",1.0)*9.8+0.1"
	pyZ    = "np.outer(ax*(10.0-ax)/25.0, 1.0+ay/10.0)"
	pyD2g  = "(lambda ax, ay, qx, qy: [ax, ay, " + pyZ + ", np.column_stack([qx, qy])])(" + pyAX + ", " + pyAY + ", " + pyQX + ", " + pyQY + ")"
	pyD2e  = "(lambda ax, ay, qx, qy: [ax, ay, " + pyZ + ", qx, qy])(" + pyAX + ", " + pyAY + ", " + pyQX + ", " + pyQY + ")"
)

// fam is one data family: the Python expression of its arguments, their names in
// the reference lambda, and the Go arrays (the bytes the data-parity test hashes).
type fam struct {
	setup, names string
	data         [][]float64
}

func f1() fam {
	x := nodes(n)
	return fam{pyD1, "x, y, q", [][]float64{x, signal(x), queries(n)}}
}

func f1t() fam {
	f := f1()
	f.setup, f.names = pyD1t, "x, y, q, t"
	f.data = append(f.data, lsqKnots())
	return f
}

func fppoly() fam {
	x := nodes(n)
	return fam{pyDppo, "x, cs", [][]float64{x, ppolyC(4, n-1)}}
}

func fB() fam {
	return fam{pyDB, "t, c, q", [][]float64{knots(n), coefs(n), queries(n)}}
}

func fDM() fam {
	const m = 1000
	return fam{pyDDM, "q, t", [][]float64{queries(m), knots(m)}}
}

func f2(eval bool) fam {
	ax, ay := axis(mx), axis(my)
	qx, qy := scatter(n, gold), scatter(n, plastic)
	z := gridZ(ax, ay)
	if eval {
		return fam{pyD2e, "ax, ay, z, qx, qy", [][]float64{ax, ay, z, qx, qy}}
	}
	xi := make([]float64, 0, 2*n)
	for i := range qx {
		xi = append(xi, qx[i], qy[i])
	}
	return fam{pyD2g, "ax, ay, z, xi", [][]float64{ax, ay, z, xi}}
}

var (
	sink    float64
	sinkS   []float64
	sinkErr error
	sinkAny any
)

// row is one case.
type row struct {
	name, fn, ref, parity string
	f                     fam
	n                     int
	body                  string // a constructor: the call, with the names of the family
	obj, use              string // a method: the object built outside the loop, the call timed
	mk                    func() func()
}

const imp = `__import__("scipy.interpolate", fromlist=["_"])`

func (r row) call() string {
	if r.body != "" {
		return "(lambda I: lambda " + r.f.names + ": " + r.body + ")(" + imp + ")"
	}
	return "(lambda I, " + r.f.names + ": (lambda s: lambda *_: " + r.use + ")(" + r.obj + "))(" + imp + ", *" + r.f.setup + ")"
}

func (r row) size() int {
	if r.n != 0 {
		return r.n
	}
	return n
}

func rows() []row {
	var rs []row
	add := func(r row) { rs = append(rs, r) }
	const ex, tol = refbench.ParityExact, refbench.ParityTolerance

	a := f1()
	x, y, q := a.data[0], a.data[1], a.data[2]
	dst := make([]float64, n)
	half := make([]float64, n)
	for i, v := range y {
		half[i] = 0.5 * v
	}

	ctor := func(name, fn, ref, parity string, f fam, body string, mk func() func()) {
		add(row{name: name, fn: fn, ref: ref, parity: parity, f: f, body: body, mk: mk})
	}
	meth := func(name, fn, ref, parity string, f fam, obj, use string, mk func() func()) {
		add(row{name: name, fn: fn, ref: ref, parity: parity, f: f, obj: obj, use: use, mk: mk})
	}

	// Cubic splines and the other piecewise cubics (PPoly).
	for _, c := range []struct{ tag, bc, py string }{
		{"not-a-knot", "not-a-knot", ""},
		{"natural", "natural", ", bc_type='natural'"},
		{"clamped", "clamped", ", bc_type='clamped'"},
	} {
		ctor("NewCubicSpline "+c.tag, "NewCubicSpline", "CubicSpline", tol, a, "I.CubicSpline(x, y"+c.py+")",
			func() func() { return func() { sinkAny, sinkErr = interp.NewCubicSpline(x, y, c.bc) } })
	}
	ctor("NewCubicSplineBC D1,D2", "NewCubicSplineBC", "CubicSpline bc_type=((1,.5),(2,.25))", tol, a,
		"I.CubicSpline(x, y, bc_type=((1, 0.5), (2, 0.25)))",
		func() func() {
			return func() {
				sinkAny, sinkErr = interp.NewCubicSplineBC(x, y, interp.BC{Kind: interp.D1, Value: 0.5}, interp.BC{Kind: interp.D2, Value: 0.25})
			}
		})
	ctor("NewCubicHermite", "NewCubicHermite", "CubicHermiteSpline", ex, a, "I.CubicHermiteSpline(x, y, 0.5*y)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewCubicHermite(x, y, half) } })
	ctor("NewPchip", "NewPchip", "PchipInterpolator", tol, a, "I.PchipInterpolator(x, y)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewPchip(x, y) } })
	ctor("NewAkima", "NewAkima", "Akima1DInterpolator", tol, a, "I.Akima1DInterpolator(x, y)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewAkima(x, y, false) } })
	ctor("NewAkima makima", "NewAkima", "Akima1DInterpolator method=makima", tol, a, "I.Akima1DInterpolator(x, y, method='makima')",
		func() func() { return func() { sinkAny, sinkErr = interp.NewAkima(x, y, true) } })

	cs, _ := interp.NewCubicSpline(x, y, "not-a-knot")
	const obj = "I.CubicSpline(x, y)"
	meth("PPoly.EvalNu", "PPoly.EvalNu", "CubicSpline.__call__(q, 1)", tol, a, obj, "s(q, 1)",
		func() func() { return func() { cs.EvalNu(dst, q, 1); sink = dst[0] } })
	meth("PPoly.EvalAll", "PPoly.EvalAll", "CubicSpline.__call__(q)", tol, a, obj, "s(q)",
		func() func() { return func() { cs.EvalAll(dst, q); sink = dst[0] } })
	meth("PPoly.At", "PPoly.At", "CubicSpline.__call__(0.37)", tol, a, obj, "s(0.37)",
		func() func() { return func() { sink = cs.At(0.37) } })
	meth("PPoly.Derivative", "PPoly.Derivative", "PPoly.derivative", ex, a, obj, "s.derivative()",
		func() func() { return func() { sinkAny = cs.Derivative(1) } })
	meth("PPoly.Antiderivative", "PPoly.Antiderivative", "PPoly.antiderivative", ex, a, obj, "s.antiderivative()",
		func() func() { return func() { sinkAny = cs.Antiderivative(1) } })
	meth("PPoly.Integrate", "PPoly.Integrate", "PPoly.integrate", tol, a, obj, "s.integrate(1.0, 9.0)",
		func() func() { return func() { sink = cs.Integrate(1, 9) } })

	pp := fppoly()
	cG := make([]float64, 4*(n-1)) // piece-major, ascending powers
	for r := 0; r < 4; r++ {
		for i := 0; i < n-1; i++ {
			cG[i*4+3-r] = pp.data[1][r*(n-1)+i]
		}
	}
	ctor("NewPPoly", "NewPPoly", "PPoly", ex, pp, "I.PPoly(cs, x)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewPPoly(x, 4, cG, interp.Extend) } })

	// B-splines.
	b := fB()
	t, c, qb := b.data[0], b.data[1], b.data[2]
	ctor("NewBSpline", "NewBSpline", "BSpline", ex, b, "I.BSpline(t, c, 3)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewBSpline(t, c, 3) } })
	bs, _ := interp.NewBSpline(t, c, 3)
	const bobj = "I.BSpline(t, c, 3)"
	meth("BSpline.EvalNu", "BSpline.EvalNu", "BSpline.__call__(q, nu=1)", tol, b, bobj, "s(q, 1)",
		func() func() { return func() { bs.EvalNu(dst, qb, 1); sink = dst[0] } })
	meth("BSpline.EvalAll", "BSpline.EvalAll", "BSpline.__call__(q)", tol, b, bobj, "s(q)",
		func() func() { return func() { bs.EvalAll(dst, qb); sink = dst[0] } })
	meth("BSpline.At", "BSpline.At", "BSpline.__call__(0.37)", tol, b, bobj, "s(0.37)",
		func() func() { return func() { sink = bs.At(0.37) } })
	meth("BSpline.Derivative", "BSpline.Derivative", "BSpline.derivative", ex, b, bobj, "s.derivative()",
		func() func() { return func() { sinkAny, sinkErr = bs.Derivative(1) } })
	meth("BSpline.Antiderivative", "BSpline.Antiderivative", "BSpline.antiderivative", ex, b, bobj, "s.antiderivative()",
		func() func() { return func() { sinkAny = bs.Antiderivative(1) } })
	meth("BSpline.Integrate", "BSpline.Integrate", "BSpline.integrate", tol, b, bobj, "s.integrate(1.0, 9.0)",
		func() func() { return func() { sink = bs.Integrate(1, 9) } })

	dm := fDM()
	dq, dt := dm.data[0], dm.data[1]
	add(row{name: "NewDesignMatrix", fn: "NewDesignMatrix", ref: "BSpline.design_matrix", parity: tol, f: dm, n: 1000,
		body: "I.BSpline.design_matrix(q, t, 3)",
		mk: func() func() {
			return func() { sinkAny, sinkErr = interp.NewDesignMatrix(dq, dt, 3, false) }
		}})
	dmat, _ := interp.NewDesignMatrix(dq, dt, 3, false)
	add(row{name: "DesignMatrix.Dense", fn: "DesignMatrix.Dense", ref: "design_matrix(...).toarray", parity: ex, f: dm, n: 1000,
		obj: "I.BSpline.design_matrix(q, t, 3)", use: "s.toarray()",
		mk: func() func() { return func() { sinkS = dmat.Dense() } }})

	ctor("MakeInterpSpline k=3", "MakeInterpSpline", "make_interp_spline(k=3)", tol, a, "I.make_interp_spline(x, y, k=3)",
		func() func() { return func() { sinkAny, sinkErr = interp.MakeInterpSpline(x, y, 3, nil) } })
	ctor("MakeInterpSpline k=5", "MakeInterpSpline", "make_interp_spline(k=5)", tol, a, "I.make_interp_spline(x, y, k=5)",
		func() func() { return func() { sinkAny, sinkErr = interp.MakeInterpSpline(x, y, 5, nil) } })
	lq := f1t()
	lt := lq.data[3]
	ctor("MakeLsqSpline", "MakeLsqSpline", "make_lsq_spline(k=3)", tol, lq, "I.make_lsq_spline(x, y, t, k=3)",
		func() func() { return func() { sinkAny, sinkErr = interp.MakeLsqSpline(x, y, lt, 3, nil) } })
	// The smoothing factor of both splrep rows: the noise variance (0.01^2/12) times n.
	const s = 0.0000083333 * n
	ctor("Splrep", "Splrep", "splrep(k=3, s)", tol, a, fmt.Sprintf("I.splrep(x, y, k=3, s=%v)", s),
		func() func() { return func() { sinkAny, sinkErr = interp.Splrep(x, y, 3, s, nil) } })
	ctor("MakeSplrep", "MakeSplrep", "make_splrep(k=3, s)", tol, a, fmt.Sprintf("I.make_splrep(x, y, k=3, s=%v)", s),
		func() func() { return func() { sinkAny, sinkErr = interp.MakeSplrep(x, y, 3, s, nil) } })

	// interp1d, numpy.interp and MATLAB interp1.
	for _, k := range []struct {
		tag  string
		kind interp.Kind
		py   string
	}{{"linear", interp.Linear, ""}, {"nearest", interp.Nearest, ", kind='nearest'"}} {
		ctor("NewInterp1D "+k.tag, "NewInterp1D", "interp1d", ex, a, "I.interp1d(x, y"+k.py+", bounds_error=False)",
			func() func() { return func() { sinkAny, sinkErr = interp.NewInterp1D(x, y, k.kind, nil) } })
		f, _ := interp.NewInterp1D(x, y, k.kind, nil)
		o := "I.interp1d(x, y" + k.py + ", bounds_error=False)"
		meth("Interp1D.EvalAll "+k.tag, "Interp1D.EvalAll", "interp1d.__call__(q)", ex, a, o, "s(q)",
			func() func() { return func() { sinkErr = f.EvalAll(dst, q); sink = dst[0] } })
		if k.tag == "linear" {
			meth("Interp1D.At", "Interp1D.At", "interp1d.__call__(0.37)", ex, a, o, "s(0.37)",
				func() func() { return func() { sink, sinkErr = f.At(0.37) } })
		}
	}
	ctor("Interp", "Interp", "numpy.interp", ex, a, "np.interp(q, x, y, y[0], y[-1])",
		func() func() { return func() { sinkErr = interp.Interp(dst, q, x, y, y[0], y[n-1]) } })
	ctor("Interp1 linear", "Interp1", "interp1d(q)", ex, a, "I.interp1d(x, y, bounds_error=False)(q)",
		func() func() { return func() { sinkS, sinkErr = interp.Interp1(x, y, q, "linear", interp.Extrap{}) } })
	ctor("Interp1 pchip", "Interp1", "PchipInterpolator(q)", tol, a, "I.PchipInterpolator(x, y)(q)",
		func() func() { return func() { sinkS, sinkErr = interp.Interp1(x, y, q, "pchip", interp.Extrap{}) } })

	// Tensor-product interpolants on the 100 x 80 grid.
	g := f2(false)
	ax, ay, z, xi := g.data[0], g.data[1], g.data[2], g.data[3]
	axes := [][]float64{ax, ay}
	out := make([]float64, n)
	for _, m := range []string{"linear", "nearest", "cubic"} {
		py := "I.RegularGridInterpolator((ax, ay), z, method='" + m + "', bounds_error=False)"
		ctor("NewRGI "+m, "NewRGI", "RegularGridInterpolator "+m, tol, g, py,
			func() func() { return func() { sinkAny, sinkErr = interp.NewRGI(axes, z, m, nil) } })
		r, _ := interp.NewRGI(axes, z, m, nil)
		meth("RGI.Eval "+m, "RGI.Eval", "RegularGridInterpolator.__call__ "+m, tol, g, py, "s(xi)",
			func() func() { return func() { sinkErr = r.Eval(out, xi); sink = out[0] } })
	}
	ctor("Interpn linear", "Interpn", "interpn linear", tol, g, "I.interpn((ax, ay), z, xi, method='linear', bounds_error=False)",
		func() func() { return func() { sinkS, sinkErr = interp.Interpn(axes, z, xi, "linear", nil) } })
	ctor("MakeNdInterp k=3", "MakeNdInterp", "make_ndbspl(k=3)", tol, g, "I._ndbspline.make_ndbspl((ax, ay), z, k=3)",
		func() func() { return func() { sinkAny, sinkErr = interp.MakeNdInterp(axes, z, 3) } })
	nd, _ := interp.MakeNdInterp(axes, z, 3)
	meth("NdBSpline.EvalNu", "NdBSpline.EvalNu", "NdBSpline.__call__(xi, nu=(1,0))", tol, g,
		"I._ndbspline.make_ndbspl((ax, ay), z, k=3)", "s(xi, nu=np.array([1, 0]))",
		func() func() { return func() { nd.EvalNu(out, xi, []int{1, 0}); sink = out[0] } })

	e := f2(true)
	qx, qy := e.data[3], e.data[4]
	ctor("NewBicubic", "NewBicubic", "RectBivariateSpline", tol, e, "I.RectBivariateSpline(ax, ay, z)",
		func() func() { return func() { sinkAny, sinkErr = interp.NewBicubic(ax, ay, z, 3, 3) } })
	bc, _ := interp.NewBicubic(ax, ay, z, 3, 3)
	const rb = "I.RectBivariateSpline(ax, ay, z)"
	meth("Bicubic.Deriv", "Bicubic.Deriv", "RectBivariateSpline.ev(dx=1)", tol, e, rb, "s.ev(0.37, 0.52, dx=1, dy=0)",
		func() func() { return func() { sink = bc.Deriv(0.37, 0.52, 1, 0) } })
	meth("Bicubic.At", "Bicubic.At", "RectBivariateSpline.ev", tol, e, rb, "s.ev(0.37, 0.52)",
		func() func() { return func() { sink = bc.At(0.37, 0.52) } })
	meth("Bicubic.Ev", "Bicubic.Ev", "RectBivariateSpline.ev(qx, qy)", tol, e, rb, "s.ev(qx, qy)",
		func() func() { return func() { sinkErr = bc.Ev(out, qx, qy); sink = out[0] } })
	grid := make([]float64, mx*my)
	meth("Bicubic.EvalGrid", "Bicubic.EvalGrid", "RectBivariateSpline(ax, ay, dx=1, dy=1)", tol, e, rb,
		"s(ax, ay, dx=1, dy=1)",
		func() func() { return func() { sinkErr = bc.EvalGrid(grid, ax, ay, 1, 1); sink = grid[0] } })
	return rs
}

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		body := r.mk()
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.f.setup, r.call(), r.size())
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: "scipy 1.18.1 interpolate." + r.ref,
			Parity:    r.parity,
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					body()
				}
			},
			Ref: ref,
		})
	}
	return out
}

var (
	parityFlag = flag.Bool("perf.parity", false, "TestDataParity and TestWorkParity: compare with SciPy's (starts the worker through uv)")
	smokeFlag  = flag.Bool("perf.smoke", false, "TestRefSmoke: run every SciPy call once")
)

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-approx/interp"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

func digest(arrs [][]float64) string {
	h := sha256.New()
	for _, a := range arrs {
		for _, v := range a {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestDataParity checks, for every row, that the Go arrays and SciPy's are the
// same bytes: a ratio between two functions on different data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	seen := map[string]bool{}
	for _, r := range rows() {
		key := fmt.Sprint(r.f.setup, r.size())
		if seen[key] {
			continue
		}
		seen[key] = true
		got, err := w.DataHash(r.f.setup, r.size())
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		if want := digest(r.f.data); got != want {
			t.Errorf("%s: SciPy data %s, Go data %s", r.name, got, want)
		}
	}
}

// TestWorkParity checks that both sides do the same work where the work is a
// choice of the algorithm: the knots of the two smoothing fits (the number of
// knots is the work of a spline fit). The Python side builds the fit in the
// setup expression and hands its knot count back as the argument to hash.
func TestWorkParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	a := f1()
	x, y := a.data[0], a.data[1]
	const s = 0.0000083333 * n
	for _, c := range []struct {
		name string
		py   string
		got  func() int
	}{
		{"Splrep", fmt.Sprintf("(lambda x, y: [np.array([float(len(I.splrep(x, y, k=3, s=%v)[0]))])])(*(%s)[:2])", s, pyD1),
			func() int { f, _ := interp.Splrep(x, y, 3, s, nil); return len(f.Spline.T) }},
		{"MakeSplrep", fmt.Sprintf("(lambda x, y: [np.array([float(len(I.make_splrep(x, y, k=3, s=%v).t))])])(*(%s)[:2])", s, pyD1),
			func() int { f, _ := interp.MakeSplrep(x, y, 3, s, nil); return len(f.Spline.T) }},
	} {
		setup := `(lambda I: ` + c.py + `)(` + imp + `)`
		got, err := w.DataHash(setup, n)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if want := digest([][]float64{{float64(c.got())}}); got != want {
			t.Errorf("%s: SciPy places a different number of knots than Go (%d)", c.name, c.got())
		}
	}
}

// TestRefSmoke runs every SciPy call once: a typo in a reference expression is a
// failure here, not in a timing run.
func TestRefSmoke(t *testing.T) {
	if !*smokeFlag {
		t.Skip("set -perf.smoke (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, c := range cases(w) {
		if _, err := c.Ref(1); err != nil {
			t.Errorf("%s: %v", c.Name, err)
		}
	}
}
