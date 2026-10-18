//go:build oracle

package interp

// Performance against SciPy/NumPy on one core. Both sides time the same
// operations on same-shaped data (jittered nodes on [0, 10], a smooth noisy
// signal, uniformly spread queries). Rounds are interleaved (Go, SciPy, Go,
// SciPy, ...) and the median of the rounds is reported; the host's load
// average is recorded with every round because the machine is shared.
//
//	GOWORK=off go test -tags oracle -run TestPerf -timeout 6h -v
//
// writes docs/perf.md.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

const perfPython = `
import sys, json, time, math
import numpy as np
import scipy
from scipy import interpolate as I
import scipy.sparse.linalg as ssl
def timeit(f, mintime=0.05, maxreps=2000):
    f()
    t0 = time.perf_counter(); f(); one = time.perf_counter() - t0
    reps = max(1, min(maxreps, int(mintime / max(one, 1e-9))))
    t0 = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t0) / reps
def data(N):
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, N) + (rng.random(N) - .5) * 0.8 * 10 / max(N - 1, 1)
    x[0] = 0; x[-1] = 10
    y = np.sin(1.7 * x) + 0.3 * np.cos(5.1 * x) + 0.05 * (rng.random(N) - .5)
    xq = np.sort(rng.random(N) * 10)
    return x, y, xq, rng
d = json.load(sys.stdin)
op, N = d['op'], int(d['n'])
x, y, xq, rng = data(N)
out = {}
if op == 'cubic-nak':
    out['construct'] = timeit(lambda: I.CubicSpline(x, y)); s = I.CubicSpline(x, y); out['eval'] = timeit(lambda: s(xq))
elif op == 'cubic-nak-shuffled':
    q = xq.copy(); rng.shuffle(q); s = I.CubicSpline(x, y); out['eval'] = timeit(lambda: s(q))
elif op == 'cubic-periodic':
    yp = y.copy(); yp[-1] = yp[0]
    out['construct'] = timeit(lambda: I.CubicSpline(x, yp, bc_type='periodic'))
elif op == 'pchip':
    out['construct'] = timeit(lambda: I.PchipInterpolator(x, y)); s = I.PchipInterpolator(x, y); out['eval'] = timeit(lambda: s(xq))
elif op == 'akima':
    out['construct'] = timeit(lambda: I.Akima1DInterpolator(x, y)); s = I.Akima1DInterpolator(x, y); out['eval'] = timeit(lambda: s(xq))
elif op == 'makima':
    out['construct'] = timeit(lambda: I.Akima1DInterpolator(x, y, method='makima'))
elif op == 'bspline3':
    out['construct'] = timeit(lambda: I.make_interp_spline(x, y, k=3)); s = I.make_interp_spline(x, y, k=3); out['eval'] = timeit(lambda: s(xq))
elif op == 'bspline5':
    out['construct'] = timeit(lambda: I.make_interp_spline(x, y, k=5)); s = I.make_interp_spline(x, y, k=5); out['eval'] = timeit(lambda: s(xq))
elif op == 'interp1d-linear':
    out['construct'] = timeit(lambda: I.interp1d(x, y, bounds_error=False)); f = I.interp1d(x, y, bounds_error=False); out['eval'] = timeit(lambda: f(xq))
elif op == 'interp1d-nearest':
    out['construct'] = timeit(lambda: I.interp1d(x, y, kind='nearest', bounds_error=False)); f = I.interp1d(x, y, kind='nearest', bounds_error=False); out['eval'] = timeit(lambda: f(xq))
elif op == 'np.interp':
    out['eval'] = timeit(lambda: np.interp(xq, x, y))
elif op == 'splrep':
    s = N * 1e-3
    out['construct'] = timeit(lambda: I.make_splrep(x, y, k=3, s=s), mintime=0.0); out['legacy'] = timeit(lambda: I.splrep(x, y, k=3, s=s), mintime=0.0)
elif op == 'lsq':
    t = np.r_[[0.] * 4, np.linspace(0, 10, max(N // 50, 2) + 2)[1:-1], [10.] * 4]
    out['construct'] = timeit(lambda: I.make_lsq_spline(x, y, t, k=3))
elif op in ('rgi-linear', 'rgi-cubic', 'rgi-nearest'):
    m = int(round(math.sqrt(N)))
    ax = [np.linspace(0, 10, m), np.linspace(0, 10, m)]
    vals = np.sin(ax[0][:, None]) * np.cos(ax[1][None, :])
    xi = rng.random((N, 2)) * 10
    meth = op.split('-')[1]
    kw = {'solver': ssl.spsolve} if meth == 'cubic' else {}
    out['construct'] = timeit(lambda: I.RegularGridInterpolator(ax, vals, method=meth, **kw), mintime=0.0)
    f = I.RegularGridInterpolator(ax, vals, method=meth, **kw); out['eval'] = timeit(lambda: f(xi), mintime=0.0)
elif op == 'rbs':
    m = int(round(math.sqrt(N)))
    xs = np.linspace(0, 10, m); ys = np.linspace(0, 10, m)
    z = np.sin(xs)[:, None] * np.cos(ys)[None, :]
    xi = rng.random(N) * 10; yi = rng.random(N) * 10
    out['construct'] = timeit(lambda: I.RectBivariateSpline(xs, ys, z), mintime=0.0)
    f = I.RectBivariateSpline(xs, ys, z); out['eval'] = timeit(lambda: f.ev(xi, yi), mintime=0.0)
print(json.dumps(out))
`

func pyTime(t *testing.T, op string, n int) map[string]float64 {
	cmd := exec.Command("uv", "run", "--quiet", "--with", "numpy=="+oracleNumpy, "--with", "scipy=="+oracleScipy, "python", "-c", perfPython)
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME"),
		"OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1"}
	cmd.Stdin = strings.NewReader(fmt.Sprintf(`{"op":%q,"n":%d}`, op, n))
	var so, se bytes.Buffer
	cmd.Stdout, cmd.Stderr = &so, &se
	if err := cmd.Run(); err != nil {
		t.Fatalf("python %s %d: %v\n%s", op, n, err, se.String())
	}
	var out map[string]float64
	if err := json.Unmarshal(so.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func goTimeit(f func(), mintime float64) float64 {
	f()
	runtime.GC()
	t0 := time.Now()
	f()
	one := time.Since(t0).Seconds()
	reps := 1
	if mintime > 0 {
		reps = max(1, min(2000, int(mintime/math.Max(one, 1e-9))))
	}
	t0 = time.Now()
	for i := 0; i < reps; i++ {
		f()
	}
	return time.Since(t0).Seconds() / float64(reps)
}

// perfData mirrors the Python data().
func perfData(n int) (x, y, xq []float64) { return perfDataPlain(n) }

type perfOp struct {
	name   string
	sizes  []int
	rounds int
	// goTimes returns the Go timings for n, keyed like the Python output.
	goTimes func(n int) map[string]float64
}

func goOps() []perfOp {
	all := []int{1000, 10000, 100000, 1000000}
	mid := []int{1000, 10000, 100000}
	small := []int{1000, 10000, 40000}
	sink := 0.0
	return []perfOp{
		{"cubic-nak", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			p, _ := NewCubicSpline(x, y, "not-a-knot")
			return map[string]float64{
				"construct": goTimeit(func() { p, _ = NewCubicSpline(x, y, "not-a-knot") }, 0.05),
				"eval":      goTimeit(func() { p.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"cubic-nak-shuffled", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			r := newRNG(2)
			r.Shuffle(len(xq), func(i, j int) { xq[i], xq[j] = xq[j], xq[i] })
			out := make([]float64, n)
			p, _ := NewCubicSpline(x, y, "not-a-knot")
			return map[string]float64{"eval": goTimeit(func() { p.EvalAll(out, xq) }, 0.05)}
		}},
		{"cubic-periodic", all, 7, func(n int) map[string]float64 {
			x, y, _ := perfData(n)
			y[n-1] = y[0]
			return map[string]float64{"construct": goTimeit(func() { _, _ = NewCubicSpline(x, y, "periodic") }, 0.05)}
		}},
		{"pchip", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			p, _ := NewPchip(x, y)
			return map[string]float64{
				"construct": goTimeit(func() { p, _ = NewPchip(x, y) }, 0.05),
				"eval":      goTimeit(func() { p.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"akima", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			p, _ := NewAkima(x, y, false)
			return map[string]float64{
				"construct": goTimeit(func() { p, _ = NewAkima(x, y, false) }, 0.05),
				"eval":      goTimeit(func() { p.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"makima", all, 7, func(n int) map[string]float64 {
			x, y, _ := perfData(n)
			return map[string]float64{"construct": goTimeit(func() { _, _ = NewAkima(x, y, true) }, 0.05)}
		}},
		{"bspline3", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			b, _ := MakeInterpSpline(x, y, 3, nil)
			return map[string]float64{
				"construct": goTimeit(func() { b, _ = MakeInterpSpline(x, y, 3, nil) }, 0.05),
				"eval":      goTimeit(func() { b.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"bspline5", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			b, _ := MakeInterpSpline(x, y, 5, nil)
			return map[string]float64{
				"construct": goTimeit(func() { b, _ = MakeInterpSpline(x, y, 5, nil) }, 0.05),
				"eval":      goTimeit(func() { b.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"interp1d-linear", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			f, _ := NewInterp1D(x, y, Linear, nil)
			return map[string]float64{
				"construct": goTimeit(func() { f, _ = NewInterp1D(x, y, Linear, nil) }, 0.05),
				"eval":      goTimeit(func() { _ = f.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"interp1d-nearest", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			f, _ := NewInterp1D(x, y, Nearest, nil)
			return map[string]float64{
				"construct": goTimeit(func() { f, _ = NewInterp1D(x, y, Nearest, nil) }, 0.05),
				"eval":      goTimeit(func() { _ = f.EvalAll(out, xq) }, 0.05),
			}
		}},
		{"np.interp", all, 7, func(n int) map[string]float64 {
			x, y, xq := perfData(n)
			out := make([]float64, n)
			return map[string]float64{"eval": goTimeit(func() { _ = Interp(out, xq, x, y, y[0], y[n-1]) }, 0.05)}
		}},
		{"splrep", mid, 5, func(n int) map[string]float64 {
			x, y, _ := perfData(n)
			s := float64(n) * 1e-3
			return map[string]float64{"construct": goTimeit(func() {
				sm, _ := MakeSplrep(x, y, 3, s, nil)
				sink += sm.Fp
			}, 0)}
		}},
		{"lsq", all, 7, func(n int) map[string]float64 {
			x, y, _ := perfData(n)
			ni := max(n/50, 2)
			t := []float64{0, 0, 0, 0}
			for i := 1; i <= ni; i++ {
				t = append(t, 10*float64(i)/float64(ni+1))
			}
			t = append(t, 10, 10, 10, 10)
			return map[string]float64{"construct": goTimeit(func() { _, _ = MakeLsqSpline(x, y, t, 3, nil) }, 0.05)}
		}},
		{"rgi-linear", small, 7, func(n int) map[string]float64 { return goRGI(n, "linear") }},
		{"rgi-nearest", small, 7, func(n int) map[string]float64 { return goRGI(n, "nearest") }},
		{"rgi-cubic", small, 5, func(n int) map[string]float64 { return goRGI(n, "cubic") }},
		{"rbs", small, 5, func(n int) map[string]float64 {
			m := int(math.Round(math.Sqrt(float64(n))))
			xs := sortedX(nil, m, 0, 10, false)
			z := make([]float64, m*m)
			for i := range xs {
				for j := range xs {
					z[i*m+j] = math.Sin(xs[i]) * math.Cos(xs[j])
				}
			}
			r := newRNG(3)
			xi, yi := make([]float64, n), make([]float64, n)
			for i := range xi {
				xi[i], yi[i] = r.Float64()*10, r.Float64()*10
			}
			b, _ := NewBicubic(xs, xs, z, 3, 3)
			return map[string]float64{
				"construct": goTimeit(func() { b, _ = NewBicubic(xs, xs, z, 3, 3) }, 0),
				"eval": goTimeit(func() {
					for i := range xi {
						sink += b.At(xi[i], yi[i])
					}
				}, 0),
			}
		}},
	}
}

func goRGI(n int, method string) map[string]float64 {
	m := int(math.Round(math.Sqrt(float64(n))))
	ax := sortedX(nil, m, 0, 10, false)
	vals := make([]float64, m*m)
	for i := range ax {
		for j := range ax {
			vals[i*m+j] = math.Sin(ax[i]) * math.Cos(ax[j])
		}
	}
	r := newRNG(3)
	xi := make([]float64, 2*n)
	for i := range xi {
		xi[i] = r.Float64() * 10
	}
	out := make([]float64, n)
	f, _ := NewRGI([][]float64{ax, ax}, vals, method, nil)
	return map[string]float64{
		"construct": goTimeit(func() { f, _ = NewRGI([][]float64{ax, ax}, vals, method, nil) }, 0),
		"eval":      goTimeit(func() { _ = f.Eval(out, xi) }, 0),
	}
}

func loadAvg() string {
	b, err := exec.Command("uptime").Output()
	if err != nil {
		return "?"
	}
	s := string(b)
	if i := strings.Index(s, "load average"); i >= 0 {
		f := strings.Fields(strings.TrimPrefix(s[i:], "load averages:"))
		if len(f) > 0 {
			return strings.Trim(f[0], ",")
		}
	}
	return "?"
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

type perfRow struct {
	op, phase string
	n         int
	goT, pyT  float64
	load      float64
}

func TestPerf(t *testing.T) {
	only := os.Getenv("PERF_OPS")
	var rows []perfRow
	for _, op := range goOps() {
		if only != "" && !strings.Contains(","+only+",", ","+op.name+",") {
			continue
		}
		for _, n := range op.sizes {
			goR := map[string][]float64{}
			pyR := map[string][]float64{}
			var loads []float64
			rounds := op.rounds
			if n >= 1000000 {
				rounds = min(rounds, 5)
			}
			for r := 0; r < rounds; r++ {
				l, _ := strconv.ParseFloat(loadAvg(), 64)
				loads = append(loads, l)
				for k, v := range op.goTimes(n) {
					goR[k] = append(goR[k], v)
				}
				for k, v := range pyTime(t, op.name, n) {
					pyR[k] = append(pyR[k], v)
				}
			}
			for _, phase := range []string{"construct", "eval", "legacy"} {
				if len(pyR[phase]) == 0 {
					continue
				}
				g := goR[phase]
				if phase == "legacy" {
					g = goR["construct"]
				}
				row := perfRow{op.name, phase, n, median(g), median(pyR[phase]), median(loads)}
				rows = append(rows, row)
				t.Logf("%-20s %-9s n=%-8d go %.3e s  scipy %.3e s  ratio %.2f  load %.0f", row.op, row.phase, row.n, row.goT, row.pyT, row.pyT/row.goT, row.load)
			}
		}
	}
	var b strings.Builder
	b.WriteString("# Performance against SciPy/NumPy\n\n")
	fmt.Fprintf(&b, "Measured %s on %s/%s, %s, **one core on each side** (Go is single-threaded; the Python child runs with OPENBLAS/OMP/MKL/VECLIB threads = 1).\n", time.Now().Format("2006-01-02"), runtime.GOOS, runtime.GOARCH, runtime.Version())
	fmt.Fprintf(&b, "SciPy %s, NumPy %s. Rounds are interleaved (Go, SciPy, Go, SciPy, ...), each round timing the mean of several repetitions; the table shows the median of 5-7 rounds.\n", oracleScipy, oracleNumpy)
	b.WriteString("The host is shared and was loaded (the load average column is the median of `uptime` before each round), so absolute times are inflated and noisy; ratios of interleaved rounds are the meaningful figure. Ratio = SciPy time / Go time, so a ratio above 1 means Go is faster.\n\n")
	b.WriteString("Data: n jittered nodes on [0, 10], a smooth noisy signal, n sorted uniform queries (the 2-D rows use a sqrt(n) x sqrt(n) grid and n scattered queries; `splrep` uses s = 1e-3 n).\n\n")
	b.WriteString("| operation | phase | n | Go (s) | SciPy (s) | SciPy / Go | load |\n|---|---|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		phase := r.phase
		if phase == "legacy" {
			phase = "construct (legacy splrep)"
		}
		fmt.Fprintf(&b, "| %s | %s | %d | %.3g | %.3g | %.2f | %.0f |\n", r.op, phase, r.n, r.goT, r.pyT, r.pyT/r.goT, r.load)
	}
	if only == "" {
		if err := os.WriteFile("docs/perf.md", []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	} else {
		t.Log("\n" + b.String())
	}
}
