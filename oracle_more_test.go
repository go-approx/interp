//go:build oracle

package interp

import (
	"fmt"
	"math"
	"testing"
)

// ---------------------------------------------------------------- make_lsq_spline

const lsqBody = `
def run(op, v, n):
    x, y, xq, t = v['x'], v['y'], v['xq'], v['t']
    k = int(n['k'])
    w = v['w'] if 'w' in v else None
    b = I.make_lsq_spline(x, y, t, k=k, w=w)
    big = n.get('big')
    return {'c': samp(b.c, big), 'y0': samp(b(xq), big), 'y1': samp(b(xq, 1), big)}
`

func lsqCases() []Case {
	r := newRNG(5)
	var cs []Case
	for _, k := range []int{1, 2, 3, 4, 5} {
		for _, m := range []int{k + 2, 12, 60, 400} {
			for _, nint := range []int{0, 1, 3, 8} {
				if nint+k+1 > m {
					continue
				}
				x := sortedX(r, m, 0, 5, true)
				y := smoothY(r, x, 0.3)
				t := []float64{}
				for i := 0; i <= k; i++ {
					t = append(t, x[0])
				}
				for i := 1; i <= nint; i++ {
					t = append(t, x[0]+(x[m-1]-x[0])*float64(i)/float64(nint+1))
				}
				for i := 0; i <= k; i++ {
					t = append(t, x[m-1])
				}
				vec := map[string][]float64{"x": x, "y": y, "t": t, "xq": queries(r, 25, x[0], x[m-1], x, true)}
				if m%2 == 0 {
					w := make([]float64, m)
					for i := range w {
						w[i] = 0.2 + 3*r.Float64()
					}
					vec["w"] = w
				}
				cs = append(cs, mk(nameOf("lsq", fmt.Sprint(k), fmt.Sprint(m), fmt.Sprint(nint)), "lsq", map[string]float64{"k": float64(k)}, vec))
			}
		}
	}
	x := sortedX(r, 20, 0, 5, true)
	y := smoothY(r, x, 0.1)
	t3 := []float64{x[0], x[0], x[0], x[0], x[5], x[10], x[19], x[19], x[19], x[19]}
	bad := func(name string, k int, x, y, t []float64) {
		cs = append(cs, mk(name, "lsq", map[string]float64{"k": float64(k)}, map[string][]float64{"x": x, "y": y, "t": t, "xq": {1, 2}}))
	}
	bad("lsq/bad/t-outside", 3, x, y, []float64{x[2], x[2], x[2], x[2], x[5], x[10], x[19], x[19], x[19], x[19]})
	bad("lsq/bad/unsorted-x", 3, append([]float64{x[3]}, x[:19]...), y, t3)
	bad("lsq/bad/t-unsorted", 3, x, y, []float64{x[0], x[0], x[0], x[0], x[10], x[5], x[19], x[19], x[19], x[19]})
	bad("lsq/bad/nan-y", 3, x, append([]float64{math.NaN()}, y[1:]...), t3)
	bad("lsq/bad/too-many-knots", 3, x[:6], y[:6], []float64{x[0], x[0], x[0], x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[5], x[5]})
	bad("lsq/bad/mismatch", 3, x, y[:10], t3)
	bad("lsq/bad/sw", 3, x, y, []float64{x[0], x[0], x[0], x[0], 2.5, 2.5001, 2.5002, x[19], x[19], x[19], x[19]}[:11])
	bc := bigCase("lsq/big/1e5", "lsq", 100000, 51, map[string]float64{"k": 3})
	bc.Vec["t"] = toF(lsqBigKnots())
	cs = append(cs, bc)
	return cs
}

// lsqBigKnots are the knots of the big least-squares case: cubic, 200 interior
// knots on [0, 10].
func lsqBigKnots() []float64 {
	tb := []float64{0, 0, 0, 0}
	for i := 1; i <= 200; i++ {
		tb = append(tb, 10*float64(i)/201)
	}
	return append(tb, 10, 10, 10, 10)
}

func TestGenerateLsq(t *testing.T) { generate(t, "lsq", lsqBody, lsqCases()) }

// ---------------------------------------------------------------- make_splrep / splrep

const splrepBody = `
def run(op, v, n):
    x, y, xq = v['x'], v['y'], v['xq']
    k = int(n['k']); s = n['s']; big = n.get('big')
    w = v['w'] if 'w' in v else None
    out = {}
    try:
        spl = I.make_splrep(x, y, w=w, k=k, s=s)
        out['t'] = samp(spl.t, big); out['c'] = samp(spl.c[:len(spl.t) - k - 1], big); out['y0'] = samp(spl(xq), big); out['ok'] = [1]; out['nt'] = [len(spl.t)]
    except Exception as e:
        out['ok'] = [0]; out['t'] = []; out['c'] = []; out['y0'] = []
    try:
        tck = I.splrep(x, y, w=w, k=k, s=s)
        out['lt'] = samp(tck[0], big); out['lc'] = samp(tck[1][:len(tck[0]) - k - 1], big); out['ly0'] = samp(I.splev(xq, tck), big); out['lok'] = [1]; out['lnt'] = [len(tck[0])]
    except Exception as e:
        out['lok'] = [0]; out['lt'] = []; out['lc'] = []; out['ly0'] = []
    return out
`

func splrepCases() []Case {
	r := newRNG(6)
	var cs []Case
	for _, k := range []int{1, 2, 3, 4, 5} {
		for _, m := range []int{k + 1, k + 4, 20, 60, 150} {
			for fi, f := range []float64{0, 1e-4, 1e-3, 1e-2, 0.1, 1} {
				x := sortedX(r, m, 0, 10, true)
				y := smoothY(r, x, 0.1)
				vec := map[string][]float64{"x": x, "y": y, "xq": queries(r, 40, 0, 10, x, true)}
				if (m+fi)%3 == 0 && f > 0 {
					w := make([]float64, m)
					for i := range w {
						w[i] = 0.5 + 2*r.Float64()
					}
					vec["w"] = w
				}
				cs = append(cs, mk(nameOf("splrep", fmt.Sprint(k), fmt.Sprint(m), fmt.Sprint(fi)), "splrep",
					map[string]float64{"k": float64(k), "s": f * float64(m)}, vec))
			}
		}
	}
	// pure noise, sharp features, constant, exact polynomial
	x := sortedX(r, 100, 0, 1, true)
	noise := make([]float64, 100)
	step := make([]float64, 100)
	constant := make([]float64, 100)
	poly := make([]float64, 100)
	for i, v := range x {
		noise[i] = r.Float64()
		if v > 0.5 {
			step[i] = 1
		}
		constant[i] = 3
		poly[i] = 1 + 2*v - v*v*v
	}
	for name, y := range map[string][]float64{"noise": noise, "step": step, "constant": constant, "poly": poly} {
		for _, s := range []float64{0.5, 5, 50} {
			cs = append(cs, mk(nameOf("splrep", "shape", name, fmt.Sprint(s)), "splrep", map[string]float64{"k": 3, "s": s},
				map[string][]float64{"x": x, "y": y, "xq": queries(r, 40, 0, 1, x, false)}))
		}
	}
	for _, f := range []float64{1e-3, 0.05, 1} {
		cs = append(cs, bigCase(fmt.Sprintf("splrep/big/2e4/%g", f), "splrep", 20000, 61, map[string]float64{"k": 3, "s": f * 20000}))
	}
	cs = append(cs, bigCase("splrep/big/1e5/k5", "splrep", 100000, 62, map[string]float64{"k": 5, "s": 100}))
	cs = append(cs, mk("splrep/bad/neg-s", "splrep", map[string]float64{"k": 3, "s": -1}, map[string][]float64{"x": x, "y": noise, "xq": {0.5}}))
	cs = append(cs, mk("splrep/bad/unsorted", "splrep", map[string]float64{"k": 3, "s": 1}, map[string][]float64{"x": append([]float64{0.9}, x[1:]...), "y": noise, "xq": {0.5}}))
	cs = append(cs, mk("splrep/bad/k0", "splrep", map[string]float64{"k": 0, "s": 1}, map[string][]float64{"x": x, "y": noise, "xq": {0.5}}))
	cs = append(cs, mk("splrep/bad/few", "splrep", map[string]float64{"k": 3, "s": 1}, map[string][]float64{"x": x[:3], "y": noise[:3], "xq": {0.5}}))
	cs = append(cs, mk("splrep/bad/nan", "splrep", map[string]float64{"k": 3, "s": 1}, map[string][]float64{"x": x, "y": append([]float64{math.NaN()}, noise[1:]...), "xq": {0.5}}))
	return cs
}

func TestGenerateSplrep(t *testing.T) { generate(t, "splrep", splrepBody, splrepCases()) }

// ---------------------------------------------------------------- RegularGridInterpolator

const rgiBody = `
import scipy.sparse.linalg as ssl
def run(op, v, n):
    d = int(n['d'])
    pts = [v['ax%d' % i] for i in range(d)]
    shape = [len(p) for p in pts]
    vals = v['vals'].reshape(shape)
    kw = {}
    if op in ('slinear', 'cubic', 'quintic'):
        kw['solver'] = ssl.spsolve
    f = I.RegularGridInterpolator(pts, vals, method=op, bounds_error=False,
                                  fill_value=(None if n.get('extrap') else np.nan), **kw)
    xi = v['xi'].reshape(-1, d)
    out = {'y': f(xi)}
    if op != 'quintic' or True:
        out['yi'] = I.interpn(pts, vals, xi, method=op, bounds_error=False,
                              fill_value=(None if n.get('extrap') else np.nan)) if op in ('linear', 'nearest') else f(xi)
    return out
`

func rgiCases() []Case {
	r := newRNG(7)
	var cs []Case
	fn := func(p []float64) float64 {
		s := 0.0
		for i, v := range p {
			s += math.Sin(1.3*v+float64(i)) + 0.2*v*v
		}
		return s
	}
	for _, method := range []string{"linear", "nearest", "slinear", "cubic", "quintic"} {
		for _, d := range []int{1, 2, 3, 4} {
			for _, extrap := range []float64{0, 1} {
				for _, jit := range []bool{false, true} {
					nPer := 8
					if d >= 3 {
						nPer = 7
					}
					if d == 4 {
						nPer = 6
					}
					shape := make([]int, d)
					axes := make([][]float64, d)
					tot := 1
					for i := range axes {
						shape[i] = nPer + i%2
						axes[i] = sortedX(r, shape[i], -1+float64(i), 2+float64(i), jit)
						tot *= shape[i]
					}
					vals := make([]float64, tot)
					idx := make([]int, d)
					p := make([]float64, d)
					for flat := 0; flat < tot; flat++ {
						rem := flat
						for i := d - 1; i >= 0; i-- {
							idx[i] = rem % shape[i]
							rem /= shape[i]
							p[i] = axes[i][idx[i]]
						}
						vals[flat] = fn(p)
					}
					npts := 30
					xi := make([]float64, 0, npts*d)
					for q := 0; q < npts; q++ {
						for i := 0; i < d; i++ {
							lo, hi := axes[i][0], axes[i][shape[i]-1]
							xi = append(xi, lo-0.1*(hi-lo)+1.2*(hi-lo)*r.Float64())
						}
					}
					// grid nodes, a corner and a NaN coordinate
					for i := 0; i < d; i++ {
						xi = append(xi, axes[i][shape[i]/2])
					}
					for i := 0; i < d; i++ {
						xi = append(xi, axes[i][0])
					}
					for i := 0; i < d; i++ {
						xi = append(xi, axes[i][shape[i]-1])
					}
					for i := 0; i < d; i++ {
						if i == 0 {
							xi = append(xi, math.NaN())
						} else {
							xi = append(xi, axes[i][1])
						}
					}
					vec := map[string][]float64{"vals": vals, "xi": xi}
					for i := range axes {
						vec[fmt.Sprintf("ax%d", i)] = axes[i]
					}
					cs = append(cs, mk(nameOf("rgi", method, fmt.Sprint(d), fmt.Sprint(extrap), fmt.Sprint(jit)), method,
						map[string]float64{"d": float64(d), "extrap": extrap}, vec))
				}
			}
		}
	}
	// 2-point axes (linear/nearest only valid), and too-few-points errors for splines
	for _, method := range []string{"linear", "nearest", "cubic"} {
		cs = append(cs, mk("rgi/two-points/"+method, method, map[string]float64{"d": 2}, map[string][]float64{
			"ax0": {0, 1}, "ax1": {0, 2}, "vals": {1, 2, 3, 5}, "xi": {0.5, 1, 0, 0, 2, 3, -1, 0.5}}))
	}
	cs = append(cs, mk("rgi/bad/unsorted", "linear", map[string]float64{"d": 2}, map[string][]float64{
		"ax0": {0, 2, 1}, "ax1": {0, 2}, "vals": {1, 2, 3, 5, 6, 7}, "xi": {0.5, 1}}))
	cs = append(cs, mk("rgi/bad/shape", "linear", map[string]float64{"d": 2}, map[string][]float64{
		"ax0": {0, 1, 2}, "ax1": {0, 2}, "vals": {1, 2, 3, 5, 6}, "xi": {0.5, 1}}))
	return cs
}

func TestGenerateRGI(t *testing.T) { generate(t, "rgi", rgiBody, rgiCases()) }

// ---------------------------------------------------------------- RectBivariateSpline

const bicubicBody = `
def run(op, v, n):
    x, y = v['x'], v['y']
    z = v['z'].reshape(len(x), len(y))
    kx, ky = int(n['kx']), int(n['ky'])
    f = I.RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=0)
    xi, yi = v['xi'], v['yi']
    out = {'ev': f.ev(xi, yi), 'grid': f(v['xg'], v['yg'])}
    if kx >= 2: out['dx'] = f.ev(xi, yi, dx=1)
    if ky >= 2: out['dy'] = f.ev(xi, yi, dy=1)
    if kx >= 2 and ky >= 2: out['dxy'] = f.ev(xi, yi, dx=1, dy=1)
    if ky >= 3: out['dyy'] = f.ev(xi, yi, dy=2)
    return out
`

func bicubicCases() []Case {
	r := newRNG(8)
	var cs []Case
	for _, kk := range [][2]int{{3, 3}, {1, 1}, {2, 2}, {5, 5}, {3, 5}, {4, 3}} {
		for _, n := range [][2]int{{6, 6}, {7, 9}, {12, 10}, {40, 35}} {
			for _, jit := range []bool{false, true} {
				if n[0] <= kk[0] || n[1] <= kk[1] {
					continue
				}
				x := sortedX(r, n[0], -1, 2, jit)
				y := sortedX(r, n[1], 0, 4, jit)
				z := make([]float64, 0, n[0]*n[1])
				for _, xv := range x {
					for _, yv := range y {
						z = append(z, math.Sin(2*xv)*math.Cos(yv)+0.1*xv*yv)
					}
				}
				var xi, yi []float64
				for q := 0; q < 40; q++ {
					xi = append(xi, -1.3+3.6*r.Float64())
					yi = append(yi, -0.4+4.8*r.Float64())
				}
				xi = append(xi, x[0], x[n[0]-1], x[1])
				yi = append(yi, y[0], y[n[1]-1], y[2])
				xg := sortedX(r, 5, x[0]-0.3, x[n[0]-1]+0.2, true)
				yg := sortedX(r, 6, y[0]-0.2, y[n[1]-1]+0.5, true)
				cs = append(cs, mk(nameOf("bicubic", fmt.Sprint(kk), fmt.Sprint(n), fmt.Sprint(jit)), "rbs",
					map[string]float64{"kx": float64(kk[0]), "ky": float64(kk[1])},
					map[string][]float64{"x": x, "y": y, "z": z, "xi": xi, "yi": yi, "xg": xg, "yg": yg}))
			}
		}
	}
	return cs
}

func TestGenerateBicubic(t *testing.T) { generate(t, "bicubic", bicubicBody, bicubicCases()) }
