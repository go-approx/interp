//go:build oracle

package interp

import (
	"fmt"
	"math"
	"testing"
)

// ---------------------------------------------------------------- pchip/akima

const shapeBody = `
def run(op, v, n):
    x, y, xq = v['x'], v['y'], v['xq']
    big = n.get('big')
    if op == 'pchip':
        s = I.PchipInterpolator(x, y)
    elif op == 'akima':
        s = I.Akima1DInterpolator(x, y)
    else:
        s = I.Akima1DInterpolator(x, y, method='makima')
    out = {'c': samp(s.c, big), 'y0': samp(s(xq), big), 'y1': samp(s(xq, 1), big), 'y2': samp(s(xq, 2), big)}
    out['y0ne'] = samp(s(xq, extrapolate=False), big)
    return out
`

func shapeCases() []Case {
	r := newRNG(2)
	var cs []Case
	add := func(name, op string, x, y []float64) {
		var xq []float64
		if len(x) > 0 {
			xq = queries(r, 30, x[0], x[len(x)-1], x, true)
		}
		cs = append(cs, mk(name, op, nil, map[string][]float64{"x": x, "y": y, "xq": xq}))
	}
	for _, op := range []string{"pchip", "akima", "makima"} {
		for _, n := range []int{2, 3, 4, 5, 6, 7, 10, 25, 100} {
			for _, jit := range []bool{false, true} {
				x := sortedX(r, n, -2, 3, jit)
				add(nameOf("shape", op, fmt.Sprint(n), fmt.Sprint(jit)), op, x, smoothY(r, x, 0.1))
			}
		}
		x := sortedX(r, 12, 0, 11, false)
		mono := make([]float64, 12)
		for i := range mono {
			mono[i] = math.Tanh(float64(i) - 5.5)
		}
		add("shape/"+op+"/monotone", op, x, mono)
		add("shape/"+op+"/step", op, x, []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1})
		add("shape/"+op+"/constant", op, x, []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2})
		add("shape/"+op+"/linear", op, x, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
		add("shape/"+op+"/zigzag", op, x, []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1})
		add("shape/"+op+"/plateau", op, x, []float64{0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6, 6})
		xc := []float64{0, 1e-12, 2e-12, 1e-6, 1, 1 + 1e-9, 5, 5 + 1e-14, 100}
		add("shape/"+op+"/clustered", op, xc, smoothY(r, xc, 0.1))
		xh := sortedX(r, 15, 1e8, 1e8+1, true)
		add("shape/"+op+"/offset1e8", op, xh, smoothY(r, xh, 0.1))
		xs := sortedX(r, 15, 0, 1e-9, true)
		add("shape/"+op+"/tinyspan", op, xs, smoothY(r, xs, 0.1))
		x5 := sortedX(r, 5, 0, 1, false)
		y5 := smoothY(r, x5, 0)
		add("shape/"+op+"/bad/dup", op, []float64{0, 1, 1, 2, 3}, y5)
		add("shape/"+op+"/bad/unsorted", op, []float64{0, 2, 1, 3, 4}, y5)
		add("shape/"+op+"/bad/n1", op, []float64{1}, []float64{2})
		add("shape/"+op+"/bad/nan", op, x5, []float64{0, 1, math.NaN(), 3, 4})
		add("shape/"+op+"/bad/mismatch", op, x5, y5[:4])
		add("shape/"+op+"/bad/inf-x", op, []float64{0, 1, 2, 3, math.Inf(1)}, y5)
	}
	cs = append(cs, bigCase("shape/big/1e5/pchip", "pchip", 100000, 21, nil))
	cs = append(cs, bigCase("shape/big/1e5/akima", "akima", 100000, 22, nil))
	cs = append(cs, bigCase("shape/big/1e6/makima", "makima", 1000000, 23, nil))
	return cs
}

func TestGenerateShape(t *testing.T) { generate(t, "shape", shapeBody, shapeCases()) }

// ---------------------------------------------------------------- interp1d

const interp1dBody = `
def run(op, v, n):
    x, y, xq = v['x'], v['y'], v['xq']
    big = n.get('big')
    if op == 'npinterp':
        left = n['left'] if 'left' in n else None
        right = n['right'] if 'right' in n else None
        return {'y': samp(np.interp(xq, x, y, left=left, right=right), big)}
    fv = np.nan
    if 'fb' in n:
        fv = (n['fb'], n['fa'])
    elif 'fill' in n:
        fv = n['fill']
    elif n.get('extrap'):
        fv = 'extrapolate'
    f = I.interp1d(x, y, kind=op, bounds_error=bool(n.get('be')), fill_value=fv,
                   assume_sorted=bool(n.get('sorted')))
    return {'y': samp(f(xq), big)}
`

func interp1dCases() []Case {
	r := newRNG(3)
	var cs []Case
	add := func(name, op string, num map[string]float64, x, y, xq []float64) {
		cs = append(cs, mk(name, op, num, map[string][]float64{"x": x, "y": y, "xq": xq}))
	}
	special := func(x []float64) []float64 {
		q := queries(r, 25, x[0], x[len(x)-1], x, true)
		return append(q, math.NaN(), math.Inf(1), math.Inf(-1), x[0], x[len(x)-1], x[len(x)/2])
	}
	opts := []map[string]float64{nil, {"fill": -7}, {"fb": -1, "fa": 9}, {"extrap": 1}, {"be": 1}, {"sorted": 1}}
	for _, op := range []string{"linear", "nearest", "nearest-up", "previous", "next"} {
		for _, n := range []int{2, 3, 5, 20, 200} {
			for oi, o := range opts {
				x := sortedX(r, n, -1, 4, true)
				y := smoothY(r, x, 0.2)
				if oi != 5 && n > 2 && oi%2 == 1 { // an unsorted copy
					perm := r.Perm(n)
					xs, ys := make([]float64, n), make([]float64, n)
					for i, p := range perm {
						xs[i], ys[i] = x[p], y[p]
					}
					x, y = xs, ys
				}
				xq := special(sortedX(r, n, -1, 4, true))
				if o != nil && o["be"] == 1 {
					xq = xq[:20] // some in range, some not: scipy raises
				}
				add(nameOf("interp1d", op, fmt.Sprint(n), fmt.Sprint(oi)), op, o, x, y, xq)
			}
		}
		// exact midpoints (ties) and repeated queries
		x := []float64{0, 1, 2, 3, 4}
		add("interp1d/"+op+"/ties", op, nil, x, []float64{5, 4, 3, 2, 1}, []float64{0.5, 1.5, 2.5, 3.5, 0, 4, 1, 2, 0.5, 0.5})
		add("interp1d/"+op+"/single", op, nil, []float64{2}, []float64{7}, []float64{1, 2, 3})
		add("interp1d/"+op+"/dup-x", op, nil, []float64{0, 1, 1, 2, 3}, []float64{0, 1, 5, 2, 3}, []float64{-1, 0, 0.5, 1, 1.5, 2, 3, 4})
		add("interp1d/"+op+"/nan-y", op, nil, []float64{0, 1, 2, 3}, []float64{0, math.NaN(), 2, 3}, []float64{0.5, 1.5, 2.5, 1})
		add("interp1d/"+op+"/inf-y", op, nil, []float64{0, 1, 2, 3}, []float64{0, math.Inf(1), 2, 3}, []float64{0.5, 1, 1.5, 2.5})
		add("interp1d/"+op+"/huge", op, nil, []float64{-1e300, 0, 1e300}, []float64{1, 2, 3}, []float64{-5e299, 5e299, 0, 1e300})
		add("interp1d/"+op+"/bad/empty", op, nil, nil, nil, []float64{1})
		add("interp1d/"+op+"/bad/mismatch", op, nil, []float64{0, 1, 2}, []float64{1, 2}, []float64{1})
		add("interp1d/"+op+"/noquery", op, nil, []float64{0, 1, 2}, []float64{1, 2, 3}, nil)
		add("interp1d/"+op+"/extrap-single", op, map[string]float64{"extrap": 1}, []float64{2}, []float64{7}, []float64{1, 2, 3})
	}
	// numpy.interp
	for _, n := range []int{1, 2, 3, 10, 100} {
		x := sortedX(r, n, 0, 5, true)
		y := smoothY(r, x, 0.2)
		add(nameOf("npinterp", fmt.Sprint(n)), "npinterp", nil, x, y, special(sortedX(r, max(n, 2), 0, 5, true)))
		add(nameOf("npinterp", fmt.Sprint(n), "lr"), "npinterp", map[string]float64{"left": -3, "right": 8}, x, y, special(sortedX(r, max(n, 2), 0, 5, true)))
	}
	add("npinterp/dup", "npinterp", nil, []float64{0, 1, 1, 1, 2}, []float64{0, 1, 2, 3, 4}, []float64{-1, 0, 0.5, 1, 1.5, 2, 3})
	add("npinterp/inf-y", "npinterp", nil, []float64{0, 1, 2}, []float64{math.Inf(1), math.Inf(1), 0}, []float64{0.5, 1, 1.5})
	add("npinterp/nan-y", "npinterp", nil, []float64{0, 1, 2}, []float64{math.NaN(), 1, 0}, []float64{0.5, 1, 1.5, 0})
	add("npinterp/flat-inf", "npinterp", nil, []float64{0, 1, 2}, []float64{math.Inf(1), math.Inf(1), 0}, []float64{0.25, 0.5})
	cs = append(cs, bigCase("interp1d/big/1e6/linear", "linear", 1000000, 31, nil))
	cs = append(cs, bigCase("interp1d/big/1e6/nearest", "nearest", 1000000, 32, map[string]float64{"sorted": 1}))
	cs = append(cs, bigCase("interp1d/big/1e6/linear-extrap", "linear", 1000000, 33, map[string]float64{"extrap": 1}))
	return cs
}

func TestGenerateInterp1d(t *testing.T) { generate(t, "interp1d", interp1dBody, interp1dCases()) }

// ---------------------------------------------------------------- make_interp_spline

const bsplineBody = `
def run(op, v, n):
    x, y, xq = v['x'], v['y'], v['xq']
    k = int(n['k'])
    big = n.get('big')
    t = v['t'] if 't' in v else None
    bc = None
    if op == 'periodic':
        bc = 'periodic'
    elif op in ('clamped', 'natural'):
        bc = op
    elif op == 'deriv':
        bc = (list(zip([int(o) for o in v['ol']], v['vl'])) or None, list(zip([int(o) for o in v['or']], v['vr'])) or None)
    b = I.make_interp_spline(x, y, k=k, t=t, bc_type=bc)
    out = {'t': samp(b.t, big), 'c': samp(b.c, big), 'y0': samp(b(xq), big), 'y1': samp(b(xq, 1), big),
           'y2': samp(b(xq, 2), big), 'y3': samp(b(xq, 3), big), 'y0ne': samp(b(xq, extrapolate=False), big)}
    if k >= 1:
        out['der'] = samp(b.derivative(1).c, big)
        out['der2'] = samp(b.derivative(2).c, big) if k >= 2 else np.zeros(1)
    out['anti'] = samp(b.antiderivative()(xq), big)
    out['antic'] = samp(b.antiderivative().c, big)
    lo, hi = float(x[0]), float(x[-1]); sp = hi - lo
    ab = [lo + 0.2 * sp, lo + 0.9 * sp, hi + 0.3 * sp, lo - 0.2 * sp, lo - 1 * sp, lo + 3 * sp]
    out['int'] = [b.integrate(ab[0], ab[1]), b.integrate(ab[2], ab[3]), b.integrate(ab[4], ab[5])]
    out['intne'] = [b.integrate(ab[0], ab[1], extrapolate=False), b.integrate(ab[2], ab[3], extrapolate=False),
                    b.integrate(ab[4], ab[5], extrapolate=False)]
    return out
`

func bsplineCases() []Case {
	r := newRNG(4)
	var cs []Case
	add := func(name, op string, k int, x, y []float64, extra map[string][]float64) {
		var xq []float64
		if len(x) > 0 {
			xq = queries(r, 30, x[0], x[len(x)-1], x, true)
		}
		vec := map[string][]float64{"x": x, "y": y, "xq": xq}
		for k, v := range extra {
			vec[k] = v
		}
		cs = append(cs, mk(name, op, map[string]float64{"k": float64(k)}, vec))
	}
	for k := 0; k <= 5; k++ {
		for _, n := range []int{1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 100} {
			for _, jit := range []bool{false, true} {
				x := sortedX(r, n, -2, 3, jit)
				y := smoothY(r, x, 0.1)
				if k >= 1 && n == 1 {
					continue // scipy returns garbage for a single point with k >= 1
				}
				if n == 100 && (!jit || (k != 3 && k != 5)) {
					continue
				}
				if n < k+1 && (n > 2 || jit) {
					continue // too few points: a couple of error cases are enough
				}
				add(nameOf("bspline", "nak", fmt.Sprint(k), fmt.Sprint(n), fmt.Sprint(jit)), "nak", k, x, y, nil)
				if n >= k+1 && k == 3 {
					for _, op := range []string{"clamped", "natural"} {
						add(nameOf("bspline", op, fmt.Sprint(k), fmt.Sprint(n), fmt.Sprint(jit)), op, k, x, y, nil)
					}
				}
				// general derivative conditions: k-1 of them, split left/right
				if k >= 2 && n >= k+1 {
					nl := k / 2
					nr := (k - 1) - nl
					ol, vl, or, vr := []float64{}, []float64{}, []float64{}, []float64{}
					for i := 0; i < nl; i++ {
						ol, vl = append(ol, float64(1+i)), append(vl, 0.3*float64(i+1))
					}
					for i := 0; i < nr; i++ {
						or, vr = append(or, float64(1+i)), append(vr, -0.2*float64(i+1))
					}
					add(nameOf("bspline", "deriv", fmt.Sprint(k), fmt.Sprint(n), fmt.Sprint(jit)), "deriv", k, x, y,
						map[string][]float64{"ol": ol, "vl": vl, "or": or, "vr": vr})
				}
				if k >= 2 && n >= 2 {
					yp := append([]float64(nil), y...)
					yp[n-1] = yp[0]
					add(nameOf("bspline", "periodic", fmt.Sprint(k), fmt.Sprint(n), fmt.Sprint(jit)), "periodic", k, x, yp, nil)
				}
			}
		}
	}
	// custom knots: k=3, n=10, interior knots at chosen data sites
	x := sortedX(r, 12, 0, 5, true)
	y := smoothY(r, x, 0.1)
	t := []float64{x[0], x[0], x[0], x[0], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[11], x[11], x[11], x[11]}
	add("bspline/custom-t", "nak", 3, x, y, map[string][]float64{"t": t})
	t5 := []float64{x[0], x[0], x[0], x[0], x[0], x[0], x[3], x[4], x[5], x[6], x[7], x[8], x[11], x[11], x[11], x[11], x[11], x[11]}
	add("bspline/custom-t5", "nak", 5, x, y, map[string][]float64{"t": t5})
	// mismatched derivative counts and knots: scipy raises
	add("bspline/bad/deriv-count", "deriv", 3, x, y, map[string][]float64{"ol": {1, 2}, "vl": {0, 0}, "or": {1}, "vr": {0}})
	add("bspline/bad/deriv-order", "deriv", 3, x, y, map[string][]float64{"ol": {5}, "vl": {0}, "or": {1}, "vr": {0}})
	add("bspline/bad/k-too-big", "nak", 5, x[:4], y[:4], nil)
	add("bspline/bad/t-short", "nak", 3, x, y, map[string][]float64{"t": t[:12]})
	add("bspline/bad/t-unsorted", "nak", 3, x, y, map[string][]float64{"t": {x[0], x[0], x[0], x[0], x[5], x[4], x[6], x[7], x[8], x[9], x[10], x[10], x[11], x[11], x[11], x[11]}})
	add("bspline/bad/t-range", "nak", 3, x, y, map[string][]float64{"t": {x[3], x[3], x[3], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[10], x[11], x[11], x[11], x[11]}})
	add("bspline/bad/periodic-t", "periodic", 3, x, append(append([]float64(nil), y[:11]...), y[0]), map[string][]float64{"t": t})
	add("bspline/bad/periodic-mismatch", "periodic", 3, x, y, nil)
	add("bspline/bad/dup", "nak", 3, []float64{0, 1, 1, 2, 3, 4}, []float64{1, 2, 3, 4, 5, 6}, nil)
	add("bspline/bad/unsorted", "nak", 3, []float64{0, 2, 1, 3, 4, 5}, []float64{1, 2, 3, 4, 5, 6}, nil)
	add("bspline/bad/nan", "nak", 3, []float64{0, 1, 2, 3, 4, 5}, []float64{1, 2, math.NaN(), 4, 5, 6}, nil)
	add("bspline/bad/neg-k", "nak", -1, []float64{0, 1, 2, 3, 4, 5}, []float64{1, 2, 3, 4, 5, 6}, nil)
	add("bspline/bad/k0-t", "nak", 0, x, y, map[string][]float64{"t": t})
	add("bspline/bad/mismatch", "nak", 3, x, y[:5], nil)
	add("bspline/bad/k0-clamped", "clamped", 0, x, y, nil)
	add("bspline/bad/k1-natural", "natural", 1, x, y, nil)
	xc := []float64{0, 1e-12, 2e-12, 1e-6, 1, 1 + 1e-9, 5, 5 + 1e-14, 100}
	add("bspline/clustered", "nak", 3, xc, smoothY(r, xc, 0.1), nil)
	xh := sortedX(r, 15, 1e8, 1e8+1, true)
	add("bspline/offset1e8", "nak", 3, xh, smoothY(r, xh, 0.1), nil)
	cs = append(cs, bigCase("bspline/big/1e5/k3", "nak", 100000, 41, map[string]float64{"k": 3}))
	cs = append(cs, bigCase("bspline/big/1e5/k5", "nak", 100000, 42, map[string]float64{"k": 5}))
	cs = append(cs, bigCase("bspline/big/1e5/natural", "natural", 100000, 43, map[string]float64{"k": 3}))
	xb, yb, xqb := bigInputs(44, 100000)
	yb[len(yb)-1] = yb[0]
	pc := mk("bspline/big/1e5/periodic", "periodic", map[string]float64{"k": 3, "big": 1, "n": 100000, "seed": 44}, map[string][]float64{"x": xb, "y": yb, "xq": xqb})
	cs = append(cs, pc)
	return cs
}

func TestGenerateBSpline(t *testing.T) { generate(t, "bspline", bsplineBody, bsplineCases()) }
