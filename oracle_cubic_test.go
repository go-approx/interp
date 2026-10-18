//go:build oracle

package interp

import (
	"fmt"
	"math"
	"testing"
)

const cubicBody = `
def run(op, v, n):
    x, y, xq = v['x'], v['y'], v['xq']
    big = n.get('big')
    if op == 'd1d2':
        bc = ((int(n['o0']), n['b0']), (int(n['o1']), n['b1']))
    else:
        bc = op
    cs = I.CubicSpline(x, y, bc_type=bc)
    out = {'c': samp(cs.c, big), 'y0': samp(cs(xq), big), 'y1': samp(cs(xq, 1), big),
           'y2': samp(cs(xq, 2), big), 'y3': samp(cs(xq, 3), big),
           'anti': samp(cs.antiderivative()(xq), big)}
    if 'ab' in v:
        out['int'] = [cs.integrate(v['ab'][0], v['ab'][1]), cs.integrate(v['ab2'][0], v['ab2'][1])]
    return out
`

func cubicCases() []Case {
	r := newRNG(1)
	var cs []Case
	add := func(name, op string, num map[string]float64, x, y []float64) {
		var xq []float64
		if len(x) > 0 {
			xq = queries(r, 30, x[0], x[len(x)-1], x, true)
		}
		lo, hi := 0.0, 1.0
		if len(x) > 1 {
			lo, hi = x[0], x[len(x)-1]
		}
		span := hi - lo
		c := mk(name, op, num, map[string][]float64{"x": x, "y": y, "xq": xq,
			"ab":  {lo + 0.2*span, lo + 0.9*span},
			"ab2": {hi + 0.3*span, lo - 0.2*span}})
		cs = append(cs, c)
	}
	for _, bc := range []string{"not-a-knot", "natural", "clamped", "periodic"} {
		for _, n := range []int{2, 3, 4, 5, 6, 7, 10, 25, 100} {
			for _, jit := range []bool{false, true} {
				x := sortedX(r, n, -2, 3, jit)
				y := smoothY(r, x, 0.1)
				if bc == "periodic" {
					y[n-1] = y[0]
				}
				add(nameOf("cubic", bc, fmt.Sprint(n), fmt.Sprint(jit)), bc, nil, x, y)
			}
		}
	}
	for _, o := range [][2]float64{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		for _, n := range []int{2, 3, 5, 40} {
			x := sortedX(r, n, 0, 4, true)
			y := smoothY(r, x, 0.1)
			add(nameOf("cubic", "d1d2", fmt.Sprint(o), fmt.Sprint(n)), "d1d2",
				map[string]float64{"o0": o[0], "o1": o[1], "b0": 0.7, "b1": -1.3}, x, y)
		}
	}
	// scales and extreme magnitudes
	x := sortedX(r, 20, 1e8, 1e8+1, true)
	add("cubic/offset1e8", "not-a-knot", nil, x, smoothY(r, x, 0.1))
	x = sortedX(r, 20, 0, 1e-9, true)
	add("cubic/tinyspan", "natural", nil, x, smoothY(r, x, 0.1))
	x = sortedX(r, 20, -1e6, 1e6, true)
	y := smoothY(r, x, 0.1)
	for i := range y {
		y[i] *= 1e150
	}
	add("cubic/hugey", "not-a-knot", nil, x, y)
	for i := range y {
		y[i] *= 1e-290
	}
	add("cubic/tinyy", "clamped", nil, x, y)
	// clustered nodes: very unequal spacings
	xc := []float64{0, 1e-12, 2e-12, 1e-6, 1, 1 + 1e-9, 5, 5 + 1e-14, 100}
	add("cubic/clustered", "not-a-knot", nil, xc, smoothY(r, xc, 0.1))
	add("cubic/clustered-nat", "natural", nil, xc, smoothY(r, xc, 0.1))
	// invalid input: scipy raises, so must we
	x5 := sortedX(r, 5, 0, 1, false)
	y5 := smoothY(r, x5, 0)
	add("cubic/bad/dup", "not-a-knot", nil, []float64{0, 1, 1, 2, 3}, y5)
	add("cubic/bad/unsorted", "not-a-knot", nil, []float64{0, 2, 1, 3, 4}, y5)
	add("cubic/bad/descending", "natural", nil, []float64{4, 3, 2, 1, 0}, y5)
	add("cubic/bad/n1", "not-a-knot", nil, []float64{1}, []float64{2})
	add("cubic/bad/empty", "not-a-knot", nil, nil, nil)
	add("cubic/bad/mismatch", "not-a-knot", nil, x5, y5[:4])
	add("cubic/bad/nan-y", "not-a-knot", nil, x5, []float64{0, 1, math.NaN(), 3, 4})
	add("cubic/bad/inf-y", "natural", nil, x5, []float64{0, 1, math.Inf(1), 3, 4})
	add("cubic/bad/nan-x", "natural", nil, []float64{0, 1, math.NaN(), 3, 4}, y5)
	add("cubic/bad/inf-x", "natural", nil, []float64{0, 1, 2, 3, math.Inf(1)}, y5)
	add("cubic/bad/periodic-mismatch", "periodic", nil, x5, y5)
	add("cubic/bad/bc", "wobbly", nil, x5, y5)
	// huge n, sampled
	cs = append(cs, bigCase("cubic/big/1e5/not-a-knot", "not-a-knot", 100000, 11, nil))
	cs = append(cs, bigCase("cubic/big/1e5/natural", "natural", 100000, 12, nil))
	cs = append(cs, bigCase("cubic/big/1e6/not-a-knot", "not-a-knot", 1000000, 13, nil))
	return cs
}

func TestGenerateCubic(t *testing.T) { generate(t, "cubic", cubicBody, cubicCases()) }
