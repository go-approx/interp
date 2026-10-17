package interp

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"testing"
)

// F is a float64 that survives JSON: non-finite values travel as strings.
type F float64

func (f F) MarshalJSON() ([]byte, error) {
	v := float64(f)
	switch {
	case math.IsNaN(v):
		return []byte(`"NaN"`), nil
	case math.IsInf(v, 1):
		return []byte(`"Infinity"`), nil
	case math.IsInf(v, -1):
		return []byte(`"-Infinity"`), nil
	}
	return []byte(strconv.FormatFloat(v, 'g', -1, 64)), nil
}

func (f *F) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		switch string(b) {
		case `"NaN"`:
			*f = F(math.NaN())
		case `"Infinity"`:
			*f = F(math.Inf(1))
		case `"-Infinity"`:
			*f = F(math.Inf(-1))
		default:
			return fmt.Errorf("bad float %s", b)
		}
		return nil
	}
	v, err := strconv.ParseFloat(string(b), 64)
	*f = F(v)
	return err
}

func toF(v []float64) []F {
	out := make([]F, len(v))
	for i, x := range v {
		out[i] = F(x)
	}
	return out
}

func fromF(v []F) []float64 {
	if v == nil {
		return nil
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// Case is one oracle call: the inputs the Go generator chose and the outputs
// SciPy or NumPy produced (or the exception it raised).
type Case struct {
	Name string             `json:"name"`
	Op   string             `json:"op,omitempty"`
	Num  map[string]float64 `json:"num,omitempty"`
	Vec  map[string][]F     `json:"vec,omitempty"`
	Out  map[string][]F     `json:"out,omitempty"`
	Err  string             `json:"err,omitempty"`
}

func (c *Case) v(name string) []float64 { return fromF(c.Vec[name]) }
func (c *Case) o(name string) []float64 { return fromF(c.Out[name]) }
func (c *Case) n(name string) float64   { return c.Num[name] }

// Fixture is a testdata file.
type Fixture struct {
	Numpy string `json:"numpy"`
	Scipy string `json:"scipy"`
	Cases []Case `json:"cases"`
}

func loadFixture(t testing.TB, name string) *Fixture {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name+".json"))
	if err != nil {
		t.Fatalf("fixture %s: %v (generate with: GOWORK=off go test -tags oracle -run TestGenerate)", name, err)
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("fixture %s: %v", name, err)
	}
	return &f
}

// ulpDiff is the distance in units in the last place between a and b.
func ulpDiff(a, b float64) uint64 {
	if a == b {
		return 0
	}
	ia, ib := orderedBits(a), orderedBits(b)
	if ia > ib {
		return ia - ib
	}
	return ib - ia
}

func orderedBits(f float64) uint64 {
	u := math.Float64bits(f)
	if u>>63 != 0 {
		return ^u
	}
	return u | 1<<63
}

var statsMu sync.Mutex

// ---- deterministic inputs shared by the generator and the tests ----

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15)) }

// sortedX returns n strictly increasing nodes in [lo, hi]: jittered when
// jitter is true, equispaced otherwise.
func sortedX(r *rand.Rand, n int, lo, hi float64, jitter bool) []float64 {
	x := make([]float64, n)
	if n == 1 {
		x[0] = lo
		return x
	}
	for i := range x {
		x[i] = lo + (hi-lo)*float64(i)/float64(n-1)
	}
	if jitter {
		for i := 1; i < n-1; i++ {
			h := (hi - lo) / float64(n-1)
			x[i] += (r.Float64() - 0.5) * 0.8 * h
		}
	}
	return x
}

func smoothY(r *rand.Rand, x []float64, noise float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = math.Sin(1.7*v) + 0.3*math.Cos(5.1*v) + noise*(r.Float64()-0.5)
	}
	return y
}

// queries returns m points spanning slightly beyond [lo, hi] and including
// every node and the exact end points when include is set.
func queries(r *rand.Rand, m int, lo, hi float64, x []float64, include bool) []float64 {
	span := hi - lo
	q := make([]float64, 0, m+len(x)+2)
	for i := 0; i < m; i++ {
		q = append(q, lo-0.15*span+1.3*span*r.Float64())
	}
	if include {
		q = append(q, x...)
		q = append(q, lo, hi)
	}
	return q
}

// bigSample strides a long slice down to about 200 values (what the Python
// side does with [::step]).
func bigSample(v []float64) []float64 {
	step := max(1, len(v)/200)
	var out []float64
	for i := 0; i < len(v); i += step {
		out = append(out, v[i])
	}
	return out
}

func sortedKeys(m map[string][]F) []string {
	k := make([]string, 0, len(m))
	for n := range m {
		k = append(k, n)
	}
	sort.Strings(k)
	return k
}

// bigInputs regenerates the data of a "big" case from its seed: n jittered
// nodes on [0, 10], a noisy smooth signal and 2n queries.
func bigInputs(seed uint64, n int) (x, y, xq []float64) {
	r := newRNG(seed)
	x = sortedX(r, n, 0, 10, true)
	y = smoothY(r, x, 0.05)
	xq = queries(r, 2*n, 0, 10, x, false)
	return
}

// inputs returns the x, y, xq vectors of a case, regenerated from the seed
// when the fixture stores only the seed.
func (c *Case) inputs() (x, y, xq []float64) {
	if c.Num["big"] == 1 {
		x, y, xq = bigInputs(uint64(c.Num["seed"]), int(c.Num["n"]))
		if c.Op == "periodic" {
			y[len(y)-1] = y[0]
		}
		return x, y, xq
	}
	return c.v("x"), c.v("y"), c.v("xq")
}

// stripBig drops the stored inputs of a big case before writing it.
func (c *Case) stripBig() {
	if c.Num["big"] == 1 {
		delete(c.Vec, "x")
		delete(c.Vec, "y")
		delete(c.Vec, "xq")
	}
}
