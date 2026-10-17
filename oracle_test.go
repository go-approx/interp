//go:build oracle

package interp

// This file is the oracle harness: it drives SciPy and NumPy (the reference
// tools, run as oracles, never ported from) with inputs generated here and
// writes the answers to testdata/ as JSON. It is built only with
//
//	GOWORK=off go test -tags oracle -run TestGenerate -timeout 3h
//
// The ordinary tests read the fixtures and need no Python.

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const (
	oracleNumpy = "2.5.4"
	oracleScipy = "1.18.1"
)

const oraclePrelude = `
import sys, json, math, warnings
import numpy as np
import scipy
from scipy import interpolate as I
warnings.simplefilter('ignore')
def dec(v): return [float(a) for a in v]
def enc(a):
    return [a if math.isfinite(a) else ('NaN' if a != a else ('Infinity' if a > 0 else '-Infinity')) for a in np.asarray(a, dtype=float).ravel().tolist()]
def samp(a, big):
    a = np.asarray(a, dtype=float).ravel()
    return a[::max(1, a.size // 200)] if big else a
`

const oracleEpilogue = `
res = []
for c in json.load(sys.stdin):
    v = {k: np.array(dec(x)) for k, x in (c.get('vec') or {}).items()}
    n = c.get('num') or {}
    try:
        out = run(c['op'], v, n)
        res.append({'out': {k: enc(a) for k, a in out.items()}})
    except Exception as e:
        res.append({'err': type(e).__name__ + ': ' + str(e)[:200]})
print(json.dumps({'numpy': np.__version__, 'scipy': scipy.__version__, 'res': res}))
`

// callOracle runs the Python body (which defines run(op, v, n) -> dict of
// arrays) over all cases in one process and fills in Out/Err.
func callOracle(t *testing.T, body string, cases []Case) *Fixture {
	t.Helper()
	in, err := json.Marshal(cases)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("uv", "run", "--quiet", "--with", "numpy=="+oracleNumpy, "--with", "scipy=="+oracleScipy,
		"python", "-c", oraclePrelude+body+oracleEpilogue)
	// A minimal environment: no inherited variables, so nothing secret can
	// reach the child. One thread: the oracle is also timed on one core.
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME"),
		"OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"}
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("oracle: %v\n%s", err, stderr.String())
	}
	var r struct {
		Numpy string `json:"numpy"`
		Scipy string `json:"scipy"`
		Res   []Case `json:"res"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("oracle output: %v", err)
	}
	if r.Numpy != oracleNumpy || r.Scipy != oracleScipy || len(r.Res) != len(cases) {
		t.Fatalf("oracle versions/shape: numpy %s scipy %s, %d results for %d cases", r.Numpy, r.Scipy, len(r.Res), len(cases))
	}
	fx := &Fixture{Numpy: r.Numpy, Scipy: r.Scipy}
	for i, c := range cases {
		c.Out, c.Err = r.Res[i].Out, r.Res[i].Err
		c.stripBig()
		fx.Cases = append(fx.Cases, c)
	}
	return fx
}

func writeFixture(t *testing.T, name string, fx *Fixture) {
	t.Helper()
	b, err := json.Marshal(fx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join("testdata", name+".json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	errs := 0
	for _, c := range fx.Cases {
		if c.Err != "" {
			errs++
		}
	}
	t.Logf("%s: %d cases (%d expect an exception), %d bytes", name, len(fx.Cases), errs, len(b))
}

// generate runs one family if the -run pattern selects it.
func generate(t *testing.T, name, body string, cases []Case) {
	t.Helper()
	writeFixture(t, name, callOracle(t, body, cases))
}

// ---- case builders ----

func mk(name, op string, num map[string]float64, vec map[string][]float64) Case {
	c := Case{Name: name, Op: op, Num: num, Vec: map[string][]F{}}
	for k, v := range vec {
		c.Vec[k] = toF(v)
	}
	return c
}

func bigCase(name, op string, n int, seed uint64, extra map[string]float64) Case {
	x, y, xq := bigInputs(seed, n)
	num := map[string]float64{"big": 1, "n": float64(n), "seed": float64(seed)}
	for k, v := range extra {
		num[k] = v
	}
	return mk(name, op, num, map[string][]float64{"x": x, "y": y, "xq": xq})
}

func nameOf(parts ...string) string { return strings.Join(parts, "/") }
