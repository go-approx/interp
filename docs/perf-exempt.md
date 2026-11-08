# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named here
with its reason. The method and the table are in [perf.md](perf.md).

- `BSpline.N`: `len(t) - k - 1`; SciPy has no call to compare (`len(spl.c)` is Python's `len`).
- `PPoly.Coef`: one indexing operation (`C[i*K+power]`); SciPy reads the array `c[k-1-power, i]` directly, so there is no call.
- `ClampedDerivs`: returns the constant condition `{{1, 0}}`; SciPy's `bc_type="clamped"` is a string, not a call.
- `NaturalDerivs`: as `ClampedDerivs`, with `{{2, 0}}`.
