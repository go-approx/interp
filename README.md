# interp

Interpolation, splines and smoothing in pure Go (`CGO_ENABLED=0`), validated
against SciPy and NumPy.

```go
import "github.com/go-approx/interp"

s, _ := interp.NewCubicSpline(x, y, "not-a-knot")   // scipy CubicSpline, MATLAB spline
v := s.At(2.5); d := s.Derivative(1).At(2.5); area := s.Integrate(0, 4)

b, _ := interp.MakeInterpSpline(x, y, 5, nil)       // make_interp_spline
sm, _ := interp.MakeSplrep(x, y, 3, 0.05, nil)      // FITPACK smoothing spline
```

## What is provided, and the equivalent elsewhere

| here | SciPy / NumPy | MATLAB |
|---|---|---|
| `NewInterp1D` (`Linear`, `Nearest`, `NearestUp`, `Previous`, `Next`; unsorted x, duplicates, fill values, bounds error, extrapolation) | `interpolate.interp1d` | `interp1` (`linear`, `nearest`, `previous`, `next`) |
| `Interp` | `numpy.interp` | `interp1` linear with `left`/`right` |
| `Interp1(x, v, xq, method, extrap)` | the matching classes | `interp1` with `linear nearest next previous pchip cubic spline makima` and its default-extrapolation rules |
| `NewCubicSpline`, `NewCubicSplineBC` (not-a-knot, natural, clamped, periodic, prescribed first or second derivatives), `PPoly` (value, derivative, antiderivative, integral, wrap/NaN/extend) | `CubicSpline`, `PPoly` | `spline`, `csape`, `ppval` |
| `NewCubicHermite` | `CubicHermiteSpline` | |
| `NewPchip` | `PchipInterpolator` | `pchip`, `interp1(..., 'pchip')` |
| `NewAkima(x, y, false)` / `(x, y, true)` | `Akima1DInterpolator` / `method="makima"` | `makima` |
| `MakeInterpSpline` (any degree, not-a-knot / clamped / natural / derivative lists / user knots / periodic), `BSpline` (de Boor evaluation, derivatives, antiderivatives, integrals, extrapolation modes), `NewDesignMatrix` | `make_interp_spline`, `BSpline`, `BSpline.design_matrix` | `spapi`, `fnval`, `fnder`, `fnint` (Curve Fitting) |
| `MakeLsqSpline` | `make_lsq_spline` | `spap2` |
| `MakeSplrep` (FITPACK smoothing: automatic knots, smoothing factor `s`, weights, user knots, `nest`) | `make_splrep`, `splrep`, `UnivariateSpline` | `spaps`, `csaps` (different penalty) |
| `NewRGI`, `Interpn` (`linear`, `nearest`, `slinear`, `cubic`, `quintic`; bounds error / fill / extrapolate) | `RegularGridInterpolator`, `interpn` | `interpn`, `griddedInterpolant` (`linear`, `nearest`, `spline`) |
| `MakeNdInterp`, `NdBSpline` (tensor-product splines, derivatives per axis) | `make_ndbspl`, `NdBSpline` | |
| `NewBicubic` (any degrees, derivatives, clamping outside the grid) | `RectBivariateSpline(..., s=0)` | `interp2(..., 'spline')` |

## Verification

Reference tools: SciPy 1.18.1 and NumPy 2.5.4, run as oracles from Go by the
tag-gated harness `oracle_test.go` (`uv run --with scipy==1.18.1 --with
numpy==2.5.4 python -c ...`, JSON over a pipe, the Python is the minimal call
only). The harness generates the inputs, writes the answers to `testdata/*.json`,
and the ordinary tests read them (no Python needed):

```
GOWORK=off go test -tags oracle -run TestGenerate -timeout 3h   # regenerate
GOWORK=off go test ./...                                         # verify
```

Inputs include the edge cases: 1, 2 and 3 points, duplicate and unsorted x,
NaN and Inf in x and y, queries outside the range and exactly on the nodes,
nodes clustered over 14 orders of magnitude, offsets of 1e8, spans of 1e-9,
values of 1e150 and 1e-290, mismatched lengths, invalid boundary conditions and
knots, and n = 1e5 and 1e6 (stored as seeds; the data are generated with
explicitly rounded arithmetic so that they are bit-identical on every
architecture). Where SciPy raises, the Go constructor returns an error and the
fixture says so.

Measures: ULPs and relative error on the entries that are not next to a zero
crossing ("well-conditioned": at least 2^-10 of the largest value of their
vector), and for every entry the error scaled by the largest magnitude of the
vector. Exact-equality counts are reported where the algorithm is the same.
Run `go test -v -run Fixture` to print the table; the budgets asserted by the
tests are in the `*_fixture_test.go` files.

RESULTS_TABLE

Octave is not installed on the build host, so MATLAB/Octave semantics
(`interp1` defaults, `nearest` ties going up, `makima`) are implemented from the
MathWorks documentation and checked through the SciPy objects that implement the
same maths, not against Octave.

## Differences from SciPy (deliberate)

- Errors are returned, not raised; no function panics on input.
- `NewInterp1D` rejects NaN in x (SciPy sorts it to the end). Degree-1
  `MakeInterpSpline` rejects a single point (SciPy returns garbage).
- `numpy.interp` is reproduced with `math.FMA` because NumPy's C loop is compiled
  with fused multiply-add on arm64; every other kernel rounds each product
  before summing, so results do not depend on the architecture's FMA fusion.
- For degree 0, SciPy's antiderivative is degenerate at and beyond the right
  end and `integrate(..., extrapolate=False)` returns NaN when the clipped
  interval reaches the end of the knots; this package returns the mathematical
  value.
- Tensor-product splines are solved one axis at a time (a banded system per
  axis), which is the exact interpolant; SciPy's `RegularGridInterpolator`
  solves the full sparse system with an iterative solver by default.
- `RectBivariateSpline` smoothing (`s > 0`, FITPACK `regrid`/`surfit`) is not
  provided, nor is the periodic smoothing spline of `make_splrep`.

## Performance

`docs/perf.md` has the table: construction and evaluation at 10^3 to 10^6
points, Go against SciPy on one core, interleaved rounds, with the load average
of the host. It states the load it was taken at; the host is shared.

Measured 2026-10-19 (load 3–5): faster than SciPy on PCHIP, Akima/makima,
cubic evaluation, periodic cubic construction (1.4–4.7x), nearest-neighbour
interp1d, `make_splrep` (2x), least-squares splines (3–1700x) and every
`RegularGridInterpolator` construction (13–107x).

**Open defects — still slower than the reference:**
- `splrep` against SciPy's legacy FITPACK `splrep` (0.33–0.38x). Cause: like
  `make_splrep`, every knot added within a pass re-solves the full problem,
  where FITPACK's `fpknot` updates the per-interval residual sums
  incrementally (one solve per pass). Fix: a `Splrep` porting `fpcurf` /
  `fpknot` (it also places knots like the legacy routine).
- `RectBivariateSpline` evaluation (0.55–0.72x) and construction (0.84–0.88x).
- `np.interp` / linear `interp1d` evaluation (0.86–0.89x), quintic B-spline
  evaluation (0.92–0.97x), not-a-knot cubic construction from 10^5 points
  (0.92x: the `dgtsv` division chain SciPy also runs).

## Gaps

- Scattered-data interpolation (Delaunay, natural neighbour, RBF) is in a later
  work item.
- No periodic smoothing spline (`make_splrep(bc_type="periodic")`), no parametric
  splines (`make_splprep`), no `make_smoothing_spline` (Reinsch, GCV).
- No MATLAB `interp2` `cubic` (Keys) kernel; `RegularGridInterpolator` `pchip`
  method; complex-valued data; N-dimensional `y` (one dependent variable).
- `PPoly.roots` and `BSpline.roots` are not provided.

## Licence

BSD-3-Clause. The smoothing spline follows Dierckx's algorithm; its structure
was checked against SciPy's `_fitpack_repro.py` (BSD-3-Clause, Copyright (c)
2001-2002 Enthought, Inc., 2003-2026 SciPy Developers), which this code ports in
Go. No GPL or LGPL code is read or ported; Octave and MATLAB are only compared
against.
