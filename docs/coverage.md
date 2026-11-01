# Coverage against SciPy and MATLAB (interpolation)

Inventory of what `go-approx/interp` (main, v0.2.1) implements against
`scipy.interpolate` v1.18.1 and the MATLAB interpolation, spline and Curve
Fitting functions. Polynomials (`numpy.polynomial`, Chebfun-style capabilities)
are in the sibling module `go-approx/poly`, which has its own
`docs/coverage.md`. Status: **done** (same semantics and options that matter),
**partial** (what is missing is named), **missing**. "Dep" is what a missing
item needs beyond this package.

Sources. SciPy: `scipy/interpolate/__init__.py` at tag v1.18.1 (the rows follow
its autosummary list, section by section). MATLAB: the MathWorks function lists
could not be read by the tool used (client-rendered pages), so the MATLAB
column is built from the documented function families (Interpolation, Curve
Fitting Toolbox "splines" functions, `fit`) and must be re-checked against the
MathWorks reference before a slice is started. Go side: README "Gaps", `go doc`,
and the sources on main.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 | 11 | 14 | 31 | 56 |
| MATLAB | 12 | 16 | 27 | 55 |

A row counts under SciPy when its SciPy cell is filled and under MATLAB when its
MATLAB cell is filled.

## Univariate interpolation

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `make_interp_spline` | `spapi`, `griddedInterpolant('spline')` | `MakeInterpSpline` | partial | complex `y`, N-D `y` (`axis`), `check_finite` forms; all boundary conditions and degrees are done |
| 2 | `CubicSpline` | `spline`, `csape` | `NewCubicSpline`, `NewCubicSplineBC` | partial | N-D `y`, `axis`; every `bc_type` is done |
| 3 | `PchipInterpolator` | `pchip`, `interp1 'pchip'`, `'cubic'` | `NewPchip` | partial | N-D `y`, `axis` |
| 4 | `Akima1DInterpolator` (`akima`, `makima`) | `makima`, `interp1 'makima'` | `NewAkima` | partial | N-D `y`, `axis` |
| 5 | `FloaterHormannInterpolator` | | | missing | none (barycentric rational, needs weights only) |
| 6 | `BarycentricInterpolator` | (Chebfun `chebfun`) | | missing | none |
| 7 | `KroghInterpolator` | | | missing | none (Newton divided differences with derivatives) |
| 8 | `CubicHermiteSpline` | `pchip` with slopes, `spline` with end slopes | `NewCubicHermite` | partial | N-D `y` |
| 9 | `PPoly` | `mkpp`, `ppval`, `unmkpp`, `ppmak` | `PPoly`, `NewPPoly` | partial | `roots`, `solve`, `extrema`, `from_spline`, `from_bernstein_basis`, `extend`, N-D coefficients (README: `PPoly.roots` not provided) |
| 10 | `BPoly` (`from_derivatives`, `from_power_basis`) | | | missing | none (Bernstein basis) |
| 11 | `BSpline` | `fnval`, `fnder`, `fnint`, `spmak`, `bspline` | `BSpline`, `NewBSpline` | partial | `roots`, `basis_element`, `from_power_basis`, `insert_knot`, `construct_fast`, N-D coefficients, complex |
| 12 | `interp1d` (legacy) | `interp1` | `NewInterp1D`, `Interp1` | done | `kind` quadratic and cubic go through `MakeInterpSpline`; complex `y` missing |
| 13 | `numpy.interp` | `interp1 linear` | `Interp` | done | complex `fp`, `period` |
| 14 | `pchip_interpolate` | `pchip` | `NewPchip` | done | |
| 15 | `barycentric_interpolate`, `krogh_interpolate` | | | missing | rows 6 and 7 |
| 16 | `lagrange` | `polyfit` + `polyval` | | missing | none (returns a `poly1d`; use `go-approx/poly`) |
| 17 | `approximate_taylor_polynomial` | `taylor` (Symbolic) | | missing | sampling, `go-approx/poly` `Polyfit` |
| 18 | `pade` | `pade`, `padeapprox` | | missing | dense solve; `go-approx/poly` for the polynomials |
| 19 | | `interpft` | | missing | `go-fft/fft` |
| 20 | | `fillmissing`, `fillmissing 'spline'/'pchip'/'makima'` | `Interp1` per gap | partial | gap detection and API (NaN-aware wrapper) |

## Multivariate interpolation: unstructured data

None of these is provided (README: "Scattered-data interpolation ... is in a
later work item").

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 21 | `LinearNDInterpolator` | `griddata 'linear'`, `scatteredInterpolant 'linear'` | | missing | Delaunay triangulation (2-D, then N-D; written from Bowyer-Watson/Guibas-Stolfi, not from Qhull source), point location, barycentric weights |
| 22 | `NearestNDInterpolator` | `griddata 'nearest'`, `scatteredInterpolant 'nearest'` | | missing | k-d tree (`cKDTree` equivalent; a spatial library) |
| 23 | `CloughTocher2DInterpolator` | `griddata 'cubic'` (C1, 2-D), `'v4'` is different | | missing | Delaunay, gradient estimation (global iterative scheme) |
| 24 | `RBFInterpolator` (kernels, `neighbors`, `smoothing`, `degree`, `epsilon`) | `rbfinterp`, `griddata 'v4'` | | missing | dense solve (LU/QR from the dense linear algebra library), k-d tree for `neighbors`, `go-approx/poly` Vandermonde for the polynomial tail |
| 25 | `Rbf` (legacy) | | | missing | row 24 |
| 26 | `griddata` (nearest, linear, cubic) | `griddata`, `griddatan` | | missing | rows 21 to 23 |
| 27 | | `scatteredInterpolant` (`linear`, `nearest`, `natural`; `none`/`linear`/`nearest` extrapolation) | | missing | Delaunay, natural-neighbour (Sibson) weights |
| 28 | | `delaunay`, `delaunayTriangulation`, `triangulation`, `tsearchn`, `dsearchn` | | missing | a computational-geometry package; outside interp, but the dependency of rows 21 to 27 |
| 29 | | `tpaps` (thin-plate smoothing spline) | | missing | RBF machinery (row 24), `tpaps` smoothing parameter |

## Multivariate interpolation: data on a grid

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 30 | `RegularGridInterpolator` (`linear`, `nearest`, `slinear`, `cubic`, `quintic`) | `griddedInterpolant`, `interpn`, `interp2`, `interp3` | `RGI`, `NewRGI`, `Interpn` | partial | `pchip` method (README), `cubic_legacy`, gradients (`nu`), complex and N-D `values`, `bounds_error`/fill are done |
| 31 | `interpn` | `interpn` | `Interpn` | done | |
| 32 | | `interp2 'linear'`, `'nearest'` | `Interpn` on 2-D | done | |
| 33 | | `interp2 'spline'` | `NewBicubic` | done | |
| 34 | | `interp2 'cubic'` (Keys convolution), `'makima'` | | missing | Keys kernel on a uniform or non-uniform grid; makima per axis (row 4 in 2-D) |
| 35 | | `interp3`, `interpn 'cubic'`/`'spline'`/`'makima'` | `RGI` (cubic, quintic), `MakeNdInterp` | partial | Keys cubic, makima per axis |
| 36 | `NdPPoly` | | | missing | none (tensor product piecewise polynomial; `PPoly` generalised) |
| 37 | `NdBSpline` | | `NdBSpline`, `MakeNdInterp` | partial | `__call__` with extrapolate flag, N-D `c` (trailing axes) |
| 38 | `scipy.ndimage.map_coordinates`, `spline_filter` (referenced) | `imwarp`, `interp2` | | missing | belongs to an image package (go-ndimage?); uses B-spline prefilter, which `MakeNdInterp` implements per axis |

## 1-D spline smoothing and approximation

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 39 | `make_lsq_spline` | `spap2`, `fit(..., 'bspline')` | `MakeLsqSpline` | partial | N-D `y`, `axis`; weights and degree done |
| 40 | `make_smoothing_spline` (Reinsch, `lam` or GCV) | `csaps`, `spaps` (tolerance form) | | missing | banded Cholesky; GCV criterion (Craven-Wahba); `spaps` by root-finding on `lam` (go-numerics/optimize `Brent`) |
| 41 | `make_splrep` | `spaps` (FITPACK-like) | `MakeSplrep` | partial | periodic (`bc_type="periodic"`, README), `per` option |
| 42 | `make_splprep` (parametric curves) | `cscvn`, `rscvn`, `spcrv` | | missing | row 41 on each coordinate with a shared parameter; chord-length parametrisation |
| 43 | `generate_knots` | `newknt`, `optknt`, `aptknt`, `augknt`, `aveknt` | internal to `MakeSplrep` | missing | export the knot-placement search (`MakeSplrep` already runs it); MATLAB knot utilities are independent small functions |

## Rational approximation

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 45 | `AAA` (adaptive Antoulas-Anderson) | `chebfun/aaa` (Chebfun), `rationalfit` (RF Toolbox) | | missing | SVD of a Loewner matrix (dense SVD from the dense linear algebra library); `poles`, `residues`, `zeros` need a dense generalised eigensolver |

## FITPACK, functional and OOP interfaces (1-D)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 46 | `splrep` | | `Splrep` | done | identical knots and FITPACK `ier` on 166 fixtures (README) |
| 47 | `splprep` | | | missing | FITPACK `parcur`; or row 42 for the modern form |
| 48 | `splev` | `fnval` | `BSpline.EvalNu`, `At` | done | |
| 49 | `splint` | `fnint` + difference | `BSpline.Integrate` | done | |
| 50 | `sproot` | `fnzeros` | | missing | B-spline root finding (cubic only in FITPACK `sproot`; general degree needs a Bezier-extraction step) |
| 51 | `spalde` | | `EvalNu` per order | partial | one call returning all derivatives at a point |
| 52 | `splder`, `splantider` | `fnder`, `fnint` | `Derivative`, `Antiderivative` | done | |
| 53 | `insert` (knot insertion) | `fnrfn`, `spmak` | | missing | Boehm's algorithm (small) |
| 54 | `UnivariateSpline` | `spaps` | `Smoothing`, `MakeSplrep` | partial | `set_smoothing_factor`, `get_knots`, `get_coeffs`, `get_residual`, `roots`, `derivatives` as methods |
| 55 | `InterpolatedUnivariateSpline` | `spapi` | `Splrep` with `s=0`, `MakeInterpSpline` | done | |
| 56 | `LSQUnivariateSpline` | `spap2` | `MakeLsqSpline`, `Splrep` with user knots | done | |
| 57 | | `fnbrk`, `fncmb`, `fnjmp`, `fnxtr`, `fnmin`, `fnplt`, `fn2fm`, `fnchg`, `fndir`, `fntlr` | | missing | function-object algebra: `fncmb` (combine), `fnjmp`, `fnxtr` (extrapolate), `fnmin` need `roots`; `fnplt` is plotting |

## 2-D FITPACK splines

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 58 | `RectBivariateSpline` | `interp2 'spline'`, `spapi` (2-D) | `Bicubic`, `NewBicubic` | partial | smoothing `s > 0` (FITPACK `regrid`, README) |
| 59 | `RectSphereBivariateSpline` | | | missing | FITPACK `spgrid` port (BSD via SciPy), periodic and pole conditions |
| 60 | `BivariateSpline` (base), `SmoothBivariateSpline` | `tpaps`, `spaps` (2-D) | | missing | FITPACK `surfit` port; needs the Givens-based least-squares solver already in `Splrep` |
| 61 | `SmoothSphereBivariateSpline`, `LSQSphereBivariateSpline` | | | missing | FITPACK `sphere` port |
| 62 | `LSQBivariateSpline` | `spap2` (2-D) | | missing | FITPACK `surfit` with `task=-1` |
| 63 | `bisplrep`, `bisplev` | | | missing | `surfit` and `bispev` ports |
| 64 | `interp2d` (removed in 1.14, raises `NotImplementedError`) | `interp2` | `Interpn` | done | not applicable in SciPy; use rows 30 and 58 |

## Curve Fitting Toolbox, other

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 65 | `scipy.optimize.curve_fit` (outside interpolate) | `fit` with library models (`poly1`..`poly9`, `exp1`, `gauss1`, `power1`, `rat`, `sin`, `fourier`, `weibull`) | | missing | `go-numerics/optimize` least squares; a model catalogue; confidence intervals |
| 66 | `scipy.signal.savgol_filter`, `statsmodels lowess` | `smooth` (moving average, `lowess`, `loess`, `rlowess`, `rloess`, `sgolay`) | | missing | LOWESS/LOESS local regression; Savitzky-Golay lives in `go-dsp/signal` (also missing) |
| 67 | `scipy.interpolate.NearestNDInterpolator` + `KDTree` | `knnsearch`, `rangesearch` | | missing | spatial index package |
| 68 | | `slvblk`, `spcol`, `spmak`, `sprpp`, `sptool` | `DesignMatrix` | partial | `spcol` is `NewDesignMatrix`; `slvblk` is a block-banded solver (banded solver inside `MakeInterpSpline`) |
| 69 | | `csapi`, `csaps`, `cscvn` | `NewCubicSpline` | partial | `csaps` is row 40, `cscvn` row 42 |

## Proposed slices

Ordered by value per cost; "dep" names what has to exist first. Rows in
parentheses.

1. **Object algebra on existing types** (rows 9, 11, 51, 53, 54, 57): `PPoly.Roots/Solve/Extrema`, `BSpline.Roots`, `InsertKnot`, `Spalde`, `FromSpline`, `FromPower`, `Extend`, accessors on `Smoothing`. Dep: none (cubic-segment root solving is closed form; higher degree uses `go-approx/poly` `Roots`).
2. **Barycentric family** (rows 5, 6, 7, 10, 15, 16, 17, 18): `Barycentric`, `Krogh`, `FloaterHormann`, `BPoly`, `Lagrange`, `Pade`, `ApproximateTaylorPolynomial`. Dep: none; `Pade` needs a small dense solve.
3. **Smoothing splines** (rows 40, 41, 42, 43, 47): `MakeSmoothingSpline` (Reinsch with GCV), periodic `MakeSplrep`, `MakeSplprep`, `GenerateKnots`, `Splprep`. Dep: banded Cholesky (own); `spaps` needs `go-numerics/optimize`.
4. **Grid interpolation completeness** (rows 30, 34, 35, 36, 37): RGI `pchip` and `cubic_legacy`, gradients, Keys cubic and makima for `interp2`/`interpn`, `NdPPoly`. Dep: none.
5. **Spatial index and triangulation** (rows 22, 28): k-d tree and Delaunay in a computational-geometry module. Dep: none, but a new repository (written from the papers, not from Qhull).
6. **Scattered data** (rows 21, 23, 24, 25, 26, 27, 29): `LinearND`, `NearestND`, `CloughTocher2D`, `RBFInterpolator`, `Griddata`, natural neighbour, `tpaps`. Dep: slice 5; `RBFInterpolator` also needs a dense LU/QR.
7. **2-D FITPACK** (rows 58 to 63): smoothing `RectBivariateSpline`, `SmoothBivariateSpline`, `LSQBivariateSpline`, spherical variants, `bisplrep`. Dep: ports of `regrid`, `surfit`, `spgrid`, `sphere` (SciPy BSD-3, as `Splrep` already does).
8. **Rational approximation** (row 45): `AAA`. Dep: dense SVD and generalised eigensolver.
9. **Curve fitting front end** (rows 65, 66, 69): `fit`-style model catalogue, LOWESS/LOESS. Dep: `go-numerics/optimize`; Savitzky-Golay from `go-dsp/signal`.
10. **Types** (rows 1 to 4, 8, 11, 30, 37): N-D `y` with `axis`, complex values. Dep: a design decision on `[]float64` versus `go-ndarray`.
