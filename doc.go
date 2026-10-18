// Package interp is interpolation and spline smoothing in pure Go: the part of
// SciPy's scipy.interpolate and of MATLAB's interp1, spline, pchip, makima and
// interpn that works on one-dimensional data and on rectilinear grids.
//
// # What is here
//
//   - Piecewise linear, nearest, previous and next interpolation with
//     interp1d / interp1 extrapolation rules: [NewInterp1D], [Interp1], and
//     numpy.interp as [Interp].
//   - Piecewise polynomials ([PPoly]) with exact derivatives, antiderivatives
//     and integrals: cubic splines with not-a-knot, natural, clamped,
//     periodic or prescribed end conditions ([NewCubicSpline]), the monotone
//     PCHIP ([NewPchip]), Akima and modified Akima ([NewAkima]) and the cubic
//     Hermite interpolant ([NewCubicHermite]).
//   - B-splines of any degree ([BSpline]): de Boor evaluation, derivatives,
//     antiderivatives and integrals, interpolation with end conditions or
//     periodic ([MakeInterpSpline]), weighted least squares on given knots
//     ([MakeLsqSpline]) and the design matrix ([NewDesignMatrix]).
//   - Smoothing splines with automatic knot placement, the problem of
//     FITPACK's curfit / splrep / UnivariateSpline ([MakeSplrep]).
//   - Rectilinear grids in any dimension: [RGI] (scipy RegularGridInterpolator) methods
//     linear, nearest, slinear, cubic and quintic ([NewRGI], [Interpn]),
//     tensor-product splines ([MakeNdInterp]) and bicubic interpolation on a
//     grid ([NewBicubic], scipy RectBivariateSpline with s = 0).
//
// Polynomials (polyfit, roots, Chebyshev series) live in the sibling module
// github.com/go-approx/poly.
//
// # Conventions
//
// Everything is float64 and one-dimensional slices; grid values are row-major
// (last axis fastest). Constructors validate their input and return one of
// the Err values of this package, compared with errors.Is; they mirror the
// conditions under which SciPy raises. [PPoly] stores coefficients per piece
// in ascending powers of (x - X[i]), the reverse of SciPy's c array.
//
// # Verification
//
// SciPy 1.18.1 and NumPy 2.5.4 are run as oracles by a Go harness (see
// oracle_test.go, build tag "oracle") on generated inputs including edge
// cases; the fixtures are in testdata/ and the ordinary tests compare against
// them. See the README for the measured agreement.
package interp
