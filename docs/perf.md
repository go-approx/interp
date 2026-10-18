# Performance against SciPy

**Final table not yet measured.** The comparison (Go against SciPy on one core,
interleaved rounds, median of 5-7 rounds, load average recorded) must run on an
idle host. The shared build host had a one-minute load average between 12 and
200 when this was written, so ratios from it would be noise; the table is left
to be generated rather than published misleadingly.

Run it with

```
GOWORK=off go test -tags oracle -run TestPerf -timeout 6h -v
```

(`PERF_OPS=cubic-nak,pchip` restricts it to some operations). The driver
(`perf_test.go`, build tag `oracle`) times construction and evaluation of the
cubic spline, PCHIP, Akima, B-splines (k = 3, 5), `interp1d`, `numpy.interp`,
`make_splrep`, `make_lsq_spline`, `RegularGridInterpolator` and
`RectBivariateSpline` at 10^3 to 10^6 points.

An early, unoptimised run at load about 20 (cubic spline, not-a-knot; SciPy time
over Go time, above 1 means Go is faster) gave: construction 3.0 / 1.0 / 0.6 / 0.9
and evaluation 1.2 / 0.7 / 0.7 / 0.85 at n = 10^3 / 10^4 / 10^5 / 10^6, and
`interp1d` linear 0.3 to 0.9. Both the interval search and the linear
interpolation were then rewritten (hinted search, unrolled cubic, sort skipped
for sorted input); those figures are therefore superseded, not current.
