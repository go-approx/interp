# Performance against SciPy 1.18.1

The comparison runs under [refbench](https://github.com/go-scikit/refbench)
v0.1.2 (`rbtest.Perf`): one core (GOMAXPROCS=1; BLAS threads pinned to 1),
interleaved rounds (Go, SciPy, Go, SciPy, ...), the 1-minute load read before
every round and the measurement refused at a load of 8 or more. Target: no
slower than SciPy (ratio <= 1); a row above 1 is an open defect.

- The cases are in [`bench/perf_test.go`](../bench/perf_test.go): one or more
  per public function, against the `scipy.interpolate` (or `numpy.interp`)
  function it ports (SciPy 1.18.1 and NumPy 2.5.4, pinned with their hashes in
  `bench/refs.txt`).
- Functions with no SciPy call to compare are named in
  [perf-exempt.md](perf-exempt.md) with the reason.
- The table, with the allocation count of every case, is
  [perf-table.md](perf-table.md), generated from [perf.json](perf.json). Its
  timings are **deferred**: the host was at a 1-minute load of 10 to 24 when this
  was written, so only the allocation counts are recorded (exact under any load).
- The previous table (own harness, sizes 10^3 to 10^6, some rows at a load
  above 8) is kept as [perf-legacy.md](perf-legacy.md).

```
cd bench
go test -run TestPerf .                          CI mode: allocations, table, inventory
go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
go test -run TestDataParity -perf.parity .       same bytes on both sides
go test -run TestWorkParity -perf.parity .       same work (same number of knots in the smoothing fits)
go test -run TestRefSmoke -perf.smoke .          every SciPy expression runs once
```

CI runs the first line: the allocation count of every case must equal
`perf.json` (an increase is a regression, a decrease must be recorded), the table
must be what `perf.json` generates, and every public function must have a case or
an exemption.

## What makes a row comparable

- **Same data.** The nodes, signal, queries, knots, coefficients and grids are
  built by one formula on both sides, with every product rounded before it is
  added (explicit `float64` conversions, so arm64 does not fuse them). The test
  `TestDataParity` hashes the arrays (SHA-256) on both sides for every data
  family, and it passes: 1-D data (10 000 nodes), the LSQ knots, the PPoly
  coefficients (in SciPy's layout; the Go side uses its piece-major layout built
  from the same numbers), the B-spline knots and coefficients, the design-matrix
  inputs (1 000 points) and the 100 x 80 grids with 10 000 scattered queries.
- **Same work.** A spline fit is the same work when it places the same knots:
  `TestWorkParity` compares the number of knots of `Splrep` / `splrep` and of
  `MakeSplrep` / `make_splrep` at the same smoothing factor, and it passes. An
  interpolating fit (`MakeInterpSpline`, the cubic families) has no choice to
  make. `RGI` with `cubic` and `MakeNdInterp` are the one place where the work
  differs by design: SciPy solves the tensor-product system iteratively (its
  default solver, `gcrotmk`), this package solves one banded system per axis;
  the interpolants are the same, the algorithms are not, and the row compares
  what a user of each gets.
- **Callbacks.** There is none in this package: every interpolant takes arrays,
  so no Python function is called inside a SciPy row. A method row builds its
  object outside the timed loop on both sides; the scalar rows (`At`, `Deriv`)
  time one call on the scalar path of each (SciPy's pays the Python call
  overhead of its `__call__`, which is what a caller of it pays).
- `NdBSpline` and `make_ndbspl` are reached through `scipy.interpolate._ndbspline`
  (SciPy exports `NdBSpline` but not `make_ndbspl` at the package level).

## Allocation counts

The counts of the table are per call. They do not depend on the number of
points (none of the evaluation methods allocates; the constructors allocate
their output and a constant number of work arrays). 32-bit lanes are not part of
the `perf` job; a count that differed there would be recorded per `GOARCH`.
