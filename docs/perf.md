# Performance against SciPy/NumPy

Measured 2026-10-19 on darwin/arm64, go1.27.2, **one core on each side** (Go is single-threaded; the Python child runs with OPENBLAS/OMP/MKL/VECLIB threads = 1).
SciPy 1.18.1, NumPy 2.5.4. Rounds are interleaved (Go, SciPy, Go, SciPy, ...), each round timing the mean of several repetitions; the table shows the median of 5-7 rounds.
The host is shared and was loaded (the load average column is the median of `uptime` before each round), so absolute times are inflated and noisy; ratios of interleaved rounds are the meaningful figure. Ratio = SciPy time / Go time, so a ratio above 1 means Go is faster.

The rows `cubic-nak`, `bspline3`, `bspline5`, `interp1d-linear`, `np.interp`, `splrep` and `rbs` construct were re-measured on 2026-10-21 (load 5–7) after the second performance pass; `rbs` eval was re-measured the same day under a load of 34–37 (above the threshold of 8 below which ratios are trusted; the ratios agreed with a run at load 10–11). The `splrep` "legacy" rows now time `Splrep` (the `fpcurf`/`fpknot` port) against SciPy's legacy `splrep`; the plain `construct` rows time `MakeSplrep` against `make_splrep`.

Data: n jittered nodes on [0, 10], a smooth noisy signal, n sorted uniform queries (the 2-D rows use a sqrt(n) x sqrt(n) grid and n scattered queries; `splrep` uses s = 1e-3 n).

| operation | phase | n | Go (s) | SciPy (s) | SciPy / Go | load |
|---|---|---:|---:|---:|---:|---:|
| cubic-nak | construct | 1000 | 2.26e-05 | 9.03e-05 | 3.99 | 6 |
| cubic-nak | eval | 1000 | 6.89e-06 | 1.89e-05 | 2.74 | 6 |
| cubic-nak | construct | 10000 | 0.000145 | 0.000276 | 1.90 | 6 |
| cubic-nak | eval | 10000 | 9.86e-05 | 0.000228 | 2.31 | 6 |
| cubic-nak | construct | 100000 | 0.00129 | 0.00193 | 1.50 | 5 |
| cubic-nak | eval | 100000 | 0.00148 | 0.00297 | 2.00 | 5 |
| cubic-nak | construct | 1000000 | 0.0144 | 0.0188 | 1.30 | 5 |
| cubic-nak | eval | 1000000 | 0.0161 | 0.0326 | 2.02 | 5 |
| cubic-nak-shuffled | eval | 1000 | 1.61e-05 | 2.71e-05 | 1.69 | 3 |
| cubic-nak-shuffled | eval | 10000 | 0.000648 | 0.000612 | 0.94 | 3 |
| cubic-nak-shuffled | eval | 100000 | 0.0148 | 0.0155 | 1.04 | 3 |
| cubic-nak-shuffled | eval | 1000000 | 0.207 | 0.215 | 1.04 | 3 |
| cubic-periodic | construct | 1000 | 3e-05 | 0.000141 | 4.70 | 3 |
| cubic-periodic | construct | 10000 | 0.000272 | 0.000486 | 1.79 | 3 |
| cubic-periodic | construct | 100000 | 0.00241 | 0.00348 | 1.44 | 3 |
| cubic-periodic | construct | 1000000 | 0.0218 | 0.0346 | 1.59 | 4 |
| pchip | construct | 1000 | 9.83e-06 | 0.000105 | 10.68 | 4 |
| pchip | eval | 1000 | 7.21e-06 | 1.9e-05 | 2.64 | 4 |
| pchip | construct | 10000 | 9.4e-05 | 0.000273 | 2.90 | 4 |
| pchip | eval | 10000 | 0.000112 | 0.00024 | 2.15 | 4 |
| pchip | construct | 100000 | 0.000848 | 0.00237 | 2.79 | 4 |
| pchip | eval | 100000 | 0.00148 | 0.00297 | 2.01 | 4 |
| pchip | construct | 1000000 | 0.01 | 0.025 | 2.50 | 4 |
| pchip | eval | 1000000 | 0.0161 | 0.0325 | 2.02 | 4 |
| akima | construct | 1000 | 1.29e-05 | 8e-05 | 6.22 | 4 |
| akima | eval | 1000 | 7.33e-06 | 1.9e-05 | 2.59 | 4 |
| akima | construct | 10000 | 0.00012 | 0.000217 | 1.81 | 4 |
| akima | eval | 10000 | 0.000115 | 0.000224 | 1.95 | 4 |
| akima | construct | 100000 | 0.000753 | 0.00134 | 1.77 | 4 |
| akima | eval | 100000 | 0.00152 | 0.00296 | 1.95 | 4 |
| akima | construct | 1000000 | 0.00722 | 0.0128 | 1.77 | 4 |
| akima | eval | 1000000 | 0.0163 | 0.0325 | 1.99 | 4 |
| makima | construct | 1000 | 1.28e-05 | 8.43e-05 | 6.60 | 5 |
| makima | construct | 10000 | 0.000112 | 0.000234 | 2.10 | 5 |
| makima | construct | 100000 | 0.000784 | 0.00148 | 1.88 | 4 |
| makima | construct | 1000000 | 0.00712 | 0.0141 | 1.99 | 4 |
| bspline3 | construct | 1000 | 6.17e-05 | 0.000104 | 1.68 | 6 |
| bspline3 | eval | 1000 | 8.59e-06 | 3.09e-05 | 3.60 | 6 |
| bspline3 | construct | 10000 | 0.000617 | 0.000706 | 1.14 | 6 |
| bspline3 | eval | 10000 | 9.83e-05 | 0.000297 | 3.02 | 6 |
| bspline3 | construct | 100000 | 0.00575 | 0.00678 | 1.18 | 6 |
| bspline3 | eval | 100000 | 0.00158 | 0.00303 | 1.92 | 6 |
| bspline3 | construct | 1000000 | 0.0549 | 0.0652 | 1.19 | 7 |
| bspline3 | eval | 1000000 | 0.0167 | 0.0293 | 1.75 | 7 |
| bspline5 | construct | 1000 | 0.0001 | 0.000135 | 1.35 | 7 |
| bspline5 | eval | 1000 | 2.03e-05 | 4.92e-05 | 2.43 | 7 |
| bspline5 | construct | 10000 | 0.00104 | 0.00103 | 0.99 | 7 |
| bspline5 | eval | 10000 | 0.000224 | 0.000508 | 2.27 | 7 |
| bspline5 | construct | 100000 | 0.00957 | 0.0099 | 1.03 | 7 |
| bspline5 | eval | 100000 | 0.00243 | 0.00474 | 1.95 | 7 |
| bspline5 | construct | 1000000 | 0.0909 | 0.0985 | 1.08 | 6 |
| bspline5 | eval | 1000000 | 0.0244 | 0.0478 | 1.96 | 6 |
| interp1d-linear | construct | 1000 | 2.44e-06 | 1.19e-05 | 4.89 | 6 |
| interp1d-linear | eval | 1000 | 4.58e-06 | 1.14e-05 | 2.50 | 6 |
| interp1d-linear | construct | 10000 | 1.97e-05 | 5.22e-05 | 2.65 | 6 |
| interp1d-linear | eval | 10000 | 5.01e-05 | 0.000112 | 2.24 | 6 |
| interp1d-linear | construct | 100000 | 0.000179 | 0.000421 | 2.35 | 6 |
| interp1d-linear | eval | 100000 | 0.00103 | 0.00159 | 1.54 | 6 |
| interp1d-linear | construct | 1000000 | 0.00178 | 0.00418 | 2.35 | 6 |
| interp1d-linear | eval | 1000000 | 0.0118 | 0.0162 | 1.38 | 6 |
| interp1d-nearest | construct | 1000 | 2.6e-06 | 1.34e-05 | 5.16 | 4 |
| interp1d-nearest | eval | 1000 | 7.11e-06 | 1.8e-05 | 2.54 | 4 |
| interp1d-nearest | construct | 10000 | 2e-05 | 5.69e-05 | 2.85 | 4 |
| interp1d-nearest | eval | 10000 | 0.000106 | 0.000155 | 1.46 | 4 |
| interp1d-nearest | construct | 100000 | 0.000178 | 0.000454 | 2.56 | 4 |
| interp1d-nearest | eval | 100000 | 0.00163 | 0.00292 | 1.78 | 4 |
| interp1d-nearest | construct | 1000000 | 0.00171 | 0.00445 | 2.61 | 4 |
| interp1d-nearest | eval | 1000000 | 0.0189 | 0.0333 | 1.76 | 4 |
| np.interp | eval | 1000 | 5.86e-06 | 8.43e-06 | 1.44 | 6 |
| np.interp | eval | 10000 | 6.09e-05 | 0.000121 | 1.99 | 7 |
| np.interp | eval | 100000 | 0.0012 | 0.00151 | 1.25 | 7 |
| np.interp | eval | 1000000 | 0.0129 | 0.0157 | 1.22 | 7 |
| splrep | construct | 1000 | 0.00307 | 0.0061 | 1.99 | 7 |
| splrep | construct (`Splrep` vs legacy `splrep`) | 1000 | 0.00041 | 0.00115 | 2.80 | 7 |
| splrep | construct | 10000 | 0.0281 | 0.0597 | 2.12 | 7 |
| splrep | construct (`Splrep` vs legacy `splrep`) | 10000 | 0.00351 | 0.01 | 2.86 | 7 |
| splrep | construct | 100000 | 0.277 | 0.545 | 1.97 | 7 |
| splrep | construct (`Splrep` vs legacy `splrep`) | 100000 | 0.0343 | 0.0977 | 2.85 | 7 |
| lsq | construct | 1000 | 0.000101 | 0.000344 | 3.41 | 4 |
| lsq | construct | 10000 | 0.000994 | 0.0185 | 18.58 | 4 |
| lsq | construct | 100000 | 0.00988 | 1.69 | 171.46 | 4 |
| lsq | construct | 1000000 | 0.0982 | 169 | 1720.01 | 4 |
| rgi-linear | construct | 1000 | 3.75e-07 | 9.71e-06 | 25.89 | 4 |
| rgi-linear | eval | 1000 | 7.69e-05 | 7.33e-05 | 0.95 | 4 |
| rgi-linear | construct | 10000 | 5.42e-07 | 9.62e-06 | 17.76 | 4 |
| rgi-linear | eval | 10000 | 0.000925 | 0.000906 | 0.98 | 4 |
| rgi-linear | construct | 40000 | 7.5e-07 | 1.01e-05 | 13.50 | 4 |
| rgi-linear | eval | 40000 | 0.00422 | 0.00445 | 1.06 | 4 |
| rgi-nearest | construct | 1000 | 2.92e-07 | 9.67e-06 | 33.11 | 4 |
| rgi-nearest | eval | 1000 | 5.48e-05 | 8.93e-05 | 1.63 | 4 |
| rgi-nearest | construct | 10000 | 5e-07 | 9.75e-06 | 19.50 | 4 |
| rgi-nearest | eval | 10000 | 0.000941 | 0.000946 | 1.01 | 4 |
| rgi-nearest | construct | 40000 | 7.91e-07 | 1.01e-05 | 12.75 | 10 |
| rgi-nearest | eval | 40000 | 0.00429 | 0.00452 | 1.05 | 10 |
| rgi-cubic | construct | 1000 | 4.11e-05 | 0.00263 | 64.02 | 10 |
| rgi-cubic | eval | 1000 | 0.000121 | 0.000156 | 1.29 | 10 |
| rgi-cubic | construct | 10000 | 0.000387 | 0.0339 | 87.54 | 9 |
| rgi-cubic | eval | 10000 | 0.00145 | 0.00161 | 1.11 | 9 |
| rgi-cubic | construct | 40000 | 0.00154 | 0.164 | 106.80 | 9 |
| rgi-cubic | eval | 40000 | 0.00623 | 0.00794 | 1.28 | 9 |
| rbs | construct | 1000 | 2.12e-05 | 5.36e-05 | 2.53 | 7 |
| rbs | eval | 1000 | 5.65e-05 | 9.13e-05 | 1.62 | 37 |
| rbs | construct | 10000 | 0.000161 | 0.00034 | 2.11 | 7 |
| rbs | eval | 10000 | 0.000591 | 0.0011 | 1.85 | 34 |
| rbs | construct | 40000 | 0.000583 | 0.00128 | 2.20 | 7 |
| rbs | eval | 40000 | 0.0024 | 0.00614 | 2.56 | 34 |
