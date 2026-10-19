# Performance against SciPy/NumPy

Measured 2026-10-19 on darwin/arm64, go1.27.2, **one core on each side** (Go is single-threaded; the Python child runs with OPENBLAS/OMP/MKL/VECLIB threads = 1).
SciPy 1.18.1, NumPy 2.5.4. Rounds are interleaved (Go, SciPy, Go, SciPy, ...), each round timing the mean of several repetitions; the table shows the median of 5-7 rounds.
The host is shared and was loaded (the load average column is the median of `uptime` before each round), so absolute times are inflated and noisy; ratios of interleaved rounds are the meaningful figure. Ratio = SciPy time / Go time, so a ratio above 1 means Go is faster.

Data: n jittered nodes on [0, 10], a smooth noisy signal, n sorted uniform queries (the 2-D rows use a sqrt(n) x sqrt(n) grid and n scattered queries; `splrep` uses s = 1e-3 n).

| operation | phase | n | Go (s) | SciPy (s) | SciPy / Go | load |
|---|---|---:|---:|---:|---:|---:|
| cubic-nak | construct | 1000 | 2.54e-05 | 8.86e-05 | 3.48 | 3 |
| cubic-nak | eval | 1000 | 7e-06 | 1.9e-05 | 2.71 | 3 |
| cubic-nak | construct | 10000 | 0.000231 | 0.000273 | 1.18 | 4 |
| cubic-nak | eval | 10000 | 0.000103 | 0.000224 | 2.18 | 4 |
| cubic-nak | construct | 100000 | 0.0021 | 0.00193 | 0.92 | 4 |
| cubic-nak | eval | 100000 | 0.00148 | 0.00295 | 2.00 | 4 |
| cubic-nak | construct | 1000000 | 0.0202 | 0.0186 | 0.92 | 3 |
| cubic-nak | eval | 1000000 | 0.0161 | 0.0327 | 2.03 | 3 |
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
| bspline3 | construct | 1000 | 6.33e-05 | 0.000103 | 1.63 | 4 |
| bspline3 | eval | 1000 | 2.55e-05 | 3.1e-05 | 1.21 | 4 |
| bspline3 | construct | 10000 | 0.000623 | 0.000717 | 1.15 | 4 |
| bspline3 | eval | 10000 | 0.000274 | 0.000303 | 1.11 | 4 |
| bspline3 | construct | 100000 | 0.00562 | 0.0067 | 1.19 | 4 |
| bspline3 | eval | 100000 | 0.00276 | 0.00294 | 1.07 | 4 |
| bspline3 | construct | 1000000 | 0.055 | 0.0645 | 1.17 | 4 |
| bspline3 | eval | 1000000 | 0.0281 | 0.029 | 1.03 | 4 |
| bspline5 | construct | 1000 | 9.89e-05 | 0.000133 | 1.35 | 4 |
| bspline5 | eval | 1000 | 4.98e-05 | 4.82e-05 | 0.97 | 4 |
| bspline5 | construct | 10000 | 0.00102 | 0.00101 | 0.99 | 4 |
| bspline5 | eval | 10000 | 0.000501 | 0.00048 | 0.96 | 4 |
| bspline5 | construct | 100000 | 0.00943 | 0.00987 | 1.05 | 4 |
| bspline5 | eval | 100000 | 0.005 | 0.00469 | 0.94 | 4 |
| bspline5 | construct | 1000000 | 0.0894 | 0.0973 | 1.09 | 4 |
| bspline5 | eval | 1000000 | 0.0506 | 0.0463 | 0.92 | 4 |
| interp1d-linear | construct | 1000 | 2.42e-06 | 1.18e-05 | 4.88 | 4 |
| interp1d-linear | eval | 1000 | 9.1e-06 | 1.11e-05 | 1.22 | 4 |
| interp1d-linear | construct | 10000 | 1.98e-05 | 5.21e-05 | 2.64 | 4 |
| interp1d-linear | eval | 10000 | 0.000134 | 0.000117 | 0.87 | 4 |
| interp1d-linear | construct | 100000 | 0.000173 | 0.000418 | 2.42 | 4 |
| interp1d-linear | eval | 100000 | 0.00177 | 0.00154 | 0.87 | 4 |
| interp1d-linear | construct | 1000000 | 0.0016 | 0.00411 | 2.56 | 4 |
| interp1d-linear | eval | 1000000 | 0.0186 | 0.0159 | 0.86 | 4 |
| interp1d-nearest | construct | 1000 | 2.6e-06 | 1.34e-05 | 5.16 | 4 |
| interp1d-nearest | eval | 1000 | 7.11e-06 | 1.8e-05 | 2.54 | 4 |
| interp1d-nearest | construct | 10000 | 2e-05 | 5.69e-05 | 2.85 | 4 |
| interp1d-nearest | eval | 10000 | 0.000106 | 0.000155 | 1.46 | 4 |
| interp1d-nearest | construct | 100000 | 0.000178 | 0.000454 | 2.56 | 4 |
| interp1d-nearest | eval | 100000 | 0.00163 | 0.00292 | 1.78 | 4 |
| interp1d-nearest | construct | 1000000 | 0.00171 | 0.00445 | 2.61 | 4 |
| interp1d-nearest | eval | 1000000 | 0.0189 | 0.0333 | 1.76 | 4 |
| np.interp | eval | 1000 | 9.18e-06 | 8.14e-06 | 0.89 | 4 |
| np.interp | eval | 10000 | 0.000121 | 0.000129 | 1.07 | 4 |
| np.interp | eval | 100000 | 0.00169 | 0.0015 | 0.89 | 4 |
| np.interp | eval | 1000000 | 0.018 | 0.0156 | 0.86 | 4 |
| splrep | construct | 1000 | 0.00306 | 0.00608 | 1.99 | 4 |
| splrep | construct (legacy splrep) | 1000 | 0.00306 | 0.00115 | 0.38 | 4 |
| splrep | construct | 10000 | 0.028 | 0.0594 | 2.12 | 4 |
| splrep | construct (legacy splrep) | 10000 | 0.028 | 0.00985 | 0.35 | 4 |
| splrep | construct | 100000 | 0.275 | 0.541 | 1.97 | 4 |
| splrep | construct (legacy splrep) | 100000 | 0.275 | 0.0908 | 0.33 | 4 |
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
| rbs | construct | 1000 | 4.35e-05 | 5.38e-05 | 1.23 | 9 |
| rbs | eval | 1000 | 0.000165 | 9.11e-05 | 0.55 | 9 |
| rbs | construct | 10000 | 0.000389 | 0.000342 | 0.88 | 9 |
| rbs | eval | 10000 | 0.00191 | 0.00109 | 0.57 | 9 |
| rbs | construct | 40000 | 0.00154 | 0.00129 | 0.84 | 9 |
| rbs | eval | 40000 | 0.00844 | 0.00609 | 0.72 | 9 |
