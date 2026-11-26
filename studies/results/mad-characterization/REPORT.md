# MAD approximation and order characterization

240 cases: 10 seeds, 500 rows, 4 features, 4 distributions, 6 orders per generated data set.

These are descriptive finite-grid summaries, with permutations nested within the same generated data set. They are not independent replications and no confidence interval or population-accuracy claim is made.

The comparator uses exact final-sample median/MAD bounds. Differences from it characterize a different trimming procedure; they do not measure estimation error against a known population target. Sorted/reverse-sorted rows intentionally violate iid ordering to stress order dependence.

| Distribution | Order group | Max bound difference in exact-scale units, median / max | Coordinate disagreement, median / max | Max pair-correlation difference, median / max |
|---|---|---:|---:|---:|
| gaussian | generated/shuffled | 0.1140 / 0.2057 | 0.0005 / 0.0020 | 0.0053 / 0.0205 |
| gaussian | sorted | 0.5890 / 1.6827 | 0.0010 / 0.0035 | 0.0107 / 0.0200 |
| lognormal | generated/shuffled | 0.2259 / 0.5062 | 0.0055 / 0.0140 | 0.0215 / 0.0541 |
| lognormal | sorted | 2.0815 / 6.6181 | 0.0250 / 0.0575 | 0.0590 / 0.1536 |
| student_t3 | generated/shuffled | 0.1411 / 0.2662 | 0.0015 / 0.0055 | 0.0125 / 0.0253 |
| student_t3 | sorted | 1.7413 / 2.2098 | 0.0095 / 0.0155 | 0.0387 / 0.0844 |
| contaminated_gaussian | generated/shuffled | 0.2100 / 0.4300 | 0.0010 / 0.0020 | 0.0085 / 0.0328 |
| contaminated_gaussian | sorted | 7.3926 / 14.5413 | 0.0128 / 0.0195 | 0.1317 / 0.3038 |

A study-only exact-previous-prefix comparator uses exact prefix medians and the exact median of their deviations. Its final location is the exact final median. Comparing it with exact final-center MAD shows order sensitivity caused by the centering definition (including its first-row exclusion), without P² approximation. These comparisons are not an additive decomposition of correlation differences:

| Distribution | Order group | Max pair-correlation difference from exact final-center MAD, median / max |
|---|---|---:|
| gaussian | generated/shuffled | 0.0036 / 0.0188 |
| gaussian | sorted | 0.0096 / 0.0188 |
| lognormal | generated/shuffled | 0.0256 / 0.0484 |
| lognormal | sorted | 0.0680 / 0.1304 |
| student_t3 | generated/shuffled | 0.0114 / 0.0308 |
| student_t3 | sorted | 0.0202 / 0.0392 |
| contaminated_gaussian | generated/shuffled | 0.0035 / 0.0328 |
| contaminated_gaussian | sorted | 0.0109 / 0.0329 |

The deterministic correlation-only negative control gives 0 alarms, estimate -0.200000, historical accepted-pair target -0.2, and current-regime sample correlation -1.000000. Its marginal acceptance bits are constant; the detector has no correlation-change signal.

The deterministic marginal-shift positive control gives 2 alarms and coordinate reset counts [1, 1]. This verifies one designed sensitivity case, not detection power or a false-alarm calibration across distributions.

For every case, the production readout matches an independent Pearson calculation on its actual accepted rows at rtol=1e-12, atol=1e-14. Undefined pairs are excluded from comparator-difference metrics. This numerical check is distinct from the approximation characterization above.

Configuration, source/environment fingerprint, input hashes, bounds and per-case metrics are retained in config.json, provenance.json and raw.jsonl. Reproduction commands are in the studies/README.md file.
