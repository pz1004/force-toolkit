# Reconstruct the performance table

From the source archive, checkout, or extracted release evidence bundle, run:

```sh
python tools/report_020.py
```

The command reads individual measurements and reconstructs the warmed aggregate table, exploratory intervals and cold startup medians. NumPy is sufficient for replay. The measured versions were **0.2.0 and 0.1.2**. Each ratio divides baseline time by candidate time.

| Family | Warmed ratio | Exploratory 95% interval |
|---|---:|---:|
| Streaming MAD | 13.907× | 13.878–13.936 |
| Stationary MAD | 6.937× | 6.909–6.964 |
| FORCE | 1.001× | 0.995–1.011 |
| Dual-Bound | 0.997× | 0.994–1.000 |

The 1,600 fresh serial workers cover 16 configurations, five seeds and five randomized session passes. Each cell contains two pairs with opposite execution orders. Streaming measures post-calibration processing and stationary measures the full fit. The geometric means give equal weight to each configuration. The 10,000 crossed session/seed bootstrap draws reuse the same sampled indices across configurations. The intervals describe this grid on one machine. Five levels of each factor limit their interpretation. [Final confirmation manifest](evidence/020-confirmation-final/manifest.json)

Collection used Linux CPU affinity on an Intel Core Ultra 7 270K Plus, one pinned logical CPU and one numerical thread, Python 3.13.9, NumPy 2.3.5, Numba 0.65.1 and llvmlite 0.47.0. Cold collection used six pairs for each of three workloads with an empty compilation cache per worker. Startup runs from process launch through the first warm-up matrix (calibration size plus 40 streaming rows). For the clean 2,000 × 20 streaming MAD case with single-row updates, the median rose from 1.027 s to 3.439 s. [Cold manifest](evidence/020-cold/manifest.json)

## Export format 1

Each manifest retains the original schedule, configuration, input hashes, comparator source trees, wheel hashes, harness fingerprints, machine and compilation settings. It records the canonical digest and file SHA-256 of its original manifest. Each gzipped JSON line carries the original manifest digest and worker-file SHA-256, job and pair identifiers, session, seed, ordering, input/output hashes, imported module hashes, timestamps and measured values. Durations ending in `_s` are seconds and fields ending in `_bytes` are bytes. The manifest holds the environment common to every validated worker.

The export replaces comparator package and wheel paths with role-relative names. Interpreter paths become relative to `environment`, native libraries retain their basenames, the CUDA include setting uses `cuda/include` and the host has the label `benchmark-host`. SHA-256 identifies the compressed observations and the portable manifest has its own canonical digest. The exporter compared the full replayed summaries with both originals. [Exporter](../tools/export_020.py) · [Validation and aggregation](revision020.py)

The first confirmation attempt ended after 555 workers for a callback failure-boundary fix. Its candidate wheel SHA-256 began `335862b2`. The complete confirmation and cold runs used wheel `f45e96cc`. All exported worker records are bound to one of those two final manifests. The interrupted run contributes zero observations. [Portable confirmation observations](evidence/020-confirmation-final/records.jsonl.gz) · [Portable cold observations](evidence/020-cold/records.jsonl.gz)

To collect a fresh Linux experiment, install the benchmark extras and run `python benchmarks/revision020.py --suite primary --cpu N --candidate-wheel WHEEL --output DIRECTORY` with an allowed logical CPU, a wheel matching the current source, and a fresh directory. That command starts a new 1,600-worker measurement of the selected code. CI only replays the archived evidence.
