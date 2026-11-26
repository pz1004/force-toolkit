# Method credits and implementation lineage

The toolkit author is Sooyoung Jang. Cite the toolkit for the unified API and implementation, and the corresponding method paper for the estimator. The following revisions identify the imported source and verification documents.

| Method | Paper | Upstream source | Verification |
|---|---|---|---|
| FORCE | [Mathematics 14, 191](https://doi.org/10.3390/math14010191) | [68765dba](https://github.com/pz1004/force/tree/68765dba478e23fa72c894a649815dfbffce7888) | [Algorithm and legacy experiments](https://github.com/pz1004/force/blob/68765dba478e23fa72c894a649815dfbffce7888/VERIFICATION.md) |
| MAD-FORCE | [IEEE Access](https://doi.org/10.1109/ACCESS.2026.3704294) | [d265ae7d](https://github.com/pz1004/mad-force-public/tree/d265ae7d4b893aededcda09eadeb9525bf07428d) | [Algorithm coverage](https://github.com/pz1004/mad-force-public/blob/d265ae7d4b893aededcda09eadeb9525bf07428d/VERIFICATION.md) |
| Dual-Bound FORCE | [Mathematics 14, 3303](https://doi.org/10.3390/math14183303) | [9cac01dd](https://github.com/pz1004/dual-bound-force/tree/9cac01ddb7261b81dba68ec71186f4af9c1eb748) | [Validation guide](https://github.com/pz1004/dual-bound-force/blob/9cac01ddb7261b81dba68ec71186f4af9c1eb748/VERIFICATION.md) |

The import ledger identifies the original bytes, destinations and bytes at import, including adapted module imports and tests. The integration introduced a common lifecycle, explicit target names, pair diagnostics and compatible dependencies, and later compiled MAD’s sequential updates. Each predecessor retains its MIT notice. [Import hashes](imports.json) · [Licenses](../licenses) · [Migration](../docs/migration.md)

The tests bind the unoptimized source comparator and both required optimized wheel comparators to SHA-256. A module-by-module comparison to the measured 0.2.0 wheel records the version declaration as the only Python change in 0.2.1. [Unoptimized reference](baseline-sha256.json) · [0.1.2 reference](comparator-0.1.2.json) · [0.1.0 wheel identity](../benchmarks/_comparators.py) · [Measured runtime comparison](measured-release.json)

The November 23 validation run installed the measured wheel, ran 609 tests and reran the original 240-case characterization. Its configuration, individual case records, summary and report matched the 0.1.1 files byte for byte. Its source and environment fingerprint documents the 0.2.0 execution. [Validation record](validation-0.2.0.json) · [Study fingerprint](mad-study-0.2.0.json) · [Upstream example comparison](dual-bound-example.json)
