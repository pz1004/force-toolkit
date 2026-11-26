# force-toolkit 0.2.1

This release adds two executable examples, package citation metadata and a portable export of the 0.2.0 versus 0.1.2 timing measurements. The five estimator interfaces carry forward. Comparison of every executable module against the measured 0.2.0 wheel identified one change, the version declaration. [Module hashes](../provenance/measured-release.json) · [Validation](validation.md)

## Downloads

| File | Contents |
|---|---|
| `force_toolkit-0.2.1-py3-none-any.whl` | Installable Python package and license notices |
| `force_toolkit-0.2.1.tar.gz` | Library source, examples, tests, documentation and benchmark replay |
| `force-toolkit-evidence-0.2.1.zip` | Individual timing measurements, 0.2.1 MAD characterization, wheel checks and replay tools |
| `SHA256SUMS` | SHA-256 digests of the three downloads above |

Download all four files into one directory. On Linux, run `sha256sum -c SHA256SUMS` there. On macOS use `shasum -a 256 -c SHA256SUMS`. [Release downloads](https://github.com/pz1004/force-toolkit/releases/tag/v0.2.1) · [Installation](installation.md)

To assemble these files locally, place the wheel, source archive and evidence ZIP in `dist/`. Run `python tools/check_distribution.py --stage-release build/release` from the checkout. It verifies the Python distributions, copies the three named artifacts into a new directory and computes the checksums there.
