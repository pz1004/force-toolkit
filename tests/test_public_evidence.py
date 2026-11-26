"""The release table must come from complete, correctly paired original runs."""
import gzip
import json

import pytest

from tools.report_020 import EVIDENCE, RUNS, bench, replay, validate_observations


def test_archived_paper_results():
    primary_manifest, warmed = replay(EVIDENCE / "020-confirmation-final")
    cold_manifest, cold = replay(EVIDENCE / "020-cold")
    assert primary_manifest["comparators"] == cold_manifest["comparators"]
    assert warmed["measurements"] == 1600
    assert cold["measurements"] == 36
    assert {k: round(v["speedup"], 3) for k, v in warmed["aggregates"].items()} == {
        "streaming_mad": 13.907, "stationary_mad": 6.937, "force": 1.001, "dual": 0.997,
    }
    mad = next(c for c in cold["configurations"] if c["id"] == "mad-2000x20-clean-b1")
    assert [round(mad["phases"][v]["startup_to_first_result_s"], 3)
            for v in ("baseline", "candidate")] == [1.027, 3.439]


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("defect", ["missing", "duplicate", "interrupted"])
def test_reject_invalid_observations(run, defect):
    directory = EVIDENCE / run
    manifest = bench.load(directory)
    with gzip.open(directory / "records.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if defect == "missing":
        rows.pop()
        message = "Missing"
    elif defect == "duplicate":
        rows[-1] = rows[0]
        message = "Duplicate"
    else:
        rows[0]["manifest_sha256"] = "40071eda9a787fd4d98155c7e0160769f80f55bd900ea8182fcf264347430c0d"
        message = "different original run"
    with pytest.raises(bench.EvidenceError, match=message):
        validate_observations(manifest, rows)
