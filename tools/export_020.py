"""Create the portable export from the two complete, hash-bound original runs."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile

from report_020 import RUNS, bench, file_digest, replay


def export(source, target):
    count, expected = RUNS[source.name]
    original = bench.load(source)
    if original["manifest_sha256"] != expected:
        raise bench.EvidenceError("Unexpected original manifest")
    records = bench.records(source, original)
    summary = bench.summarize_observations(original, records)
    archived = json.loads((source / "summary.json").read_text())
    if len(records) != count or summary != archived:
        raise bench.EvidenceError("Original count or summary differs")
    manifest = deepcopy(original)
    env = manifest["environment"]
    env.update(executable="environment/bin/python", prefix="environment")
    env["machine"]["hostname"] = "benchmark-host"
    for lib in env["native_libraries"]:
        lib["filepath"] = "native-libraries/" + Path(lib["filepath"]).name
    env["numba_compilation"]["effective"]["CUDA_INCLUDE_PATH"] = "cuda/include"
    for variant, comparator in manifest["comparators"].items():
        comparator["package"] = f"comparators/{variant}/force_toolkit"
        comparator["wheel"] = "wheels/" + Path(comparator["wheel"]).name
    manifest["source_manifest_sha256"] = manifest.pop("manifest_sha256")
    manifest["source_manifest_file_sha256"] = file_digest(source / "manifest.json")
    manifest["export_version"] = 1
    target.mkdir()
    path = target / "records.jsonl.gz"
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as out:
        for job in manifest["jobs"]:
            row = deepcopy(records[job["job_id"]])
            del row["environment"]  # bench.records verified equality with the source manifest.
            row["runtime_identity"]["package"] = manifest["comparators"][job["variant"]]["package"]
            row["source_record_sha256"] = file_digest(source / "records" / (job["job_id"] + ".json"))
            out.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    manifest["records_sha256"] = file_digest(path)
    manifest["manifest_sha256"] = bench.digest(manifest)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _, result = replay(target)
    if dict(result, manifest_sha256=expected) != archived:
        raise bench.EvidenceError("Portable replay differs from the original summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh export directory")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        bundle = Path(tmp) / "export"
        bundle.mkdir()
        for name in RUNS:
            export(args.original_runs / name, bundle / name)
        bundle.rename(args.output)


if __name__ == "__main__":
    main()
