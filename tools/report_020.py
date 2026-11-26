"""Validate and replay the archived Linux 0.2.0 versus 0.1.2 timings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import revision020 as bench

EVIDENCE = ROOT / "benchmarks" / "evidence"
# Canonical digests of the original schema-4 manifests, before path substitution.
RUNS = {
    "020-confirmation-final": (1600, "9f76e088bd7b8c9ba900bf6d0b7eee2236605f36e252b9d583f1fdbe95e68cef"),
    "020-cold": (36, "d9a57e7b8fa8f302077696df6e8bd48dd0d51ee7e5164a4de2570dfa96cb4c5b"),
}
VERSIONS = {"baseline": "0.1.2", "candidate": "0.2.0"}


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_observations(manifest, rows):
    """Bind each row to its source run, scheduled pair, imported code and input."""
    observations = {}
    outputs = {}
    for row in rows:
        if row["manifest_sha256"] != manifest["source_manifest_sha256"]:
            raise bench.EvidenceError("Observation belongs to a different original run")
        record = dict(row, environment=manifest["environment"], manifest_sha256=manifest["manifest_sha256"])
        job = bench.validate(record, manifest)
        if record["runtime_identity"]["version"] != VERSIONS[job["variant"]]:
            raise bench.EvidenceError("Unexpected measured version")
        if job["job_id"] in observations:
            raise bench.EvidenceError("Duplicate observation: " + job["job_id"])
        observations[job["job_id"]] = record
        key = (job["workload_id"], job["seed"])
        if outputs.setdefault(key, record["output_sha256"]) != record["output_sha256"]:
            raise bench.EvidenceError("Output differs within a workload and seed")
    if len(observations) != len(manifest["jobs"]):
        raise bench.EvidenceError("Missing scheduled observations")
    return observations


def replay(directory):
    count, source_digest = RUNS[directory.name]
    manifest = bench.load(directory)
    if manifest["source_manifest_sha256"] != source_digest or len(manifest["jobs"]) != count:
        raise bench.EvidenceError("Unexpected original run or design")
    path = directory / "records.jsonl.gz"
    if file_digest(path) != manifest["records_sha256"]:
        raise bench.EvidenceError("Exported observations failed SHA-256 verification")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        observations = validate_observations(manifest, (json.loads(line) for line in stream))
    return manifest, bench.summarize_observations(manifest, observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    manifest, warmed = replay(args.evidence / "020-confirmation-final")
    _, cold = replay(args.evidence / "020-cold")
    env = manifest["environment"]
    print("Warmed speedup, 0.1.2 time divided by 0.2.0 time (1,600 workers)")
    print("| Family | Ratio | Exploratory 95% interval |")
    print("|---|---:|---:|")
    for name, entry in warmed["aggregates"].items():
        lower, upper = entry["exploratory_ci95"]
        print(f'| {name} | {entry["speedup"]:.3f}x | {lower:.3f} to {upper:.3f} |')
    print("\nCold startup to first matrix, median seconds (36 workers)")
    print("| Workload | 0.1.2 | 0.2.0 |")
    print("|---|---:|---:|")
    for entry in cold["configurations"]:
        times = [entry["phases"][v]["startup_to_first_result_s"] for v in VERSIONS]
        print(f'| {entry["id"]} | {times[0]:.3f} | {times[1]:.3f} |')
    print(f'\n{env["machine"]["cpu_model"]}, {env["machine"]["platform"]}')
    print(f'CPU affinity {manifest["design"]["affinity"]}, one numerical thread. {env["packages"]}')
    print("Measured candidate wheel SHA-256", manifest["comparators"]["candidate"]["wheel_sha256"])


if __name__ == "__main__":
    main()
