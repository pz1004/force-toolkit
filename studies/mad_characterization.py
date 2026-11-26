"""Characterize approximation and order sensitivity without changing MAD-FORCE.

Run from the repository root:
    python studies/mad_characterization.py --output studies/results/my-mad-run

The exact final-center MAD comparator is an alternative trimming procedure, not
an oracle for the population target of causal MAD-FORCE. Numerical conformance
is checked separately against Pearson correlation on the actual accepted rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

import force_toolkit
from force_toolkit import MADForceEstimator, StreamingMADForceEstimator

MAD_SCALE = 1.4826
DISTRIBUTIONS = ("gaussian", "lognormal", "student_t3", "contaminated_gaussian")
ORDERS = ("generated", "ascending_first", "descending_first", "shuffle_0", "shuffle_1", "shuffle_2")


def sha256_array(values):
    values = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def source_hashes():
    library = Path(force_toolkit.__file__).resolve().parent
    result = {"study": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result.update({"force_toolkit/" + str(path.relative_to(library)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(library.rglob("*.py"))})
    return result


def make_data(distribution, seed, n, p):
    rng = np.random.default_rng(seed)
    if distribution == "student_t3":
        data = 0.6 * rng.standard_t(3, size=(n, 1)) + 0.8 * rng.standard_t(3, size=(n, p))
    else:
        data = 0.6 * rng.normal(size=(n, 1)) + 0.8 * rng.normal(size=(n, p))
        if distribution == "lognormal":
            data = np.exp(data)
        elif distribution == "contaminated_gaussian":
            rows = rng.choice(n, n // 10, replace=False)
            data[rows] += 15 * rng.standard_t(3, size=(len(rows), p))
        elif distribution != "gaussian":
            raise ValueError(distribution)
    return np.ascontiguousarray(data)


def ordered_data(values, order, seed):
    if order == "generated":
        return values.copy()
    indices = np.argsort(values[:, 0], kind="stable")
    if order == "ascending_first":
        return values[indices]
    if order == "descending_first":
        return values[indices[::-1]]
    permutation = int(order.split("_")[1])
    return values[np.random.default_rng(1_000_000 + seed * 10 + permutation).permutation(len(values))]


def final_center_bounds(values, lam=3.0):
    location = np.median(values, axis=0)
    scale = MAD_SCALE * np.median(np.abs(values - location), axis=0)
    return location - lam * scale, location + lam * scale, scale


def exact_prefix_bounds(values, lam=3.0):
    """Counterfactual exact quantiles with the same previous-prefix centering.

    This study-only O(N²p) calculation isolates the definition's order dependence
    from P² approximation when compared with exact final-center MAD. It is not
    exposed as a production estimator.
    """
    previous_medians = np.stack([np.median(values[:end], axis=0) for end in range(1, len(values))])
    deviations = np.abs(values[1:] - previous_medians)
    location = np.median(values, axis=0)
    scale = MAD_SCALE * np.median(deviations, axis=0)
    return location - lam * scale, location + lam * scale


def accepted_pair_reference(values, accepted):
    """Independent ordinary-data Pearson readout using NumPy's correlation."""
    p = values.shape[1]
    correlation = np.eye(p)
    defined = np.zeros((p, p), dtype=bool)
    counts = np.zeros((p, p), dtype=np.int64)
    for j in range(p):
        counts[j, j] = np.count_nonzero(accepted[:, j])
        for k in range(j + 1, p):
            rows = values[accepted[:, j] & accepted[:, k]][:, [j, k]]
            counts[j, k] = counts[k, j] = len(rows)
            if len(rows) < 2 or np.any(np.ptp(rows, axis=0) == 0):
                continue
            correlation[j, k] = correlation[k, j] = np.corrcoef(rows.T)[0, 1]
            defined[j, k] = defined[k, j] = True
    return correlation, defined, counts


def summarize_values(records, field):
    values = np.array([r[field] for r in records if r[field] is not None], dtype=float)
    if not len(values):
        return {"median": None, "maximum": None}
    return {"median": float(np.median(values)), "maximum": float(np.max(values))}


def format_summary(summary):
    if summary["median"] is None:
        return "undefined"
    return f"{summary['median']:.4f} / {summary['maximum']:.4f}"


def controls():
    x = np.tile([-2.0, -1.0, 1.0, 2.0], 300)
    y = x.copy()
    y[600:] *= -1
    data = np.column_stack([x, y])
    estimator = StreamingMADForceEstimator(2).fit(data)
    diagnostics = estimator.get_diagnostics()["details"]
    negative = {
        "input_sha256": sha256_array(data), "rows": len(data), "change_at": 600,
        "initial_warmup": 200, "alarms": diagnostics["alarm_count"],
        "acceptance_rates": diagnostics["acceptance_rates"],
        "estimate": float(estimator.finalize()[0, 1]),
        "historical_accepted_target": -0.2,
        "current_regime_sample_correlation": float(np.corrcoef(data[600:].T)[0, 1]),
    }
    initial = np.tile([-2.0, -1.0, 1.0, 2.0], 250)
    tail = np.tile([-2.0, -1.0, 1.0, 2.0], 25) + 20
    shifted = np.column_stack([np.r_[initial, tail]] * 2)
    estimator = StreamingMADForceEstimator(2).fit(shifted)
    diagnostics = estimator.get_diagnostics()["details"]
    positive = {
        "input_sha256": sha256_array(shifted), "rows": len(shifted), "change_at": len(initial),
        "alarms": diagnostics["alarm_count"], "alarm_locations": diagnostics["alarm_locations"],
        "reset_counts": diagnostics["reset_counts"],
    }
    return {"correlation_only_negative_control": negative, "marginal_shift_positive_control": positive}


def run(config):
    records = []
    for distribution in config["distributions"]:
        for seed in config["seeds"]:
            data = make_data(distribution, seed, config["n"], config["p"])
            base_hash = sha256_array(data)
            for order in config["orders"]:
                values = ordered_data(data, order, seed)
                estimate = MADForceEstimator(config["p"], lam=config["lam"]).fit(values)
                actual = estimate.finalize()
                lower, upper = estimate.get_diagnostics()["details"]["thresholds"]
                exact_lower, exact_upper, exact_scale = final_center_bounds(values, config["lam"])
                prefix_lower, prefix_upper = exact_prefix_bounds(values, config["lam"])
                prefix_accepted = (values >= prefix_lower) & (values <= prefix_upper)
                prefix_comparator, prefix_valid, _ = accepted_pair_reference(values, prefix_accepted)
                accepted = (values >= lower) & (values <= upper)
                exact_accepted = (values >= exact_lower) & (values <= exact_upper)
                oracle, valid, counts = accepted_pair_reference(values, accepted)
                comparator, comparator_valid, comparator_counts = accepted_pair_reference(values, exact_accepted)
                np.testing.assert_allclose(actual, oracle, rtol=1e-12, atol=1e-14)
                jointly_valid = np.triu(valid & comparator_valid, 1)
                differences = actual[jointly_valid] - comparator[jointly_valid]
                prefix_jointly_valid = np.triu(prefix_valid & comparator_valid, 1)
                prefix_differences = prefix_comparator[prefix_jointly_valid] - comparator[prefix_jointly_valid]
                bound_difference = np.maximum(np.abs(lower - exact_lower), np.abs(upper - exact_upper))
                standardized = bound_difference / exact_scale if np.all(exact_scale > 0) else None
                records.append({
                    "distribution": distribution, "seed": seed, "order": order,
                    "base_input_sha256": base_hash, "ordered_input_sha256": sha256_array(values),
                    "causal_p2_bounds": [lower.tolist(), upper.tolist()],
                    "exact_final_center_bounds": [exact_lower.tolist(), exact_upper.tolist()],
                    "exact_final_center_scale": exact_scale.tolist(),
                    "exact_previous_prefix_bounds": [prefix_lower.tolist(), prefix_upper.tolist()],
                    "exact_prefix_vs_final_center_acceptance_disagreement": float(np.mean(prefix_accepted != exact_accepted)),
                    "exact_prefix_vs_final_center_correlation_max_abs_difference": None if not len(prefix_differences) else float(np.max(np.abs(prefix_differences))),
                    "bound_max_abs_difference": float(np.max(bound_difference)),
                    "bound_max_difference_in_exact_scale_units": None if standardized is None else float(np.max(standardized)),
                    "coordinate_acceptance_disagreement": float(np.mean(accepted != exact_accepted)),
                    "maximum_pair_count_difference": int(np.max(np.abs(counts - comparator_counts))),
                    "production_estimable_offdiagonal_pairs": int(np.count_nonzero(np.triu(valid, 1))),
                    "comparator_estimable_offdiagonal_pairs": int(np.count_nonzero(np.triu(comparator_valid, 1))),
                    "jointly_estimable_offdiagonal_pairs": int(np.count_nonzero(jointly_valid)),
                    "correlation_max_abs_difference": None if not len(differences) else float(np.max(np.abs(differences))),
                    "correlation_rmse_difference": None if not len(differences) else float(np.sqrt(np.mean(differences ** 2))),
                    "same_acceptance_oracle_max_abs_error": float(np.max(np.abs(actual - oracle))),
                })
    return records


def write_report(output, config, records, control_results):
    groups = {}
    for record in records:
        # Sorted inputs are separate adversarial order experiments, not iid replicates.
        order_group = "sorted" if record["order"] in ("ascending_first", "descending_first") else "generated/shuffled"
        groups.setdefault((record["distribution"], order_group), []).append(record)
    summary = []
    for (distribution, order_group), rows in groups.items():
        summary.append({
            "distribution": distribution, "order_group": order_group, "cases": len(rows),
            "acceptance_disagreement": summarize_values(rows, "coordinate_acceptance_disagreement"),
            "correlation_difference": summarize_values(rows, "correlation_max_abs_difference"),
            "bound_difference_scale_units": summarize_values(rows, "bound_max_difference_in_exact_scale_units"),
            "exact_prefix_definition_correlation_difference": summarize_values(rows, "exact_prefix_vs_final_center_correlation_max_abs_difference"),
        })
    (output / "summary.json").write_text(json.dumps({"groups": summary, "controls": control_results}, indent=2) + "\n")
    lines = [
        "# MAD approximation and order characterization", "",
        f"{len(records)} cases: {len(config['seeds'])} seeds, {config['n']} rows, {config['p']} features, "
        f"{len(config['distributions'])} distributions, {len(config['orders'])} orders per generated data set.", "",
        "These are descriptive finite-grid summaries, with permutations nested within the same generated data set. "
        "They are not independent replications and no confidence interval or population-accuracy claim is made.", "",
        "The comparator uses exact final-sample median/MAD bounds. Differences from it characterize a different "
        "trimming procedure; they do not measure estimation error against a known population target. "
        "Sorted/reverse-sorted rows intentionally violate iid ordering to stress order dependence.", "",
        "| Distribution | Order group | Max bound difference in exact-scale units, median / max | Coordinate disagreement, median / max | Max pair-correlation difference, median / max |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        a = row["acceptance_disagreement"]
        c = row["correlation_difference"]
        b = row["bound_difference_scale_units"]
        lines.append(f"| {row['distribution']} | {row['order_group']} | {format_summary(b)} | {format_summary(a)} | {format_summary(c)} |")
    lines += ["", "A study-only exact-previous-prefix comparator uses exact prefix medians and the exact "
              "median of their deviations. Its final location is the exact final median. Comparing it with "
              "exact final-center MAD shows order sensitivity caused by the centering definition (including "
              "its first-row exclusion), without P² approximation. These comparisons are not an additive "
              "decomposition of correlation differences:", "",
              "| Distribution | Order group | Max pair-correlation difference from exact final-center MAD, median / max |",
              "|---|---|---:|"]
    for row in summary:
        value = row["exact_prefix_definition_correlation_difference"]
        lines.append(f"| {row['distribution']} | {row['order_group']} | {format_summary(value)} |")
    negative = control_results["correlation_only_negative_control"]
    positive = control_results["marginal_shift_positive_control"]
    lines += [
        "", f"The deterministic correlation-only negative control gives {negative['alarms']} alarms, "
        f"estimate {negative['estimate']:.6f}, historical accepted-pair target -0.2, and current-regime "
        f"sample correlation {negative['current_regime_sample_correlation']:.6f}. Its marginal acceptance "
        "bits are constant; the detector has no correlation-change signal.", "",
        f"The deterministic marginal-shift positive control gives {positive['alarms']} alarms and "
        f"coordinate reset counts {positive['reset_counts']}. This verifies one designed sensitivity case, "
        "not detection power or a false-alarm calibration across distributions.", "",
        "For every case, the production readout matches an independent Pearson calculation on its actual "
        "accepted rows at rtol=1e-12, atol=1e-14. Undefined pairs are excluded from comparator-difference metrics. "
        "This numerical check is distinct from the approximation characterization above.", "",
        "Configuration, source/environment fingerprint, input hashes, bounds and per-case metrics are retained "
        "in config.json, provenance.json and raw.jsonl. Reproduction commands are in the studies/README.md file.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    if args.n < 6 or args.p < 2 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("require n>=6, p>=2 and distinct seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a fresh study directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(n=args.n, p=args.p, seeds=args.seeds, lam=3.0, tail_mode="none",
                  distributions=list(DISTRIBUTIONS), orders=list(ORDERS), mad_scale=MAD_SCALE)
    before = source_hashes()
    records = run(config)
    control_results = controls()
    after = source_hashes()
    if before != after:
        raise RuntimeError("source changed during characterization; rerun with a fresh directory")
    import numba
    provenance = dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                      platform=platform.platform(), source_sha256=before,
                      statement="descriptive characterization; exact comparator is not population truth")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    write_report(args.output, config, records, control_results)
    print(f"Recorded {len(records)} cases in {args.output}")


if __name__ == "__main__":
    main()
