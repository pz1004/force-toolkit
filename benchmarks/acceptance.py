"""Assess a declared experiment from validated raw records, outside ordinary CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._statistics import summarize


def assess(directory):
    # Never trust a stored summary or a stale optimized-sha256.json file.
    summary = summarize(directory)
    mad = summary['aggregate_streaming_mad']
    mad_ci = None if mad is None else mad['session_seed_cluster_ci95']
    mad_improvement = mad_ci is not None and mad_ci[0] > 1
    regressions = [r['id'] for r in summary['results']
                   if r['regression_assessment'] == 'regression_over_10pct_supported']
    unresolved = [r['id'] for r in summary['results']
                  if not r['regression_assessment'].startswith('noninferiority_at_10pct_supported')]
    supported = summary['complete'] and mad_improvement and not unresolved
    result = {
        'run_id': summary['run_id'], 'manifest_sha256': summary['manifest_sha256'],
        'complete': summary['complete'], 'performance_criterion_supported': supported,
        'assessment': ('supported_within_recorded_design_exploratory' if supported else 'not_established'),
        'aggregate_streaming_mad': mad,
        'aggregate_revision_streaming_mad': summary['aggregate_revision_streaming_mad'],
        'comparisons': summary['comparisons'],
        'acceleration_criterion_supported': bool(summary['complete'] and mad_improvement),
        'revision_nonregression_criterion_supported': bool(summary['complete'] and not unresolved),
        'regressions_over_10pct_supported': regressions,
        'primary_workloads_without_10pct_noninferiority_support': unresolved,
        'correctness': 'Separate prerequisite; this performance report does not certify correctness.',
        'scope': ('Few-session and few-seed intervals are exploratory, not evidence of universal improvement. '
                  'Acceleration is against unoptimized integration; the 10% regression criterion is against optimized 0.1.0, not against 0.1.1. '
                  'An inconclusive regression interval is not proof of noninferiority. '
                  'The streaming MAD aggregate excludes the separately reported two-pass MAD workload.'),
        'historical_data_reused': False,
    }
    (Path(directory) / 'acceptance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args(argv)
    result = assess(args.run)
    print(json.dumps(result, indent=2))
    return 0 if result['performance_criterion_supported'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
