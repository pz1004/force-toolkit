"""Matched timing contrasts with fixed-design and replicated-level scope."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from benchmarks._protocol import SCHEMA, load_manifest, read_records
from benchmarks._comparators import VARIANTS


def matched_intervals(logs, *, resamples=10000, seed=20261109, cycle_length=3):
    """Contrast/configuration/session/seed/repeat log ratios from matched triplets.

    Both contrasts reuse each sampled complete Latin cycle. Independent cells
    have independent cycle draws; common session/seed draws preserve measured
    blocks across configurations. One three-triplet cycle cannot identify a
    conditional repetition interval. No simultaneous-coverage claim is made.
    """
    import numpy as np

    logs = np.asarray(logs, dtype=float)
    if logs.ndim != 5 or not np.isfinite(logs).all() or min(logs.shape) < 1:
        raise ValueError('Expected finite contrast/configuration/session/seed/repeat log ratios')
    contrasts, configurations, sessions, seeds, repeats = logs.shape
    rng = np.random.default_rng(seed)
    conditional = [None] * contrasts
    clustered = [None] * contrasts
    cycles = repeats // cycle_length if repeats % cycle_length == 0 else 0
    if cycles >= 2:
        # Lexicographically sort entire contrast vectors, preserving the matched
        # triplet while making Monte Carlo output invariant to repeat labels.
        cycle_means = logs.reshape(contrasts, configurations, sessions, seeds, cycles, cycle_length).mean(axis=-1)
        order = np.lexsort(tuple(cycle_means[index] for index in range(contrasts - 1, -1, -1)), axis=-1)
        ordered = np.take_along_axis(cycle_means, order[None, ...], axis=-1)
        draws = np.zeros((contrasts, resamples))
        for configuration in range(configurations):
            indices = rng.integers(0, cycles, (resamples, sessions, seeds, cycles))
            draws += np.take_along_axis(ordered[:, configuration, None, ...],
                                       indices[None, ...], axis=-1).mean(axis=(2, 3, 4)) / configurations
        conditional = np.exp(np.quantile(draws, [.025, .975], axis=1)).T.tolist()
    if sessions >= 2 and seeds >= 2:
        means = logs.mean(axis=-1)
        session_ids = rng.integers(0, sessions, (resamples, sessions))
        seed_ids = rng.integers(0, seeds, (resamples, seeds))
        draws = np.zeros((contrasts, resamples))
        for configuration in range(configurations):
            cell = means[:, configuration]
            draws += cell[:, session_ids[:, :, None], seed_ids[:, None, :]].mean(axis=(2, 3)) / configurations
        clustered = np.exp(np.quantile(draws, [.025, .975], axis=1)).T.tolist()
    return [{
        'speedup': float(np.exp(values.mean())),
        'conditional_fixed_design_ci95': conditional[index],
        'session_seed_cluster_ci95': clustered[index],
        'session_speedups': np.exp(values.mean(axis=(0, 2, 3))).tolist(),
        'seed_speedups': np.exp(values.mean(axis=(0, 1, 3))).tolist(),
        'configurations': configurations, 'sessions': sessions, 'input_seeds': seeds,
        'matched_repetitions_per_cell': repeats,
        'conditional_resampling_unit': 'complete Latin three-triplet cycle' if cycle_length == 3 else 'single paired repetition',
        'conditional_cycles_per_cell': cycles,
        'exploratory': sessions < 10 or seeds < 10,
        'scope': 'recorded configurations, input-generator seeds, serial session passes, and machine; not arbitrary datasets or hardware',
        'method': 'equal-configuration geometric mean; complete-cycle within-cell bootstrap when replicated; crossed session/input-seed cluster bootstrap preserving configuration blocks; both contrasts share draws',
        'resamples': resamples, 'bootstrap_seed': seed,
    } for index, values in enumerate(logs)]


def intervals(logs, *, resamples=10000, seed=20261109):
    """Single-contrast compatibility helper; shape configuration/session/seed/repeat."""
    import numpy as np
    values = np.asarray(logs, dtype=float)
    if values.ndim != 4:
        raise ValueError('Expected finite configuration/session/seed/repeat log ratios')
    return matched_intervals(values[None, ...], resamples=resamples, seed=seed, cycle_length=1)[0]


def regression_assessment(summary):
    ci = summary['session_seed_cluster_ci95']
    if ci is None:
        return 'insufficient_replication'
    if ci[1] < 1 / 1.1:
        return 'regression_over_10pct_supported'
    if ci[0] > 1 / 1.1:
        return 'noninferiority_at_10pct_supported_exploratory' if summary['exploratory'] else 'noninferiority_at_10pct_supported'
    return 'inconclusive_at_10pct'


def summarize(directory, *, write=True):
    import numpy as np

    directory = Path(directory)
    manifest = load_manifest(directory)
    records = read_records(directory, manifest)
    design = manifest['design']
    sessions, seeds, repeats = design['sessions'], design['seeds'], design['repetitions']
    lookup = {(r['job']['workload_id'], r['job']['session'], r['job']['seed'],
               r['job']['repeat'], r['job']['variant']): r for r in records.values()}
    results, mad_logs = [], []
    for configuration in design['workloads']:
        name = configuration['id']
        keys = [(name, session, seed, repeat, variant)
                for session in range(sessions) for seed in seeds for repeat in range(repeats)
                for variant in VARIANTS]
        if any(key not in lookup for key in keys):
            continue
        phase = 'fit_s' if configuration['mode'] == 'batch' else 'processing_s'
        logs = np.empty((2, sessions, len(seeds), repeats))
        for session in range(sessions):
            for seed_index, seed in enumerate(seeds):
                for repeat in range(repeats):
                    base = lookup[(name, session, seed, repeat, 'baseline')]
                    prior = lookup[(name, session, seed, repeat, 'optimized_010')]
                    candidate = lookup[(name, session, seed, repeat, 'accelerated')]
                    logs[0, session, seed_index, repeat] = np.log(base[phase] / candidate[phase])
                    logs[1, session, seed_index, repeat] = np.log(prior[phase] / candidate[phase])
        acceleration, revision = matched_intervals(logs[:, None, ...])
        timing = {}
        for variant in VARIANTS:
            chosen = [lookup[key] for key in keys if key[-1] == variant]
            timing[variant] = {
                field: (None if chosen[0][field] is None else float(statistics.median(r[field] for r in chosen)))
                for field in ('import_s', 'first_use_s', 'construction_s', 'calibration_s', 'processing_s',
                              'fit_s', 'readout_s', 'scale_scatter_readout_s', 'peak_process_bytes')
            }
            timing[variant]['retained_state'] = chosen[-1]['retained_state']
            timing[variant]['retained_state_example_job_id'] = chosen[-1]['job_id']
            state_bytes = [row['retained_state']['bytes'] for row in chosen]
            timing[variant]['retained_state_bytes_across_measurements'] = {
                'median': float(statistics.median(state_bytes)),
                'minimum': min(state_bytes), 'maximum': max(state_bytes),
            }
        results.append({'id': name, 'configuration': configuration, 'primary_phase': phase,
                        'acceleration': acceleration, 'revision_regression': revision,
                        'timings': timing, 'regression_assessment': regression_assessment(revision)})
        if configuration['algorithm'] == 'mad' and configuration['mode'] == 'stream':
            mad_logs.append(logs)
    aggregates = matched_intervals(np.stack(mad_logs, axis=1)) if mad_logs else (None, None)
    complete = len(records) == len(manifest['jobs'])
    result = {
        'schema_version': SCHEMA, 'run_id': manifest['run_id'], 'manifest_sha256': manifest['manifest_sha256'],
        'complete': complete, 'measurements': len(records), 'expected_measurements': len(manifest['jobs']),
        'results': results, 'aggregate_streaming_mad': aggregates[0],
        'aggregate_revision_streaming_mad': aggregates[1],
        'comparisons': {'acceleration': 'unoptimized integration / current',
                        'revision_regression': 'optimized 0.1.0 wheel / current'},
        'historical_data_reused': False,
        'interpretation': 'Few-session/seed intervals are exploratory, without a simultaneous-coverage guarantee. Serial session passes are independently randomized, but their independence from host thermal/load history is not guaranteed.',
    }
    if write:
        (directory / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        lines = ['# Three-variant benchmark evidence', '',
                 f"Run `{manifest['run_id']}`. {len(records)}/{len(manifest['jobs'])} validated measurements.", '',
                 'Acceleration compares unoptimized integration / current. Revision regression compares optimized 0.1.0 wheel / current. Both ratios use the same matched triplets and resampling draws. Batch rows use full fit; stream rows use warmed processing.', '',
                 '**Intervals with fewer than ten sessions or seeds are exploratory.** Conditional intervals resample complete Latin three-triplet cycles and require at least two cycles per cell; the default three repeats provide only one cycle, so no conditional interval is reported. Crossed-cluster intervals resample measured sessions and generator seeds. No simultaneous confidence guarantee is claimed.', '']
        def fmt(ci):
            return 'Not estimated' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        for comparison in ('acceleration', 'revision_regression'):
            lines += [f"## {result['comparisons'][comparison]}", '',
                      '| Workload | Phase | Ratio | Conditional 95% interval | Session/seed cluster 95% interval |',
                      '|---|---|---:|---:|---:|']
            for row in results:
                value = row[comparison]
                lines.append(f"| {row['id']} | {row['primary_phase']} | {value['speedup']:.3f} | {fmt(value['conditional_fixed_design_ci95'])} | {fmt(value['session_seed_cluster_ci95'])} |")
            lines.append('')
        for label, aggregate in zip(('Acceleration', 'Revision comparison'), aggregates):
            if aggregate:
                lines.append(f"{label}, aggregate streaming MAD ({aggregate['configurations']} configurations): {aggregate['speedup']:.3f}×; exploratory cluster interval {fmt(aggregate['session_seed_cluster_ci95'])}.")
        lines += ['', result['interpretation'], 'Actual source, wheel, loaded module identities, effective JIT configuration and numerical environment are bound in the manifest and records. Historical timings are never reused.']
        (directory / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return result
