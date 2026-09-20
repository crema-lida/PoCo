"""Summarize the independent fragmentation experiment at polymer level."""
from collections import defaultdict
import json
import os
from config import OUTPUT_DIR, MODELS, CPU_THREADS, EVALUATION_SIZE

os.environ['OMP_NUM_THREADS'] = str(CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(CPU_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_THREADS)

import numpy as np

REPLICATES = 1000
METRICS = [f'{subset}_{metric}'
           for subset in ('all', 'single', 'single_nonring')
           for metric in ('auc', 'ap', 'matched_cut_recall')]
METRICS += ['matched_block_gap', 'mean_bond_cosine']
THRESHOLD_METRICS = ['f1', 'ari', 'n_fragments', 'mean_fragment_size',
                     'largest_fraction']
RNG = np.random.default_rng(20260910)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def value(row, key):
    result = row.get(key)
    return np.nan if result is None else float(result)


def estimate(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': None, 'ci95': None}
    indices = RNG.integers(n, size=(REPLICATES, n))
    interval = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {'n': n, 'mean': float(values.mean()), 'ci95': interval.tolist()}


def shuffle_means(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        if row['variant'] == 'same_type_shuffle':
            grouped[row['index']].append(row)
    means = {}
    for index, repeats in grouped.items():
        means[index] = {}
        for field in fields:
            values = [value(row, field) for row in repeats]
            available = [v for v in values if np.isfinite(v)]
            means[index][field] = (available[0] + float(np.mean(np.asarray(available) - available[0]))
                                   if available else None)
    return means


def paired_stats(left, right, fields):
    indices = sorted(left.keys() & right.keys())
    return {field: estimate([value(left[i], field) - value(right[i], field)
                             for i in indices]) for field in fields}


def grouped_stats(rows, fields):
    native = {row['index']: row for row in rows if row['variant'] == 'native'}
    shuffled = shuffle_means(rows, fields)
    result = {'n_structures': len(native),
              'native': {field: estimate([value(row, field) for row in native.values()])
                         for field in fields}}
    if shuffled:
        result['shuffle_mean'] = {
            field: estimate([value(row, field) for row in shuffled.values()])
            for field in fields}
        result['native_minus_shuffle'] = paired_stats(native, shuffled, fields)
    return result, native, shuffled


def pooled_f1(rows):
    counts = np.array([[row['tp'], row['fp'], row['fn']] for row in rows], dtype=float)
    if len(counts) == 0:
        return {'n': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'f1': None, 'ci95': None}
    total = counts.sum(axis=0)
    denominator = 2 * total[0] + total[1] + total[2]
    boot = counts[RNG.integers(len(counts), size=(REPLICATES, len(counts)))].sum(axis=1)
    boot_denominator = 2 * boot[:, 0] + boot[:, 1] + boot[:, 2]
    valid = boot_denominator > 0
    interval = (np.quantile(2 * boot[valid, 0] / boot_denominator[valid], [0.025, 0.975]).tolist()
                if valid.any() else None)
    return {'n': len(counts), 'tp': float(total[0]), 'fp': float(total[1]),
            'fn': float(total[2]),
            'f1': float(2 * total[0] / denominator) if denominator else None,
            'ci95': interval}


def paired_pooled_f1(left, right):
    indices = sorted(left.keys() & right.keys())
    a = np.array([[left[i][k] for k in ('tp', 'fp', 'fn')] for i in indices], dtype=float)
    b = np.array([[right[i][k] for k in ('tp', 'fp', 'fn')] for i in indices], dtype=float)
    if not indices:
        return {'n': 0, 'difference': None, 'ci95': None}
    def score(counts):
        den = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
        return np.divide(2 * counts[..., 0], den,
                         out=np.full(np.shape(den), np.nan), where=den > 0)
    boot = RNG.integers(len(indices), size=(REPLICATES, len(indices)))
    differences = score(a[boot].sum(axis=1)) - score(b[boot].sum(axis=1))
    differences = differences[np.isfinite(differences)]
    difference = score(a.sum(axis=0)) - score(b.sum(axis=0))
    return {'n': len(indices),
            'difference': float(difference) if np.isfinite(difference) else None,
            'ci95': np.quantile(differences, [0.025, 0.975]).tolist() if len(differences) else None}


def fmt(stat):
    if stat['mean'] is None:
        return 'NA (n=0)'
    lo, hi = stat['ci95']
    return f"{stat['mean']:.3f} [{lo:.3f}, {hi:.3f}]; n={stat['n']}"


def report(summary):
    lines = ['# BRICS correspondence and threshold sensitivity', '',
             '## Native representation metrics', '',
             '| Dataset | Model | Single nonring AUC | Matched-cut recall, single nonring | Matched block gap |',
             '|---|---|---|---|---|']
    for row in summary['metrics']:
        m = row['native']
        lines.append(f"| {row['dataset']} | {row['model']} | {fmt(m['single_nonring_auc'])} | "
                     f"{fmt(m['single_nonring_matched_cut_recall'])} | {fmt(m['matched_block_gap'])} |")
    lines += ['', '## Native minus same-type shuffle', '',
              '| Dataset | Model | Single nonring AUC gain | Matched-cut recall gain | Matched block gap gain |',
              '|---|---|---|---|---|']
    for row in summary['metrics']:
        if 'native_minus_shuffle' not in row:
            continue
        m = row['native_minus_shuffle']
        lines.append(f"| {row['dataset']} | {row['model']} | {fmt(m['single_nonring_auc'])} | "
                     f"{fmt(m['single_nonring_matched_cut_recall'])} | {fmt(m['matched_block_gap'])} |")
    lines += ['', '## Paired PoCo minus MLM', '',
              '| Dataset | Quantity | Single nonring AUC | Matched-cut recall | Matched block gap |',
              '|---|---|---|---|---|']
    for row in summary['comparisons']:
        for kind in ('native', 'context_gain'):
            m = row[kind]
            lines.append(f"| {row['dataset']} | {kind} | {fmt(m['single_nonring_auc'])} | "
                         f"{fmt(m['single_nonring_matched_cut_recall'])} | {fmt(m['matched_block_gap'])} |")
    lines += ['', '## Cutoff 0.6 without bond type weighting', '',
              '| Dataset | Model | n | Macro F1 | Pooled F1 | ARI | Fragments | Mean size | Largest fraction |',
              '|---|---|---|---|---|---|---|---|---|']
    for row in summary['thresholds']:
        if (row['scope'], row['threshold'], row['boost']) != ('mixed_brics_bonds', 0.6, 0.0):
            continue
        m = row['native']
        cells = [f"{m[k]['mean']:.3f}" if m[k]['mean'] is not None else 'NA'
                 for k in THRESHOLD_METRICS]
        pooled = row['native_pooled_f1']['f1']
        pooled_text = f'{pooled:.3f}' if pooled is not None else 'NA'
        lines.append(f"| {row['dataset']} | {row['model']} | {row['n_structures']} | {cells[0]} | "
                     f"{pooled_text} | {' | '.join(cells[1:])} |")
    lines += ['', '## PoCo threshold gain over same-type shuffle', '',
              '| Dataset | Boost | Threshold | Macro F1 gain | ARI gain | Fragment-count change |',
              '|---|---|---|---|---|---|']
    for row in summary['thresholds']:
        if row['scope'] != 'mixed_brics_bonds' or row['model'] != 'PoCo':
            continue
        m = row['native_minus_shuffle']
        lines.append(f"| {row['dataset']} | {row['boost']} | {row['threshold']} | {fmt(m['f1'])} | "
                     f"{fmt(m['ari'])} | {fmt(m['n_fragments'])} |")
    return '\n'.join(lines)


def main():
    out = OUTPUT_DIR
    candidates = defaultdict(list)
    for row in read_rows(out / 'PoCo_metrics.jsonl'):
        if row['variant'] == 'native' and 0 < row['all_cuts'] < row['all_bonds']:
            candidates[row['dataset']].append(row['index'])
    selected_indices = {dataset: set(indices[:EVALUATION_SIZE])
                        for dataset, indices in candidates.items()}
    metric_groups, threshold_groups = defaultdict(list), defaultdict(list)
    for model in sorted(MODELS):
        path = out / f'{model}_metrics.jsonl'
        for row in read_rows(path):
            if row['index'] in selected_indices[row['dataset']]:
                metric_groups[(row['dataset'], row['model'])].append(row)
    for model in sorted(MODELS):
        path = out / f'{model}_thresholds.jsonl'
        for row in read_rows(path):
            if row['index'] in selected_indices[row['dataset']]:
                threshold_groups[(row['dataset'], row['model'], row['boost'], row['threshold'])].append(row)
    summary = {'bootstrap_replicates': REPLICATES, 'bootstrap_seed': 20260910,
               'sample_counts': json.loads((out / 'sample_counts.json').read_text(encoding='utf-8')),
               'selected_indices': {dataset: sorted(indices) for dataset, indices in selected_indices.items()},
               'metrics': [], 'comparisons': [], 'thresholds': []}
    maps = {}
    for key, rows in sorted(metric_groups.items()):
        result, native, shuffled = grouped_stats(rows, METRICS)
        maps[key] = (native, shuffled)
        summary['metrics'].append(dict(dataset=key[0], model=key[1], **result))
    for dataset in sorted({key[0] for key in maps}):
        if (dataset, 'PoCo') not in maps or (dataset, 'MLM') not in maps:
            continue
        poco, poco_shuffle = maps[(dataset, 'PoCo')]
        mlm, mlm_shuffle = maps[(dataset, 'MLM')]
        gains = []
        for native, shuffled in ((poco, poco_shuffle), (mlm, mlm_shuffle)):
            gains.append({i: {k: value(native[i], k) - value(shuffled[i], k) for k in METRICS}
                          for i in native.keys() & shuffled.keys()})
        summary['comparisons'].append(dict(dataset=dataset,
            native=paired_stats(poco, mlm, METRICS), context_gain=paired_stats(*gains, METRICS)))
    for key, rows in sorted(threshold_groups.items()):
        dataset, model, boost, threshold = key
        result, native, shuffled = grouped_stats(rows, THRESHOLD_METRICS)
        result['native_pooled_f1'] = pooled_f1(list(native.values()))
        if shuffled:
            shuffled_counts = shuffle_means(rows, ['tp', 'fp', 'fn'])
            result['shuffle_pooled_f1'] = pooled_f1(list(shuffled_counts.values()))
            result['native_minus_shuffle_pooled_f1'] = paired_pooled_f1(native, shuffled_counts)
        summary['thresholds'].append(dict(dataset=dataset, model=model, boost=boost,
            threshold=threshold, scope='mixed_brics_bonds', **result))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'results.md').write_text(report(summary), encoding='utf-8')
    print(f'Summarized {len(metric_groups)} dataset/model groups into {out}')


if __name__ == '__main__':
    main()
