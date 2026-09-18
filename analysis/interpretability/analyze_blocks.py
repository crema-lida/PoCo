"""Compare BRICS block membership using within-stratum cosine ranks."""
from collections import defaultdict
import json

import numpy as np
from scipy.stats import rankdata

from config import OUTPUT_DIR, MODELS
from run_fragments import describe, log


def block_auc(gram, matched_pairs):
    aucs = []
    for pairs, inside in matched_pairs:
        values = gram[pairs[:, 0], pairs[:, 1]]
        n_inside = int(inside.sum())
        n_outside = len(inside) - n_inside
        # Mann-Whitney form of ROC-AUC; average ranks give ties half credit.
        ranks = rankdata(values, method='average')
        aucs.append((ranks[inside].sum() - n_inside * (n_inside + 1) / 2)
                    / (n_inside * n_outside))
    return float(np.mean(aucs)) if aucs else None


def paired_bootstrap(differences):
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return dict(n=0, mean_difference=None, ci95=None)
    rng = np.random.default_rng(42)
    means = values[rng.integers(len(values), size=(1000, len(values)))].mean(axis=1)
    return dict(n=len(values), mean_difference=float(values.mean()),
                ci95=np.quantile(means, [0.025, 0.975]).tolist())


def summarize(rows, samples, models):
    datasets = list(dict.fromkeys(sample['dataset'] for sample in samples))
    summaries, comparisons = [], []
    for dataset in datasets:
        n_sampled = sum(s['dataset'] == dataset for s in samples)
        selected = [r for r in rows if r['dataset'] == dataset]
        for model in models:
            for variant in ('native', 'same_type_shuffle'):
                valid = [r['matched_block_auc'] for r in selected
                         if r['model'] == model and r['variant'] == variant
                         and r['matched_block_auc'] is not None]
                summaries.append(dict(dataset=dataset, model=model, variant=variant,
                                      n_sampled=n_sampled, n_covered=len(valid),
                                      matched_block_auc=float(np.mean(valid)) if valid else None))
        lookup = {(r['model'], r['variant'], r['index']): r['matched_block_auc']
                  for r in selected if r['matched_block_auc'] is not None}
        contrasts = [(model, 'native', model, 'same_type_shuffle') for model in models]
        if 'PoCo' in models and 'MLM' in models:
            contrasts.insert(0, ('PoCo', 'native', 'MLM', 'native'))
        for left_model, left_variant, right_model, right_variant in contrasts:
            differences = [left - lookup[(right_model, right_variant, index)]
                           for (model, variant, index), left in lookup.items()
                           if (model, variant) == (left_model, left_variant)
                           and (right_model, right_variant, index) in lookup]
            comparisons.append(dict(dataset=dataset,
                                    left=f'{left_model}/{left_variant}',
                                    right=f'{right_model}/{right_variant}',
                                    **paired_bootstrap(differences)))
    return dict(bootstrap_replicates=1000, bootstrap_seed=42,
                summaries=summaries, comparisons=comparisons)


def main():
    out = OUTPUT_DIR
    samples = json.loads((out / 'samples.json').read_text(encoding='utf-8'))
    descriptions = [describe(sample) for sample in samples]
    rows = []
    for model in MODELS:
        with np.load(out / f'{model}_grams.npz') as grams:
            for index, (sample, description) in enumerate(zip(samples, descriptions)):
                gram = grams[str(index)]
                strata = description['matched_pairs']
                meta = dict(index=index, dataset=sample['dataset'], row_id=sample['row_id'],
                            model=model, matched_block_strata=len(strata))
                rows.append(meta | dict(variant='native',
                                        matched_block_auc=block_auc(gram, strata)))
                rng = np.random.default_rng(2026 + index)
                groups = defaultdict(list)
                for atom, key in enumerate(description['keys']):
                    groups[key].append(atom)
                shuffled = []
                for _ in range(3):
                    order = np.arange(len(gram))
                    for group in groups.values():
                        order[group] = rng.permutation(group)
                    shuffled.append(block_auc(gram[np.ix_(order, order)], strata))
                rows.append(meta | dict(variant='same_type_shuffle',
                                        matched_block_auc=float(np.mean(shuffled)) if strata else None))
        log(f'Block AUC: {model}, {len(samples)} structures')
    with (out / 'block_auc.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    summary = summarize(rows, samples, MODELS)
    (out / 'block_auc_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    for result in summary['comparisons']:
        log(json.dumps(result))


if __name__ == '__main__':
    main()
