"""Polymer-level bootstrap summaries of atom-aligned SMILES traversal stability."""
import json

import numpy as np
import pandas as pd
from config import OUTPUT_DIR, MODELS

REPLICATES, SEED = 1000, 20260910
METRICS = ('single_nonring_rho', 'fixed_quarter_cut_jaccard', 'same_type_bond_rho',
           'matched_type_bond_rho', 'matched_type_strata', 'ari_0.6', 'n_fragments_0.6')


def estimate(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return dict(n=0, mean=None, ci95=None)
    samples = values[rng.integers(n, size=(REPLICATES, n))].mean(axis=1)
    return dict(n=n, mean=float(values.mean()),
                ci95=np.quantile(samples, [0.025, 0.975]).tolist())


def summarize(out):
    frame = pd.read_json(out / 'view_metrics.jsonl', lines=True)
    polymers = frame.query('changed').groupby(['dataset', 'model', 'index']).mean(numeric_only=True)
    rng = np.random.default_rng(SEED)
    result = dict(
        bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
        counts=json.loads((out / 'view_counts.json').read_text(encoding='utf-8')),
        input_rows=len(frame),
        comparisons={}, models={})
    for dataset in polymers.index.get_level_values('dataset').unique():
        result['models'][dataset], result['comparisons'][dataset] = {}, {}
        for model in MODELS:
            values = polymers.loc[dataset, model]
            stats = {metric: estimate(values[metric], rng) for metric in METRICS}
            fragments = values['n_fragments_0.6']
            stats.update(polymers_with_changed_views=len(values),
                         single_fragment_fraction=float((fragments == 1).mean()),
                         min_n_fragments=float(fragments.min()),
                         max_n_fragments=float(fragments.max()))
            result['models'][dataset][model] = stats
        for model in (name for name in MODELS if name != 'PoCo'):
            # Pandas aligns sample indices before computing each paired difference.
            differences = polymers.loc[dataset, 'PoCo'] - polymers.loc[dataset, model]
            result['comparisons'][dataset][f'PoCo_minus_{model}'] = {
                metric: estimate(differences[metric], rng) for metric in METRICS}
    return result


def main():
    out = OUTPUT_DIR
    (out / 'view_comparison.json').write_text(json.dumps(summarize(out), indent=2, allow_nan=False),
                                         encoding='utf-8')
    print(f'Summarized traversal stability into {out / "view_comparison.json"}')


if __name__ == '__main__':
    main()
