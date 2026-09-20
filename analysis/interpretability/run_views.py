"""Atom-aligned fragment stability under alternative SMILES traversals."""
from collections import defaultdict
import json
import re

from config import OUTPUT_DIR, MODELS, EVALUATION_SIZE, CPU_THREADS
from run_fragments import components, describe, encode, log
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import torch
from transformers import AutoTokenizer


def main():
    torch.set_num_threads(CPU_THREADS)
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    samples = json.loads((out / 'samples.json').read_text(encoding='utf-8'))
    chosen, descriptions = [], {}
    counts = defaultdict(lambda: dict(rho=0, jaccard=0))
    selected_rho, selected_jaccard = set(), set()
    for i, sample in enumerate(samples):
        dataset = sample['dataset']
        count = counts[dataset]
        if min(count.values()) == EVALUATION_SIZE:
            continue
        d = describe(sample)
        eligible = d['eligible'] & d['single'] & d['nonring']
        strata = defaultdict(list)
        for edge_id in np.flatnonzero(eligible):
            a, b = d['edges'][edge_id]
            strata[tuple(sorted((d['keys'][a], d['keys'][b])))].append(edge_id)
        use_rho = count['rho'] < EVALUATION_SIZE and any(len(ids) >= 4 for ids in strata.values())
        use_jaccard = count['jaccard'] < EVALUATION_SIZE and eligible.sum() >= 4
        if use_rho:
            selected_rho.add(i)
            count['rho'] += 1
        if use_jaccard:
            selected_jaccard.add(i)
            count['jaccard'] += 1
        if use_rho or use_jaccard:
            chosen.append(i)
            d['type_strata'] = strata
            descriptions[i] = d
    log(f'Selected structures: {dict(counts)}')
    tokenizer = AutoTokenizer.from_pretrained(MODELS['PoCo']['path'], local_files_only=True)
    views, excluded_length = [], 0
    for i in chosen:
        sample = samples[i]
        mol = Chem.MolFromSmiles(sample['smiles'])
        rng = np.random.default_rng(917 + i)
        for repeat in range(3):
            order = rng.permutation(mol.GetNumAtoms())
            permuted = Chem.RenumberAtoms(mol, order.tolist())
            smi = Chem.MolToSmiles(permuted, canonical=False, isomericSmiles=True)
            mapping = order[json.loads(permuted.GetProp('_smilesAtomOutputOrder'))]
            smi = re.sub(r'(?<!\[)\*(?!\])', '[*]', smi)
            if len(tokenizer(smi)['input_ids']) > 384:
                excluded_length += 1
                continue
            views.append(sample | dict(smiles=smi, index=i, repeat=repeat,
                                       atom_order=mapping.tolist(), changed=smi != sample['smiles']))
    (out / 'view_samples.json').write_text(json.dumps(views, indent=2), encoding='utf-8')
    (out / 'view_counts.json').write_text(json.dumps(dict(base_counts=counts, views=len(views),
        selected_indices=dict(matched_type_bond_rho=sorted(selected_rho),
                              fixed_quarter_cut_jaccard=sorted(selected_jaccard)),
        excluded_length=excluded_length), indent=2), encoding='utf-8')
    rows = []
    for name, config in MODELS.items():
        saved = out / f'{name}_view_grams.npz'
        cached = {}
        if saved.exists():
            with np.load(saved) as cache:
                cached = {key: cache[key] for key in cache.files}
        view_keys = [f"{view['index']}-{view['repeat']}" for view in views]
        missing = [(key, view) for key, view in zip(view_keys, views) if key not in cached]
        log(f'{name}: encoding {len(missing)} new views; reusing {len(views) - len(missing)}')
        if missing:
            encoded = encode([view for _, view in missing], name, config['path'])
            cached.update(zip((key for key, _ in missing), encoded))
            np.savez_compressed(saved, **cached)
        grams = [cached[key] for key in view_keys]
        with np.load(out / f'{name}_grams.npz') as cache:
            base_grams = {i: cache[str(i)] for i in chosen}
        for view, gram in zip(views, grams):
            i = view['index']
            d = descriptions[i]
            align = np.argsort(view['atom_order'])
            gram = gram[np.ix_(align, align)]
            base = base_grams[i]
            edges = d['edges']
            first = base[edges[:, 0], edges[:, 1]]
            second = gram[edges[:, 0], edges[:, 1]]
            eligible = d['eligible'] & d['single'] & d['nonring']
            row = dict(index=i, dataset=view['dataset'], model=name, repeat=view['repeat'],
                       changed=view['changed'], n_bonds=int(eligible.sum()))
            if eligible.sum() >= 4:
                a, b = first[eligible], second[eligible]
                row['single_nonring_rho'] = float(spearmanr(a, b).statistic)
                k = max(1, round(len(a) * 0.25))
                cut1, cut2 = set(np.argsort(a)[:k]), set(np.argsort(b)[:k])
                if i in selected_jaccard:
                    row['fixed_quarter_cut_jaccard'] = len(cut1 & cut2) / len(cut1 | cut2)
            same_type = np.array([d['keys'][a] == d['keys'][b] for a, b in edges])
            subset = eligible & same_type
            if subset.sum() >= 4:
                row['same_type_bond_rho'] = float(spearmanr(first[subset], second[subset]).statistic)
            correlations = [float(spearmanr(first[ids], second[ids]).statistic)
                            for ids in d['type_strata'].values() if len(ids) >= 4]
            correlations = [r for r in correlations if np.isfinite(r)]
            if correlations and i in selected_rho:
                row['matched_type_bond_rho'] = float(np.mean(correlations))
                row['matched_type_strata'] = len(correlations)
            for threshold in (0.5, 0.6, 0.7):
                lab1 = components(len(base), edges, np.maximum(first, 0) + (~d['single']) * .6 >= threshold)
                lab2 = components(len(base), edges, np.maximum(second, 0) + (~d['single']) * .6 >= threshold)
                row[f'ari_{threshold}'] = float(adjusted_rand_score(lab1[d['heavy']], lab2[d['heavy']]))
                row[f'n_fragments_{threshold}'] = len(np.unique(lab1[d['heavy']]))
            rows.append(row)
        log(f'{name}: completed aligned-view analysis')
    frame = pd.DataFrame(rows)
    frame.to_json(out / 'view_metrics.jsonl', orient='records', lines=True)
    # Each polymer gets equal weight, irrespective of the number of eligible views.
    per_polymer = frame.query('changed').groupby(['dataset', 'model', 'index']).mean(numeric_only=True)
    summary = per_polymer.groupby(['dataset', 'model']).mean(numeric_only=True).reset_index()
    summary.to_json(out / 'view_summary.json', orient='records', indent=2)
    log(summary[['dataset', 'model', 'single_nonring_rho', 'fixed_quarter_cut_jaccard', 'ari_0.6']].to_string(index=False))


if __name__ == '__main__':
    main()
