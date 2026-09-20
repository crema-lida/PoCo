"""BRICS correspondence and threshold sensitivity of token-derived fragments."""
from collections import defaultdict
from datetime import datetime
import gc
import json
import os
import re

from config import ROOT, OUTPUT_DIR, SOURCES, MODELS, SAMPLE_SIZE, BATCH_SIZE, CPU_THREADS

os.environ['OMP_NUM_THREADS'] = str(CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(CPU_THREADS)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

ATOM = re.compile(r'\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnops]|\*')


def log(message):
    print(datetime.now().isoformat(timespec='seconds'), message, flush=True)


def atom_positions(smiles):
    positions = []
    for match in ATOM.finditer(smiles):
        if match[0].startswith('['):
            symbol = re.search(r'[A-Z][a-z]?|[bcnops]|\*', match[0])
            positions.append((match.start() + symbol.start(), match.start() + symbol.end()))
        else:
            positions.append(match.span())
    return positions


def components(n_atoms, edges, connected):
    parent = list(range(n_atoms))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (i, j), keep in zip(edges, connected):
        if keep:
            parent[root(int(i))] = root(int(j))
    return np.array([root(i) for i in range(n_atoms)])


def describe(sample):
    mol = Chem.MolFromSmiles(sample['smiles'])
    edges = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    heavy = np.array([a.GetAtomicNum() > 0 for a in mol.GetAtoms()])
    eligible = heavy[edges].all(axis=1)
    single = np.array([b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds()])
    nonring = np.array([not b.IsInRing() for b in mol.GetBonds()])
    cut_pairs = {tuple(sorted(pair)) for pair, _ in BRICS.FindBRICSBonds(mol)}
    truth = np.array([tuple(sorted(e)) in cut_pairs for e in edges])
    ref = components(mol.GetNumAtoms(), edges, ~truth)
    keys = [(a.GetAtomicNum(), a.GetIsAromatic(), a.GetFormalCharge(), a.GetIsotope())
            for a in mol.GetAtoms()]
    strata = defaultdict(list)
    distance = Chem.GetDistanceMatrix(mol)
    for i in np.flatnonzero(heavy):
        for j in np.flatnonzero(heavy):
            if i < j and 2 <= distance[i, j] <= 4:
                strata[(tuple(sorted((keys[i], keys[j]))), int(distance[i, j]))].append((i, j))
    matched_pairs = []
    for pairs in strata.values():
        pairs = np.array(pairs)
        inside = ref[pairs[:, 0]] == ref[pairs[:, 1]]
        if inside.any() and (~inside).any():
            matched_pairs.append((pairs, inside))
    return dict(mol=mol, edges=edges, heavy=heavy, eligible=eligible,
                single=single, nonring=nonring, truth=truth, ref=ref,
                keys=keys, matched_pairs=matched_pairs)


def prepare(size, out):
    tokenizer = AutoTokenizer.from_pretrained(MODELS['PoCo']['path'], local_files_only=True)
    samples, counts = [], {}
    for dataset, path in SOURCES.items():
        data = pd.read_csv(ROOT / path)
        candidates = []
        seen = set()
        rejected = defaultdict(int)
        for row_id, raw in enumerate(data['SMILES' if dataset == 'OPC' else 'smiles']):
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                rejected['invalid_smiles'] += 1
                continue
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            smi = re.sub(r'(?<!\[)\*(?!\])', '[*]', smi)
            if smi in seen:
                continue
            seen.add(smi)
            if len(tokenizer(smi)['input_ids']) > 384:
                rejected['over_384_tokens'] += 1
                continue
            candidates.append(dict(dataset=dataset, row_id=row_id, smiles=smi))
        indices = np.random.default_rng(42).permutation(len(candidates))[:size]
        samples.extend(candidates[i] for i in indices)
        counts[dataset] = dict(raw_rows=len(data), unique=len(seen), eligible=len(candidates),
                               sampled=len(indices), excluded=dict(rejected))
    (out / 'samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    (out / 'sample_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    return samples


def encode(samples, name, path, saved=None):
    if saved is not None and saved.exists():
        with np.load(saved) as cache:
            return [cache[str(i)] for i in range(len(samples))]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    torch.manual_seed(42)
    model = (AutoModel.from_config(AutoConfig.from_pretrained(path)) if name == 'random'
             else AutoModel.from_pretrained(path, local_files_only=True))
    model.eval().cuda()
    grams = []
    with torch.inference_mode():
        for start in range(0, len(samples), BATCH_SIZE):
            chunk = samples[start:start + BATCH_SIZE]
            tokens = tokenizer([s['smiles'] for s in chunk], padding=True,
                               return_offsets_mapping=True, return_tensors='pt')
            offsets = tokens.pop('offset_mapping').numpy()
            hidden = model(**{k: v.cuda() for k, v in tokens.items()}).last_hidden_state
            for sample, offset, h in zip(chunk, offsets, hidden):
                atom_embeds = []
                for a, b in atom_positions(sample['smiles']):
                    positions = np.flatnonzero((offset[:, 0] < b) & (offset[:, 1] > a))
                    atom_embeds.append(h[positions.tolist()].mean(0))
                emb = torch.nn.functional.normalize(torch.stack(atom_embeds), dim=-1)
                grams.append((emb @ emb.T).cpu().numpy())
            if start % 256 == 0:
                log(f'{name}: encoded {min(start + BATCH_SIZE, len(samples))}/{len(samples)}')
    if saved is not None:
        np.savez_compressed(saved, **{str(i): g for i, g in enumerate(grams)})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return grams


def measures(g, d):
    edges, eligible, truth = d['edges'], d['eligible'], d['truth']
    scores = g[edges[:, 0], edges[:, 1]]
    metrics = {}
    for name, subset in [('all', eligible), ('single', eligible & d['single']),
                         ('single_nonring', eligible & d['single'] & d['nonring'])]:
        y, score = truth[subset], -scores[subset]
        metrics[f'{name}_bonds'] = int(subset.sum())
        metrics[f'{name}_cuts'] = int(y.sum())
        if y.any() and (~y).any():
            metrics[f'{name}_auc'] = float(roc_auc_score(y, score))
            metrics[f'{name}_ap'] = float(average_precision_score(y, score))
            predicted = np.argsort(-score, kind='stable')[:int(y.sum())]
            metrics[f'{name}_matched_cut_recall'] = float(y[predicted].mean())
    differences = []
    for pairs, inside in d['matched_pairs']:
        values = g[pairs[:, 0], pairs[:, 1]]
        differences.append(float(values[inside].mean() - values[~inside].mean()))
    metrics['matched_block_gap'] = float(np.mean(differences)) if differences else None
    metrics['matched_block_strata'] = len(differences)
    metrics['mean_bond_cosine'] = float(scores[eligible].mean())
    sweeps = []
    for boost in (0., 0.6):
        weights = np.minimum(np.maximum(scores, 0) + (~d['single']) * boost, 1)
        for threshold in (0.5, 0.6, 0.7):
            labels = components(len(g), edges, weights >= threshold)
            cuts = labels[edges[:, 0]] != labels[edges[:, 1]]
            y, p = truth[eligible], cuts[eligible]
            tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
            _, sizes = np.unique(labels[d['heavy']], return_counts=True)
            sweeps.append(dict(boost=boost, threshold=threshold, tp=tp, fp=fp, fn=fn,
                               precision=tp/(tp+fp) if tp+fp else None,
                               recall=tp/(tp+fn) if tp+fn else None,
                               f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                               ari=float(adjusted_rand_score(d['ref'][d['heavy']], labels[d['heavy']])),
                               n_fragments=len(sizes), mean_fragment_size=float(sizes.mean()),
                               largest_fraction=float(sizes.max()/sizes.sum())))
    return metrics, sweeps


def main():
    torch.set_num_threads(CPU_THREADS)
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    samples = (json.loads((out / 'samples.json').read_text(encoding='utf-8'))
               if (out / 'samples.json').exists() else prepare(SAMPLE_SIZE, out))
    descriptions = [describe(s) for s in samples]
    log(f'Prepared {len(samples)} structures')
    for name, config in MODELS.items():
        grams = encode(samples, name, config['path'], out / f'{name}_grams.npz')
        rows, thresholds = [], []
        for i, (sample, d, gram) in enumerate(zip(samples, descriptions, grams)):
            rng = np.random.default_rng(2026 + i)
            variants = [('native', 0, gram)]
            for repeat in range(1, 4):
                order = np.arange(len(gram))
                groups = defaultdict(list)
                for atom, key in enumerate(d['keys']):
                    groups[key].append(atom)
                for group in groups.values():
                    order[group] = rng.permutation(group)
                variants.append(('same_type_shuffle', repeat, gram[np.ix_(order, order)]))
            for variant, repeat, g in variants:
                meta = dict(index=i, dataset=sample['dataset'], row_id=sample['row_id'],
                            model=name, variant=variant, repeat=repeat)
                metrics, sweeps = measures(g, d)
                rows.append(meta | metrics)
                thresholds.extend(meta | entry for entry in sweeps)
        pd.DataFrame(rows).to_json(out / f'{name}_metrics.jsonl', orient='records', lines=True)
        pd.DataFrame(thresholds).to_json(out / f'{name}_thresholds.jsonl', orient='records', lines=True)
        native = pd.DataFrame(rows).query('variant == "native"')
        log(name + ': ' + native.groupby('dataset')[['single_nonring_auc', 'matched_block_gap']].mean().to_json())
    log(f'Results saved to {out}')


if __name__ == '__main__':
    main()
