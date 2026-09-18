"""Compact scientific figure from the completed traversal experiment."""
import json
from config import OUTPUT_DIR, MODELS, SOURCES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = json.loads((OUTPUT_DIR / 'view_comparison.json').read_text(encoding='utf-8'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
datasets = list(SOURCES)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
metrics = ['matched_type_bond_rho', 'fixed_quarter_cut_jaccard']
titles = ['a  Context stability within atom-type pairs',
          'b  Cut-position stability at equal cut counts']
for ax, metric, title in zip(axes, metrics, titles):
    x = np.arange(len(datasets))
    for j, (model, config) in enumerate(MODELS.items()):
        entries = [data['models'][ds][model][metric] for ds in datasets]
        means = np.array([e['mean'] for e in entries])
        low = means - np.array([e['ci95'][0] for e in entries])
        high = np.array([e['ci95'][1] for e in entries]) - means
        ax.bar(x + (j-(len(MODELS)-1)/2)*.19, means, width=.17, color=config['color'],
               yerr=[low, high], capsize=2.5, error_kw={'linewidth': .8}, label=config['label'])
    ax.set_title(title, loc='left', fontsize=10, pad=13)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E7EBED', linewidth=.6)
    ns = [data['models'][ds]['PoCo'][metric]['n'] for ds in datasets]
    ax.set_xlabel('Eligible structures: ' + ' / '.join(map(str, ns)), labelpad=9, fontsize=9)
axes[0].set_ylabel('Atom-type-stratified Spearman correlation')
axes[1].set_ylabel('Cut-position overlap (Jaccard)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
           ncol=len(MODELS), frameon=False, bbox_to_anchor=(.5, .015), fontsize=9)
fig.suptitle('PoCo preserves contextual relationships across SMILES traversals',
             fontsize=13, fontweight='bold', y=.99)
fig.subplots_adjust(left=.07, right=.985, bottom=.25, top=.81, wspace=.30)
fig.savefig(OUTPUT_DIR / 'interpretability_evidence.png', dpi=200, facecolor='white')
fig.savefig(OUTPUT_DIR / 'interpretability_evidence.svg', facecolor='white')
plt.close(fig)
