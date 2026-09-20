"""BRICS correspondence and consistency across SMILES traversals."""
import json
from config import ROOT, OUTPUT_DIR, MODELS, SOURCES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

data = json.loads((OUTPUT_DIR / 'view_comparison.json').read_text(encoding='utf-8'))
fragments = json.loads((OUTPUT_DIR / 'summary.json').read_text(encoding='utf-8'))
brics = {(row['dataset'], row['model']): row for row in fragments['thresholds']
         if (row['scope'], row['threshold'], row['boost']) == ('mixed_brics_bonds', .6, 0.)}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
datasets = list(SOURCES)
plt.style.use(ROOT / 'analysis/style.mplstyle')
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'axes.titlesize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9})
fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.3))
x = np.arange(len(datasets))
for j, (model, config) in enumerate(MODELS.items()):
    positions = x + (j-(len(MODELS)-1)/2)*.19
    entries = [brics[ds, model] for ds in datasets]
    for position, entry in zip(positions, entries):
        axes[0].plot([position, position],
                     [entry['shuffle_mean']['f1']['mean'], entry['native']['f1']['mean']],
                     color=config['color'], linewidth=1.2, alpha=.65, zorder=2)
    for variant in ['native', 'shuffle_mean']:
        stats = [entry[variant]['f1'] for entry in entries]
        means = np.array([s['mean'] for s in stats])
        low = means - np.array([s['ci95'][0] for s in stats])
        high = np.array([s['ci95'][1] for s in stats]) - means
        size = 4.2
        face = config['color'] if variant == 'native' else 'white'
        if model == 'random' and variant == 'shuffle_mean':
            size, face = 6.2, 'none'
        axes[0].errorbar(positions, means, yerr=[low, high], fmt='o',
                         markersize=size, markeredgewidth=.9,
                         markerfacecolor=face,
                         color=config['color'], elinewidth=.7, capsize=1.7, zorder=3)
axes[0].set_ylim(.3, .7)
axes[0].set_yticks([.3, .4, .5, .6, .7])
metrics = ['matched_type_bond_rho', 'fixed_quarter_cut_jaccard']
for ax, metric in zip(axes[1:], metrics):
    for j, (model, config) in enumerate(MODELS.items()):
        entries = [data['models'][ds][model][metric] for ds in datasets]
        means = np.array([e['mean'] for e in entries])
        low = means - np.array([e['ci95'][0] for e in entries])
        high = np.array([e['ci95'][1] for e in entries]) - means
        ax.bar(x + (j-(len(MODELS)-1)/2)*.19, means, width=.17, color=config['color'],
               yerr=[low, high], capsize=2.5, error_kw={'linewidth': .8}, label=config['label'])
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, .2))
titles = ['BRICS correspondence', 'Bond similarity', 'Cut positions']
for ax, title, panel in zip(axes, titles, 'abc'):
    ax.set_title(title, pad=7)
    ax.text(-.20, 1.05, panel, transform=ax.transAxes,
            fontsize=10, fontweight='bold', ha='left', va='bottom')
    ax.set_xticks(x, datasets)
    ax.set_xlim(-.5, len(datasets)-.5)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E7EBED', linewidth=.6)
axes[0].set_ylabel('Boundary F1 score')
axes[1].set_ylabel('Spearman correlation')
axes[2].set_ylabel('Jaccard index')
fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center',
           ncol=len(MODELS), frameon=False, bbox_to_anchor=(.5, .08),
           handlelength=1.6, columnspacing=2)
axes[0].legend(handles=[
    Line2D([], [], marker='o', linestyle='none', color='#555555', markersize=4.2,
           label='Original'),
    Line2D([], [], marker='o', linestyle='none', color='#555555', markersize=4.2,
           markerfacecolor='white', label='Same-type shuffle')],
    loc='upper left', frameon=False, handlelength=1.2, fontsize=8.5, labelspacing=.25,
    borderaxespad=.1)
fig.subplots_adjust(left=.065, right=.99, bottom=.27, top=.87, wspace=.40)
fig.savefig(OUTPUT_DIR / 'interpretability_evidence.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig(OUTPUT_DIR / 'interpretability_evidence.png', bbox_inches='tight', pad_inches=.03)
fig.savefig(OUTPUT_DIR / 'interpretability_evidence.svg', bbox_inches='tight', pad_inches=.03)
plt.close(fig)
