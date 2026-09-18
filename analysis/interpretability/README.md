# Interpretability analysis

Run notebooks and scripts from this directory. Configure the SMILES and BRICS analyses in `config.py`; results are saved to `output/`.

## SME fragment attribution

Substructure masking explanation (SME) measures how masking a fragment changes the ensemble model's prediction.

- `sme_mol.ipynb`: visualize fragment contributions for an individual polymer.
- `sme_dataset.ipynb`: summarize fragment contributions across polymers with volcano and distribution plots.

Set `checkpoint_path`, `task_name`, and the input SMILES or dataset in the notebook.

## SMILES writing consistency

Measure whether atom similarities and fragment cuts remain consistent across different SMILES of the same polymer. Run `run_fragments.py` once to prepare the samples and cached representations, then:

```bash
python run_views.py
python summarize_views.py
python plot_summary.py
```

## BRICS fragment correspondence

Compare fragments derived from atom similarities with BRICS chemical fragments, and examine sensitivity to the similarity threshold.

```bash
python run_fragments.py
python summarize_fragments.py
python analyze_blocks.py
```
