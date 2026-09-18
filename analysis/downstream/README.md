# Downstream model comparison

`bayesian_comparison.py` compares PoCo with baseline models using paired outer-fold R² scores and a Bayesian correlated t-test. It reports the mean difference (`PoCo − baseline`), its 95% credible interval, and posterior probabilities that PoCo's mean R² is higher or lower.

## Usage

Edit `RESULT_DIRS`, `OUTPUT_DIR`, `DATASETS`, and `BASELINES` at the top of the script. By default, inputs are read from `checkpoints/downstream/<dataset>/results/<model>/fold_metrics.csv`. Models must use the same folds.

Run from this directory:

```bash
python bayesian_comparison.py
```

## Outputs (`output/`)

- `dataset_bayesian_pairs.json`: overall comparisons for each dataset.
- `task_bayesian_pairs.json`: comparisons for each property.
- `fold_metrics.json`: fold scores used in the analysis.
