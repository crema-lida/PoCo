"""Bayesian correlated t comparisons of matched downstream outer-fold R2 scores."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIRS = [ROOT / 'checkpoints/downstream']
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
DATASETS = {"MTL": "MTL_Khazana", "RadonPy": "RadonPy", "OPC": "OPC",
            "Gas": "Gas", "PolyOmics": "PolyOmics"}
BASELINES = ["polyBERT", "TransPolymer", "PolyCL", "MMPolymer", "PerioGT"]
KEYS = ["dataset", "model", "task", "repeat", "fold"]


def paired_summary(group, test_train_ratio):
    scores = group.pivot(index=["repeat", "fold"], columns="model", values="R2")
    rows = []
    for baseline in BASELINES:
        differences = (scores["PoCo"] - scores[baseline]).to_numpy()
        n = len(differences)
        mean = differences.mean()
        scale = np.sqrt((1 / n + test_train_ratio) * differences.var(ddof=1))
        low, high = t.ppf([0.025, 0.975], n - 1, loc=mean, scale=scale)
        rows.append({
            "baseline": baseline, "n_fold_pairs": n,
            "test_train_ratio": test_train_ratio, "mean_delta_R2": mean,
            "corrected_se": scale, "credible95_low": low, "credible95_high": high,
            "posterior_prob_poco_higher": t.cdf(mean / scale, n - 1),
            "posterior_prob_poco_lower": t.sf(mean / scale, n - 1),
            "fold_wins": int((differences > 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    categories = {}
    for dataset, directory in DATASETS.items():
        config = yaml.safe_load((ROOT / "datasets" / directory / "dataset.yaml").read_text(encoding="utf-8"))
        categories[dataset] = {task: category for category, tasks in config["categories"].items() for task in tasks}

    tables = []
    columns = ["task", "repeat", "fold", "n_outer_train", "n_test", "R2"]
    for root in RESULT_DIRS:
        for path in sorted(root.rglob("fold_metrics.csv")):
            dataset = next((parent.name for parent in path.parents if parent.name in DATASETS), None)
            model = path.parent.name
            if dataset not in DATASETS or model not in ["PoCo", *BASELINES]:
                continue
            table = pd.read_csv(path, usecols=columns)
            table = table[table["task"].isin(categories[dataset])].copy()
            table["dataset"], table["model"] = dataset, model
            table["category"] = table["task"].map(categories[dataset])
            table["source_path"] = str(path.resolve())
            tables.append(table)
    folds = pd.concat(tables, ignore_index=True).drop_duplicates(KEYS, keep="last").sort_values(KEYS)

    task_tables = []
    for (dataset, task), group in folds.groupby(["dataset", "task"]):
        sizes = group[group.model == "PoCo"]
        result = paired_summary(group, (sizes.n_test / sizes.n_outer_train).mean())
        result["dataset"], result["task"] = dataset, task
        task_tables.append(result)
    tasks = pd.concat(task_tables, ignore_index=True)

    # Equal task weights within categories, then equal category weights per dataset.
    category_folds = folds.groupby(["dataset", "model", "repeat", "fold", "category"])["R2"].mean()
    overall = category_folds.groupby(["dataset", "model", "repeat", "fold"]).mean().reset_index()
    dataset_tables = []
    for dataset, group in overall.groupby("dataset"):
        result = paired_summary(group, 0.25)  # Nominal outer test/train ratio for five-fold CV.
        result["dataset"] = dataset
        dataset_tables.append(result)
    datasets = pd.concat(dataset_tables, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in [("fold_metrics", folds), ("task_bayesian_pairs", tasks),
                        ("dataset_bayesian_pairs", datasets)]:
        table.to_json(OUTPUT_DIR / f"{name}.json", orient="records", indent=2, double_precision=15)
    print(f"Read {len(folds)} fold scores; wrote {len(datasets)} dataset and {len(tasks)} task comparisons to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
