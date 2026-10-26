"""Export per-property RMSE means and standard deviations in physical units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "checkpoints/downstream"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASETS = ["MTL", "PolyOmics", "RadonPy", "OPC", "Gas"]
MODELS = ["polyBERT", "TransPolymer", "PolyCL", "MMPolymer", "PerioGT", "PoCo"]


def physical_rmse(path, dataset):
    predictions = pd.read_csv(path)
    target = predictions["target"].to_numpy()
    predicted = predictions["prediction"].to_numpy()
    # Gas files contain log10 values; other datasets are already inverse-transformed.
    if dataset == "Gas":
        target, predicted = 10.0 ** target, 10.0 ** predicted
    return np.sqrt(np.mean((predicted - target) ** 2))


def main():
    tables = []
    for dataset in DATASETS:
        for model in MODELS:
            result_dir = RESULT_DIR / dataset / "results" / model
            folds = pd.read_csv(result_dir / "fold_metrics.csv",
                                usecols=["task", "repeat", "fold", "RMSE"])
            if dataset in ["Gas", "RadonPy"]:
                tasks = folds["task"].unique()
                if dataset == "RadonPy":
                    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
                    tasks = config["dataset"]["transforms"]["log10"]
                for row in folds[folds["task"].isin(tasks)].itertuples():
                    path = result_dir / row.task / f"fold-{row.repeat}.{row.fold}.predictions.csv.gz"
                    folds.loc[row.Index, "RMSE"] = physical_rmse(path, dataset)
            tables.append(folds.assign(dataset=dataset, model=model))

    summary = pd.concat(tables).groupby(["dataset", "task", "model"], sort=False)["RMSE"].agg(
        RMSE_mean="mean", RMSE_std="std",
    ).reset_index()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "rmse_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Wrote {len(summary)} property/model summaries to {path}.")


if __name__ == "__main__":
    main()
