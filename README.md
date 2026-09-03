# PoCo

## Overview

PoCo is a contrastive learning framework for polymer representation learning, with applications to property prediction and interpretability. This repository contains the source code for PoCo. For more details, please see our paper: [Contrastive representation learning for polymer informatics](https://doi.org/10.26434/chemrxiv.15003645/v1).

<p align="center">
  <img width="80%" alt="poco-overview" src="https://github.com/user-attachments/assets/fb984d8a-4861-4d4d-bb6d-c382288aa9ce" />
</p>

## Environment setup

Reproduce this work with [uv](https://docs.astral.sh/uv/), which automatically creates a `.venv` virtual environment:

```bash
uv sync --frozen
```

To use a different PyTorch source, change `url` under `[[tool.uv.index]]` in `pyproject.toml`, then run:

```bash
uv sync
```

## Pretraining

1. PoCo was pretrained on ~1M polymer SMILES from the [PI1M dataset](https://github.com/RUIMINMA1996/PI1M). The raw dataset contains invalid SMILES; please clean it with `src/pretrain/preprocess.py` before pretraining.
2. Use `src/pretrain/train_tokenizer.py` to train a tokenizer.
3. Run the pretraining script `./pretrain.sh`.

We provide pretrained PoCo weights at https://huggingface.co/CremaX/PoCo.

## Transfer learning

The training entry point for downstream tasks is `src/finetune/finetune.py`.

Use `src/finetune/run_benchmark.py` to reproduce the benchmark results. The adaptation code for the baseline models largely follows the original implementations. When running the benchmarks, additional dependencies or minor code modifications may be required to accommodate different local environments.

## Benchmark results

| Model | Params (M)<sup>a</sup> | d<sub>rep</sub><sup>b</sup> | [Khazana-MTL](https://doi.org/10.1016/j.patter.2021.100238) | [PolyOmics](https://doi.org/10.48550/arXiv.2511.11626) | [RadonPy](https://doi.org/10.1038/s41524-022-00906-4) | [OPC](https://doi.org/10.48550/arXiv.2512.08896) | [Gas](https://doi.org/10.1038/s41524-024-01373-9) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [polyBERT](https://doi.org/10.1038/s41467-023-39868-6) | 25 | 600 | 0.794 &plusmn; 0.019 | 0.790 &plusmn; 0.002 | 0.817 &plusmn; 0.028 | 0.787 &plusmn; 0.020 | 0.750 &plusmn; 0.029 |
| [TransPolymer](https://doi.org/10.1038/s41524-023-01016-5) | 82 | 768 | 0.792 &plusmn; 0.017 | 0.795 &plusmn; 0.002 | 0.809 &plusmn; 0.026 | 0.786 &plusmn; 0.027 | 0.762 &plusmn; 0.031 |
| [PolyCL](https://doi.org/10.1039/d4dd00236a) | 25 | 600 | 0.794 &plusmn; 0.023 | 0.792 &plusmn; 0.001 | 0.813 &plusmn; 0.025 | 0.766 &plusmn; 0.025 | 0.765 &plusmn; 0.023 |
| [MMPolymer](https://doi.org/10.1145/3627673.3679684) | 129 | 1280 | 0.799 &plusmn; 0.020 | 0.795 &plusmn; 0.002 | 0.815 &plusmn; 0.026 | 0.779 &plusmn; 0.026 | 0.761 &plusmn; 0.022 |
| [PerioGT](https://doi.org/10.1038/s43588-025-00903-9) | 91 | 2304 | 0.810 &plusmn; 0.025 | <ins>0.805 &plusmn; 0.002</ins> | 0.827 &plusmn; 0.021 | <ins>0.792 &plusmn; 0.022</ins> | <ins>0.779 &plusmn; 0.022</ins> |
| PoCo | 10 | 512 | <ins>0.815 &plusmn; 0.016</ins> | 0.800 &plusmn; 0.001 | <ins>0.830 &plusmn; 0.026</ins> | <ins>0.792 &plusmn; 0.023</ins> | **0.786 &plusmn; 0.025** |
| PoCo<sub>concat</sub> | 10 | 1536 | **0.822 &plusmn; 0.015** | **0.806 &plusmn; 0.001** | **0.842 &plusmn; 0.024** | **0.803 &plusmn; 0.022** | **0.786 &plusmn; 0.025** |

<sup>a</sup> Number of parameters in millions.  
<sup>b</sup> Representation dimension.

## Citation

If you use PoCo in your research, please cite our paper:

```bibtex
@article{wang2026poco,
  title = {Contrastive representation learning for polymer informatics},
  author = {Wang, Lida and Long, Donghui},
  journal = {ChemRxiv},
  year = {2026},
  doi = {10.26434/chemrxiv.15003645/v1}
}
```
