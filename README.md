# PoCo

## Overview

PoCo is a contrastive learning framework for learning polymer representations for property prediction and interpretability analysis. This repository provides the source code for PoCo. For method details, please see: [Contrastive representation learning for polymer informatics](https://doi.org/10.26434/chemrxiv.15003645/v1).

## Environment setup

Install a PyTorch build that matches your local CUDA environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Pretraining

1. PoCo is pretrained on ~1M polymer SMILES in the [PI1M dataset](https://github.com/RUIMINMA1996/PI1M). The raw dataset contains some invalid SMILES; make sure to clean the dataset with `src/pretrain/preprocess.py`.
2. Use `src/pretrain/train_tokenizer.py` to generate a tokenizer.
3. Run the pretraining script `./pretrain.sh`.

We provide pretrained PoCo weights at https://huggingface.co/CremaX/PoCo.

## Transfer learning

The downstream training entry point is `src/finetune/finetune.py`:

```bash
python src/finetune/finetune.py
```

## Benchmark results

| Model | Params (M)<sup>a</sup> | d<sub>rep</sub><sup>b</sup> | [Khazana-MTL](https://doi.org/10.1016/j.patter.2021.100238) | [PolyOmics](https://doi.org/10.48550/arXiv.2511.11626) | [RadonPy](https://doi.org/10.1038/s41524-022-00906-4) | [OPC](https://doi.org/10.48550/arXiv.2512.08896) | [Gas](https://doi.org/10.1038/s41524-024-01373-9) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [polyBERT](https://doi.org/10.1038/s41467-023-39868-6) | 25 | 600 | 0.794 &plusmn; 0.019 | 0.790 &plusmn; 0.002 | 0.817 &plusmn; 0.028 | 0.787 &plusmn; 0.020 | 0.750 &plusmn; 0.029 |
| [TransPolymer](https://doi.org/10.1038/s41524-023-01016-5) | 82 | 768 | 0.792 &plusmn; 0.017 | 0.795 &plusmn; 0.002 | 0.809 &plusmn; 0.026 | 0.786 &plusmn; 0.027 | 0.762 &plusmn; 0.031 |
| [PolyCL](https://doi.org/10.1039/d4dd00236a) | 25 | 600 | 0.794 &plusmn; 0.023 | 0.792 &plusmn; 0.001 | 0.813 &plusmn; 0.025 | 0.766 &plusmn; 0.025 | 0.765 &plusmn; 0.023 |
| [MMPolymer](https://doi.org/10.1145/3627673.3679684) | 129 | 1280 | 0.799 &plusmn; 0.020 | 0.795 &plusmn; 0.002 | 0.815 &plusmn; 0.026 | 0.779 &plusmn; 0.026 | 0.761 &plusmn; 0.022 |
| [PerioGT](https://doi.org/10.1038/s43588-025-00903-9) | 91 | 2304 | 0.810 &plusmn; 0.025 | <u>0.805 &plusmn; 0.002</u> | 0.827 &plusmn; 0.021 | <u>0.792 &plusmn; 0.022</u> | <u>0.779 &plusmn; 0.022</u> |
| PoCo | 10 | 512 | <u>0.815 &plusmn; 0.016</u> | 0.800 &plusmn; 0.001 | <u>0.830 &plusmn; 0.026</u> | <u>0.792 &plusmn; 0.023</u> | **0.786 &plusmn; 0.025** |
| PoCo<sub>concat</sub> | 10 | 1536 | **0.822 &plusmn; 0.015** | **0.806 &plusmn; 0.001** | **0.842 &plusmn; 0.024** | **0.803 &plusmn; 0.022** | **0.786 &plusmn; 0.025** |

<sup>a</sup> Number of parameters in millions.  
<sup>b</sup> Representation dimension.

## Citation

If you use this repository, please cite:

Wang, L.; Long, D. *Contrastive representation learning for polymer informatics*. ChemRxiv, 2026. https://doi.org/10.26434/chemrxiv.15003645/v1
