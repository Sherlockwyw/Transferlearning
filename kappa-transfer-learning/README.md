# KappaTransferLearning

Machine-learning prediction of the **thermal conductivity κ [W/(m·K)] of
Ta-Nb-O4 compounds** from three physical features, via **transfer learning**
(pretrain on a large dataset → freeze the first layers → fine-tune on a small
dataset).

**Both trained models are open-sourced** - you can use them directly for
prediction, no training required:

| Model | Weights | Description |
|-------|---------|-------------|
| **PreTrainModel** | `PreTrainModel/kappa_base.pth` | Base model pretrained on an 872-row dataset |
| **TransferLearningModel** | `TransferLearningModel/kappa_finetuned.pth` | Transfer-learning model fine-tuned on a 238-row dataset (recommended for prediction) |

## Task definition

- **Input** : 3 features, raw physical units (scaling handled internally):

  | # | Feature | Meaning | Range in pretraining data |
  |---|---------|---------|---------------------------|
  | 1 | `Z-eff-C` | effective core-electron feature (dimensionless) | 4.823 – 6.237 |
  | 2 | `Ia-b` | ionic radius difference feature (dimensionless) | 0.556 – 0.736 |
  | 3 | `T` | temperature (K) | 25 – 1500 |

- **Output** : thermal conductivity κ, range 0.905 – 3.962 W/(m·K) in
  pretraining data.

- **Architecture** (KappaMLP, 11,393 parameters, shared by both models):

  ```
  Linear(3→128) → ReLU → Linear(128→64) → ReLU → Linear(64→32) → ReLU
  → Linear(32→16) → ReLU → Linear(16→1)
  ```

- **Transfer-learning scheme**: the fine-tuned model was obtained by loading
  the base weights, **freezing the first two Linear layers** (they remain
  byte-identical to the base model), and fine-tuning the remaining layers on a
  238-row downstream dataset.

- **Preprocessing** : MinMax scaling of X and y with constants fitted on the
  872-row pretraining set (inlined in `scalers.py`, no pickle/sklearn
  dependency).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 1.13, numpy, pandas (openpyxl only for xlsx).

## Quick start - predict thermal conductivity

### Option A: one-liner CLI

```bash
# Transfer-learning model (recommended)
python inference.py --model finetuned --z 5.5 --ia 0.62 --t 300
# Pretrained base model
python inference.py --model base --z 5.5 --ia 0.62 --t 300
```

### Option B: Python API (3 lines)

```python
from inference import predict_kappa

# uses the fine-tuned TransferLearningModel by default
kappa = predict_kappa([[5.5, 0.62, 300.0]])
print(kappa)        # -> array of kappa in W/(m*K)

# or explicitly with either released model
kappa_b = predict_kappa([[5.5, 0.62, 300.0]], model="base")
kappa_f = predict_kappa([[5.5, 0.62, 300.0]], model="finetuned")

# multiple points at once
kappas = predict_kappa([[5.5, 0.62, 300.0],
                        [5.0, 0.60, 500.0],
                        [6.0, 0.73, 1300.0]])
```

### Option C: full demo script

```bash
python examples/predict_kappa.py
```

This runs three demos: single-point prediction, multi-point comparison of both
models, and batch prediction from `examples/example_input.csv`.

## Reproduce the training of the PreTrainModel

Prepare a CSV/XLSX with 4 columns (header optional) - `Z-eff-C, Ia-b, T(K), κ`:

```bash
python train_base_model.py --data examples/pretrain.csv --out output/my_base_model.pth
```

Recipe: Adam (lr 1e-3, weight decay 1e-4), MSE loss on the scaled target,
batch 32, ≤ 500 epochs, early stopping (patience 50) on training loss,
MinMax scalers fitted on the pretraining data. A sidecar `_scalers.npz` with
the fitted constants is saved next to the weights.

## Reproduce the TransferLearningModel (fine-tuning recipe)

```bash
python finetune_model.py --data data/my_small_dataset.csv \
                         --base-weights PreTrainModel/kappa_base.pth \
                         --out output/my_finetuned_model.pth
```

Pipeline (verified against the original training code and the released
weights):

1. load the open-sourced base weights;
2. **freeze the first two Linear layers** (default `--freeze 2`; the released
   fine-tuned model's frozen layers are byte-identical to the base);
3. fine-tune the remaining layers on your small dataset (Adam lr 1e-3, wd
   1e-4, batch 64, MSE on scaled target, ≤ 500 epochs, early stop patience 50);
4. keep the best checkpoint by monitored loss.

`--val-split`, `--refit-scalers` and other options: see
`python finetune_model.py --help`.

> Scaling note: the frozen layers expect the pretraining MinMax scale, so by
> default your fine-tuning data are scaled with the released constants. If
> your data live far outside the pretraining range, use `--refit-scalers`
> (constants then saved to a sidecar `.npz`).

## Performance

Metrics of the released weights on the original data splits:

| Model | Dataset | R² | RMSE [W/(m·K)] |
|-------|---------|----|----------------|
| PreTrainModel (this repo) | 872-row pretraining set | 0.8305 | 0.2149 |
| PreTrainModel (this repo) | downstream small-data test (59 rows, before fine-tuning) | 0.5330 | 0.3769 |
| TransferLearningModel (this repo) | 238-row fine-tuning set | 0.8854 | 0.1821 |
| TransferLearningModel (this repo) | same 59-row test set | 0.8409 | 0.2200 |
| base + freeze-2 fine-tune (original run reported in the paper) | same test set | **0.9017** | **0.1729** |
| base + freeze-2 fine-tune (10-fold CV over the fine-tuning set) | 10-fold mean | 0.895 | 0.161 |

> Note: the 0.9017 test score comes from the original training run reported in
> the paper; the released `kappa_finetuned.pth` is a later re-run from a
> re-trained base (0.8409 on the same test set). Re-running the fine-tuning
> recipe reproduces comparable performance (10-fold CV mean R² 0.895 / RMSE
> 0.161).

## Verify the release

```bash
python verify.py
# optionally, if you have the original thesis directory:
python verify.py --orig-root "path/to/original_thesis_dir"
```

Checks: both weights strict-load, the fine-tuned model keeps the base weights
in its first two Linear layers (transfer-learning signature), and predictions
are physically plausible.

## Repository structure

```
kappa-transfer-learning/
├── PreTrainModel/
│   └── kappa_base.pth             # ★ released pretrained base weights
├── TransferLearningModel/
│   └── kappa_finetuned.pth        # ★ released fine-tuned weights
├── examples/
│   ├── predict_kappa.py           # demo: direct prediction with the models
│   ├── example_input.csv          # example feature file for batch prediction
│   └── pretrain_example.csv       # example 4-column training file
├── kappa_model.py                 # architecture + freeze utility + loaders
├── scalers.py                     # MinMax constants (no pickle/sklearn dependency)
├── data_util.py                   # shared IO helpers
├── inference.py                   # quick prediction CLI + Python API
├── train_base_model.py            # reproduce base-model training
├── finetune_model.py              # transfer-learning fine-tune recipe
├── verify.py                      # release smoke test
├── requirements.txt
├── LICENSE                        # MIT
└── README.md
```

## Data & citation

The pretraining and fine-tuning datasets (872 / 238 / 59 rows) are not
included; see the associated paper. If you use the released weights or the
recipe, please cite the associated paper.

## License

MIT License - see [LICENSE](LICENSE).
