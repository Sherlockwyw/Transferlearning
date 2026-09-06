"""
Quick inference for thermal-conductivity prediction.

Works with BOTH released models:
    --model base       -> PreTrainModel/kappa_base.pth           (pretrained)
    --model finetuned  -> TransferLearningModel/kappa_finetuned.pth (default)

Predicts thermal conductivity kappa [W/(m*K)] of Ta-Nb-O4 compounds from
3 features (raw physical units - scaling is handled internally):
    Z-eff-C : effective core-electron configuration feature (dimensionless)
    Ia-b    : ionic radius difference feature (dimensionless)
    T       : temperature (K)

CLI examples:
    python inference.py --model finetuned --z 5.5 --ia 0.62 --t 300
    python inference.py --model base --z 5.5 --ia 0.62 --t 300
    python inference.py --weights path/to/other.pth --z 5.5 --ia 0.62 --t 300
"""

import argparse

import numpy as np
import torch

from kappa_model import load_model, BASE_WEIGHTS, FINETUNED_WEIGHTS
from scalers import transform_x, inverse_transform_y

MODEL_CHOICES = {
    "base": BASE_WEIGHTS,
    "finetuned": FINETUNED_WEIGHTS,
}


def predict(model, features, device="cpu"):
    """
    features : (N, 3) array-like of [Z-eff-C, Ia-b, T(K)] in raw physical units
    returns  : (N, 1) array of predicted thermal conductivity in W/(m*K)
    """
    X_scaled = transform_x(features).astype(np.float32)
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_scaled).to(device)).cpu().numpy()
    return inverse_transform_y(preds_scaled)


def predict_kappa(features, model="finetuned", weights=None, device="cpu"):
    """
    One-call convenience API.

    Parameters
    ----------
    features : (N, 3) array-like of [Z-eff-C, Ia-b, T(K)] in raw physical units
    model    : "finetuned" (default) or "base"
    weights  : optional path to another .pth; overrides `model`

    Returns
    -------
    (N,) numpy array of predicted kappa [W/(m*K)]
    """
    if weights is not None:
        path = weights
    elif model in MODEL_CHOICES:
        path = MODEL_CHOICES[model]
    else:
        raise ValueError(f"model must be 'base' or 'finetuned', got {model!r}")
    m = load_model(path, device=device)
    return predict(m, features, device=device).flatten()


def main():
    parser = argparse.ArgumentParser(description="kappa quick inference")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default="finetuned",
                        help="'base' = pretrained model, 'finetuned' = transfer-learning model (default)")
    parser.add_argument("--weights", default=None,
                        help="path to a .pth weights file (overrides --model)")
    parser.add_argument("--z", type=float, required=True, help="Z-eff-C feature")
    parser.add_argument("--ia", type=float, required=True, help="Ia-b feature")
    parser.add_argument("--t", type=float, required=True, help="temperature in K")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    path = args.weights or MODEL_CHOICES[args.model]
    m = load_model(path, device=args.device)
    kappa = predict(m, [[args.z, args.ia, args.t]], device=args.device)
    print(f"Model : {path}")
    print(f"Input : Z-eff-C = {args.z}  Ia-b = {args.ia}  T = {args.t} K")
    print(f"Predicted thermal conductivity: {kappa.item():.4f} W/(m*K)")


if __name__ == "__main__":
    main()
