"""
Example: predict thermal conductivity with the released models.

Three ways to use the TransferLearningModel (or PreTrainModel) directly:

    1. Single / multi-point prediction (Python API)
    2. Batch prediction from a CSV file
    3. Command-line one-liner (see inference.py)

Run from the repo root:
    python examples/predict_kappa.py
"""

import os
import sys

import numpy as np

# allow running from the repo root without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kappa_model import load_model, BASE_WEIGHTS, FINETUNED_WEIGHTS  # noqa: E402
from inference import predict  # noqa: E402


def demo_single_point():
    """1. Predict one composition at one temperature."""
    print("=" * 60)
    print("1) Single-point prediction with the fine-tuned model")
    print("=" * 60)
    model = load_model(FINETUNED_WEIGHTS)          # TransferLearningModel
    features = [[5.5, 0.62, 300.0]]                # [Z-eff-C, Ia-b, T(K)]
    kappa = predict(model, features).item()
    print(f"Input : Z-eff-C = 5.5, Ia-b = 0.62, T = 300 K")
    print(f"Output: kappa = {kappa:.4f} W/(m*K)\n")


def demo_multi_point():
    """2. Predict several conditions at once - compare both models."""
    print("=" * 60)
    print("2) Multi-point prediction - fine-tuned vs base model")
    print("=" * 60)
    base_model = load_model(BASE_WEIGHTS)          # PreTrainModel
    ft_model = load_model(FINETUNED_WEIGHTS)       # TransferLearningModel

    features = [
        [5.5, 0.62, 300.0],    # Z-eff-C, Ia-b, T(K)
        [5.0, 0.60, 500.0],
        [5.8, 0.70, 900.0],
        [6.0, 0.73, 1300.0],
    ]
    kappa_base = predict(base_model, features).flatten()
    kappa_ft = predict(ft_model, features).flatten()

    print(f"{'Z-eff-C':>8} {'Ia-b':>6} {'T [K]':>6} {'kappa_base':>12} {'kappa_finetuned':>16}")
    for f, kb, kf in zip(features, kappa_base, kappa_ft):
        print(f"{f[0]:>8.2f} {f[1]:>6.2f} {f[2]:>6.0f} {kb:>12.4f} {kf:>16.4f}")
    print("The fine-tuned model is the recommended default for prediction.\n")


def demo_from_csv():
    """3. Batch prediction from a CSV file of features."""
    print("=" * 60)
    print("3) Batch prediction from examples/example_input.csv")
    print("=" * 60)
    import pandas as pd

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "example_input.csv")
    df = pd.read_csv(csv_path)
    model = load_model(FINETUNED_WEIGHTS)
    preds = predict(model, df[["Z-eff-C", "Ia-b", "T"]].values).flatten()

    out = df.copy()
    out["predicted_kappa [W/(m*K)]"] = np.round(preds, 4)
    print(out.to_string(index=False))

    out_path = os.path.splitext(csv_path)[0] + "_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    demo_single_point()
    demo_multi_point()
    demo_from_csv()
