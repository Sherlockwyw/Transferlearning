"""
Smoke-test the released package:

    1. Both released weights load into KappaMLP with strict=True;
    2. the fine-tuned model's first two Linear layers are byte-identical
       to the base model (the transfer-learning signature);
    3. the inlined MinMax scalers reproduce the original sklearn pickles
       (if the original pkl files are reachable - auto-detected via --orig-root);
    4. prediction round-trips return physically sensible values.

Run:
    python verify.py [--orig-root "path/to/original_thesis_dir"]
"""

import argparse
import os

import numpy as np
import torch

from kappa_model import load_model, BASE_WEIGHTS, FINETUNED_WEIGHTS
from inference import predict
from scalers import X_MIN, X_MAX, Y_MIN, Y_MAX

N_PARAMS_EXPECTED = 11393
FROZEN_KEYS = ["model.0.weight", "model.0.bias", "model.2.weight", "model.2.bias"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-root", default="", help="original thesis dir (optional)")
    args = p.parse_args()

    ok = True

    # 1) strict load of both released weights
    base = load_model(BASE_WEIGHTS)
    ft = load_model(FINETUNED_WEIGHTS)
    for name, m in [("base", base), ("finetuned", ft)]:
        n = sum(p_.numel() for p_ in m.parameters())
        good = n == N_PARAMS_EXPECTED
        print(f"[1] {name:<9} strict load OK - {n} parameters "
              f"(expected {N_PARAMS_EXPECTED}: {'OK' if good else 'MISMATCH'})")
        ok &= good

    # 2) transfer-learning signature: frozen layers identical to base
    identical = all(torch.equal(base.state_dict()[k], ft.state_dict()[k])
                    for k in FROZEN_KEYS)
    print(f"[2] Fine-tuned model keeps base weights in the first two Linear "
          f"layers: {identical}")
    ok &= identical

    # 3) optional: compare inlined scalers with original sklearn pickles
    if args.orig_root:
        import joblib
        xs = joblib.load(os.path.join(args.orig_root, "scalers", "global_x_scaler.pkl"))
        ys = joblib.load(os.path.join(args.orig_root, "scalers", "global_y_scaler.pkl"))
        same = (np.allclose(xs.data_min_, X_MIN) and np.allclose(xs.data_max_, X_MAX)
                and np.allclose(ys.data_min_, Y_MIN) and np.allclose(ys.data_max_, Y_MAX))
        print(f"[3] Scaler constants match original pickles: {same}")
        ok &= same
    else:
        print("[3] Skipped (no --orig-root given)")

    # 4) smoke prediction with both models
    feats = np.array([[5.5, 0.62, 300.0], [5.0, 0.60, 1000.0]])
    preds_base = predict(base, feats).flatten()
    preds_ft = predict(ft, feats).flatten()
    print(f"[4] Predictions for {feats.tolist()}:")
    for f, kb, kf in zip(feats, preds_base, preds_ft):
        print(f"    Z={f[0]:.2f}  Ia={f[1]:.2f}  T={f[2]:.0f} K  ->  "
              f"base {kb:.4f} / finetuned {kf:.4f} W/(m*K)")
    in_range = all(0.0 < k < 10.0 for k in list(preds_base) + list(preds_ft))
    print(f"    physically plausible: {in_range}")
    ok &= in_range

    print("\nVERIFY " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
