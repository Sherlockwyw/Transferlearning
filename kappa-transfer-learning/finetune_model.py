"""
Fine-tune the released base model (PreTrainModel/kappa_base.pth) on your own
small dataset via transfer learning - the recipe that produced the released
fine-tuned model (TransferLearningModel/kappa_finetuned.pth).

Pipeline (verified against the original training code and the released
weights):

    1. Load the pretrained base weights.
    2. Freeze the first `--freeze` Linear layers (default 2). The frozen
       layers keep the pretrained representations; only the remaining layers
       are updated. The released fine-tuned model's first two Linear layers
       are byte-identical to the base model (MD5 verified).
    3. Fine-tune on your dataset: Adam (lr 1e-3, weight_decay 1e-4),
       MSE loss on MinMax-scaled target, batch 64, up to 500 epochs,
       early stopping (patience 50), keep the best checkpoint.
    4. (Optional) held-out validation split for monitoring.

IMPORTANT - scalers:
    The frozen first layer expects inputs in the *pretraining* MinMax scale.
    Therefore the fine-tuning data are scaled with the SAME scalers as the
    base model (constants inlined in scalers.py) by default. Only use
    --refit-scalers if your data live in a very different range AND you
    accept that the frozen-layer transfer assumption is weakened. In that
    case the refit constants are saved to a sidecar .npz file so inference
    stays consistent.

Data file format (.csv or .xlsx, 4 columns, header optional):
    Z-eff-C, Ia-b, T(K), thermal conductivity [W/(m*K)]

Run:
    python finetune_model.py --data data/my_small_dataset.csv \
                             --base-weights PreTrainModel/kappa_base.pth \
                             --out output/my_finetuned_model.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kappa_model import KappaMLP, freeze_layers, BASE_WEIGHTS
from data_util import load_tabular, r2_rmse
import scalers


def scale_with_constants(data):
    """Scale features/target with the released pretraining MinMax constants."""
    X_scaled = scalers.transform_x(data[:, :3])
    y_scaled = scalers.transform_y(data[:, 3:4])
    return X_scaled, y_scaled, scalers.inverse_transform_y


def scale_with_refit(data):
    """Scale with MinMax fitted on the fine-tuning data; return inverse too."""
    X, y = data[:, :3], data[:, 3:4]
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)
    inv_y = lambda p: p * (y_max - y_min) + y_min  # noqa: E731
    return (X - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min), inv_y


def build_model(base_weights_path, freeze, device):
    """Load base weights, freeze the first `freeze` Linear layers."""
    model = KappaMLP()
    state = torch.load(base_weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    freeze_layers(model, n_frozen=freeze)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded base weights from {base_weights_path}")
    print(f"Frozen {freeze} Linear layers -> trainable {n_train:,} / {n_total:,} parameters")
    return model


def finetune(args):
    data = load_tabular(args.data)
    if args.val_split > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(data))
        n_val = max(1, int(round(args.val_split * len(data))))
        val_data = data[idx[:n_val]]
        data = data[idx[n_val:]]
    else:
        val_data = None

    scale = scale_with_refit if args.refit_scalers else scale_with_constants
    X, y, inv_y = scale(data)
    Xv, yv = (None, None)
    if val_data is not None:
        Xv, yv, _ = scale(val_data)
    print(f"Fine-tuning on {len(data)} rows"
          + (f" (validation: {len(val_data)} rows)" if val_data is not None else ""))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args.base_weights, args.freeze, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=args.batch_size, shuffle=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best, best_epoch, counter, history = float("inf"), 0, 0, []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)

        monitor = avg_loss
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                monitor = criterion(
                    model(torch.FloatTensor(Xv).to(device)),
                    torch.FloatTensor(yv).to(device)).item()

        if monitor < best:
            best, best_epoch, counter = monitor, epoch, 0
            torch.save(model.state_dict(), args.out)
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for "
                      f"{args.patience} epochs)")
                break
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  loss {avg_loss:.6f}  monitor {monitor:.6f}")

    # ---- final report on the fine-tuning data, in physical units ----
    model.load_state_dict(torch.load(args.out, map_location=device))
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    preds = inv_y(preds_scaled)
    r2, rmse = r2_rmse(data[:, 3], preds.flatten())
    print(f"Fine-tuning data: R2 = {r2:.4f}, RMSE = {rmse:.4f} W/(m*K)")
    print(f"Best monitored loss: {best:.6f}")
    print(f"Saved fine-tuned weights to: {args.out}")
    if args.refit_scalers:
        X_all = np.vstack([data[:, :3], val_data[:, :3]]) if val_data is not None else data[:, :3]
        y_all = np.concatenate([data[:, 3], val_data[:, 3]]) if val_data is not None else data[:, 3]
        sidecar = os.path.splitext(args.out)[0] + "_scalers.npz"
        np.savez(sidecar,
                 x_min=X_all.min(axis=0), x_max=X_all.max(axis=0),
                 y_min=np.array([y_all.min()]), y_max=np.array([y_all.max()]))
        print(f"Saved refit scaler constants to: {sidecar}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Transfer-learning fine-tune of the pretrained base model")
    p.add_argument("--data", required=True,
                   help="CSV/XLSX with 4 columns: Z-eff-C, Ia-b, T, kappa")
    p.add_argument("--base-weights", default=BASE_WEIGHTS)
    p.add_argument("--out", default="output/finetuned_model.pth")
    p.add_argument("--freeze", type=int, default=2,
                   help="number of leading Linear layers to freeze (default 2)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.0,
                   help="held-out fraction for monitoring, e.g. 0.1")
    p.add_argument("--refit-scalers", action="store_true",
                   help="fit MinMax on your data instead of the released constants "
                        "(weakens transfer of the frozen layer - use with care)")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    finetune(p.parse_args())
