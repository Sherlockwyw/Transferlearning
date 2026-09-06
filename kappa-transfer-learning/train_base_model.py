"""
Reproduce the training of the released base model (PreTrainModel/kappa_base.pth).

The original base model was trained on an 872-row pretraining dataset with the
following recipe (extracted from the original training notebook):

    - architecture  : KappaMLP (3 -> 128 -> 64 -> 32 -> 16 -> 1, ReLU)
    - optimizer     : Adam, lr = 1e-3, weight_decay = 1e-4
    - loss          : MSE on MinMax-scaled target
    - batch size    : 32
    - max epochs    : 500, early stopping (patience = 50) on training loss
    - scalers       : MinMaxScaler fitted on the pretraining set

Expected performance of the released base weights on the original split:

    pretraining data : R2 = 0.8305, RMSE = 0.2149 W/(m*K)
    downstream test  : R2 = 0.5303, RMSE = 0.3769 W/(m*K)   (before fine-tuning)

Run:
    python train_base_model.py --data examples/pretrain.csv --out output/my_base_model.pth

Data file format (.csv or .xlsx, 4 columns, header optional):
    Z-eff-C, Ia-b, T(K), thermal conductivity [W/(m*K)]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kappa_model import KappaMLP
from data_util import load_tabular, r2_rmse


def fit_minmax(data):
    """Fit MinMax scaler parameters on [X | y] (same recipe as the original)."""
    X, y = data[:, :3], data[:, 3:4]
    return X.min(axis=0), X.max(axis=0), y.min(axis=0), y.max(axis=0)


def train(args):
    data = load_tabular(args.data)
    x_min, x_max, y_min, y_max = fit_minmax(data)
    X = (data[:, :3] - x_min) / (x_max - x_min)
    y = (data[:, 3:4] - y_min) / (y_max - y_min)
    print(f"Loaded {len(data)} rows; "
          f"x range {np.round(x_min, 3).tolist()}..{np.round(x_max, 3).tolist()}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = KappaMLP().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=args.batch_size, shuffle=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_loss, best_epoch, train_losses = float("inf"), 0, []
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
        avg = epoch_loss / len(loader)
        train_losses.append(avg)

        if avg < best_loss:
            best_loss, best_epoch = avg, epoch
            torch.save(model.state_dict(), args.out)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  loss {avg:.6f}  (best {best_loss:.6f})")

    # Final report on the pretraining data, in physical units
    model.load_state_dict(torch.load(args.out, map_location=device))
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    preds = preds_scaled * (y_max - y_min) + y_min
    r2, rmse = r2_rmse(data[:, 3], preds.flatten())
    print(f"Pretraining data: R2 = {r2:.4f}, RMSE = {rmse:.4f} W/(m*K)")
    print(f"Best train MSE (scaled): {best_loss:.6f}")
    print(f"Saved best weights to   : {args.out}")
    scaler_path = os.path.splitext(args.out)[0] + "_scalers.npz"
    np.savez(scaler_path, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    print(f"Saved fitted scalers to : {scaler_path}")
    print("NOTE: keep the .npz file together with the weights - the "
          "MinMax scalers are required at inference time.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train kappa-base from scratch")
    p.add_argument("--data", required=True,
                   help="CSV/XLSX with 4 columns: Z-eff-C, Ia-b, T, kappa")
    p.add_argument("--out", default="output/my_base_model.pth")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())
