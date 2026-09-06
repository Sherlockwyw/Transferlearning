"""
Shared data loading and metric helpers.
"""

import numpy as np
import pandas as pd

FEATURES = ["Z-eff-C", "Ia-b", "T"]
TARGET = "thermal_conductivity"
COLUMNS = FEATURES + [TARGET]


def _column_names_are_numeric(df):
    """True if pandas consumed the first data row as numeric column names."""
    return pd.to_numeric(pd.Series(list(df.columns)), errors="coerce").notna().all()


def load_tabular(path):
    """Read a CSV/XLSX file with 4 columns (3 features + 1 target).

    Returns an (N, 4) float array: [Z-eff-C, Ia-b, T, kappa].
    Headerless files (numeric column labels after default read) are re-read
    without a header.
    """
    if path.endswith(".csv"):
        reader = pd.read_csv
    elif path.endswith((".xlsx", ".xls")):
        reader = pd.read_excel
    else:
        raise ValueError("Only .csv / .xlsx are supported")

    df = reader(path)
    if df.shape[1] < 4:
        raise ValueError(f"Expected 4 columns (3 features + 1 target), got {df.shape[1]}")
    if _column_names_are_numeric(df):
        # headerless file: first data row was consumed as column names
        df = reader(path, header=None).iloc[:, :4]
    else:
        df = df.iloc[:, :4]
    df.columns = COLUMNS
    return df[COLUMNS].astype(float).values


def r2_rmse(y_true, y_pred):
    """R2 and RMSE in the given (physical) units."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return r2, rmse
