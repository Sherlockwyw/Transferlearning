"""
Feature / target scalers used by both the base model and fine-tuned models.

The original pipeline fitted two sklearn MinMaxScaler objects on the 872-row
pretraining set ("1106 transfer-learning dataset"):

    x : [Z-eff-C, Ia-b, T]           -> scaled to [0, 1]
    y : [thermal conductivity]        -> scaled to [0, 1]

    x.data_min_ = [4.8233333, 0.5564757, 25.0]
    x.data_max_ = [6.2372727, 0.7356714, 1500.0]
    y.data_min_ = [0.905086]
    y.data_max_ = [3.96221]

To make the released package free of pickle/sklearn-version dependencies, the
scaler constants are inlined here as plain numbers. This class reproduces
sklearn MinMaxScaler behaviour exactly:

    transform(x)         = (x - data_min) / (data_max - data_min)
    inverse_transform(y) = y * (data_max - data_min) + data_min
"""

import numpy as np

X_MIN = np.array([4.823333333333334, 0.5564757401283346, 25.0])
X_MAX = np.array([6.237272727272727, 0.7356713992442301, 1500.0])
Y_MIN = np.array([0.905086])
Y_MAX = np.array([3.96221])

_X_RANGE = X_MAX - X_MIN
_Y_RANGE = Y_MAX - Y_MIN


def transform_x(features):
    """Scale raw physical features [Z-eff-C, Ia-b, T(K)] to [0, 1]."""
    X = np.asarray(features, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[-1] != 3:
        raise ValueError(f"Expected 3 features [Z-eff-C, Ia-b, T(K)], got shape {X.shape}")
    return (X - X_MIN) / _X_RANGE


def inverse_transform_x(X_scaled):
    """Invert the feature scaling back to raw physical units."""
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    return X_scaled * _X_RANGE + X_MIN


def transform_y(y):
    """Scale thermal conductivity to [0, 1]."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return (y - Y_MIN) / _Y_RANGE


def inverse_transform_y(y_scaled):
    """Invert the target scaling back to W/(m*K)."""
    y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
    return y_scaled * _Y_RANGE + Y_MIN
