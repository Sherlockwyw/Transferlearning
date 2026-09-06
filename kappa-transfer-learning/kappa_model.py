"""
Model architecture shared by the pretrained base model and its transfer-learning
fine-tuned variant.

Architecture (MD5-verified against the released weights):
    Linear(3 -> 128) -> ReLU
    Linear(128 -> 64) -> ReLU
    Linear(64 -> 32) -> ReLU
    Linear(32 -> 16) -> ReLU
    Linear(16 -> 1)

Total parameters: 11,393
Input  (3 features, MinMax-scaled):  [Z-eff-C, Ia-b, T]
Output (1 value, MinMax-scaled):     thermal conductivity

Released weights in this repo:
    PreTrainModel/kappa_base.pth        - the pretrained base model
    TransferLearningModel/kappa_finetuned.pth - fine-tuned transfer model
"""

import os

import torch
import torch.nn as nn

INPUT_FEATURES = ["Z-eff-C", "Ia-b", "T"]
HIDDEN_SIZES = [128, 64, 32, 16]
OUTPUT_SIZE = 1

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_WEIGHTS = os.path.join(_ROOT, "PreTrainModel", "kappa_base.pth")
FINETUNED_WEIGHTS = os.path.join(_ROOT, "TransferLearningModel", "kappa_finetuned.pth")


class KappaMLP(nn.Module):
    """Five-layer fully connected network used for thermal-conductivity prediction."""

    def __init__(self, input_size: int = 3):
        super().__init__()
        layers = []
        prev = input_size
        for h in HIDDEN_SIZES:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, OUTPUT_SIZE))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def freeze_layers(model: nn.Module, n_frozen: int = 2) -> nn.Module:
    """
    Freeze the first `n_frozen` Linear layers (weight + bias) for transfer learning.

    Parameter indices of KappaMLP (each Linear contributes weight then bias):
        0: model.0.weight   1: model.0.bias   (Linear 3->128)
        2: model.2.weight   3: model.2.bias   (Linear 128->64)
        4: model.4.weight   5: model.4.bias   (Linear 64->32)
        ...

    The released fine-tuned model was produced with n_frozen=2 (its first two
    Linear layers are byte-identical to the base model - MD5 verified).
    """
    params = list(model.parameters())
    n_to_freeze = 2 * n_frozen
    for idx, p in enumerate(params):
        p.requires_grad = idx >= n_to_freeze
    return model


def load_model(weights_path: str = None, device: str = "cpu") -> nn.Module:
    """Load weights into KappaMLP (strict shape check) and set eval mode.

    weights_path: path to a .pth state_dict. If None, defaults to the
                  released fine-tuned model (TransferLearningModel/kappa_finetuned.pth).
    """
    model = KappaMLP()
    path = weights_path or FINETUNED_WEIGHTS
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# --- convenient aliases -----------------------------------------------------
load_base_model = load_model          # pass PreTrainModel/kappa_base.pth
load_finetuned_model = load_model     # pass TransferLearningModel/kappa_finetuned.pth


if __name__ == "__main__":
    m = KappaMLP()
    n_params = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"Total parameters: {n_params}")
    frozen = freeze_layers(m, n_frozen=2)
    n_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    print(f"Trainable after freezing 2 layers: {n_trainable} / {n_params}")
