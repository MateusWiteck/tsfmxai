"""DLinear from Zeng et al. (AAAI 2023), adapted to masked real-world windows.

Architecture follows the authors' public DLinear implementation with a
25-point moving average and shared temporal linear maps (individual=False).
Training and missing-value preprocessing below are local benchmark choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DLinear(nn.Module):
    """Shared-channel DLinear; input/output shape [examples, time, channels]."""

    def __init__(self, context_length: int = 96, horizon: int = 96, kernel_size: int = 25):
        super().__init__()
        if min(context_length, horizon) < 1 or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Lengths must be positive and moving-average kernel odd")
        self.context_length = context_length
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.seasonal = nn.Linear(context_length, horizon)
        self.trend = nn.Linear(context_length, horizon)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[1] != self.context_length:
            raise ValueError("Expected [batch, context_length, channels]")
        rows = context.permute(0, 2, 1)
        edge = self.kernel_size // 2
        moving = F.avg_pool1d(F.pad(rows, (edge, edge), mode="replicate"),
                              kernel_size=self.kernel_size, stride=1)
        return (self.seasonal(rows - moving) + self.trend(moving)).permute(0, 2, 1)


@dataclass(frozen=True)
class DLinearFit:
    model: DLinear
    mean: np.ndarray
    scale: np.ndarray
    best_epoch: int
    epochs_run: int
    validation_mse: float
    train_windows: int
    validation_windows: int
    weights_sha256: str


def _fill_context(context: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Forward-fill within each context, starting with train-only channel means."""
    filled = np.asarray(context, dtype=np.float32).copy()
    last = np.broadcast_to(mean, (len(filled), len(mean))).copy()
    for t in range(filled.shape[1]):
        valid = np.isfinite(filled[:, t, :])
        last = np.where(valid, filled[:, t, :], last)
        filled[:, t, :] = last
    return filled


def _prepare(windows, mean: np.ndarray, scale: np.ndarray):
    x = (_fill_context(windows.context, mean) - mean) / scale
    y = (np.nan_to_num(windows.target, nan=0.0) - mean) / scale
    mask = np.isfinite(windows.target)
    y = np.where(mask, y, 0.0)
    return (torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool))


def fit_dlinear(train, validation, *, seed: int = 20261117, epochs: int = 50,
                batch_size: int = 128, learning_rate: float = 1e-3,
                patience: int = 7) -> DLinearFit:
    """Train on earlier real years; select weights using a later validation year.

    This supervised in-domain reference has more information than a zero-shot
    synthetic-trained model. It is reported as such, never as an equal-data run.
    """
    if min(epochs, batch_size, patience) < 1 or learning_rate <= 0:
        raise ValueError("Invalid training controls")
    if train.channels != validation.channels or train.context.shape[1:] != validation.context.shape[1:]:
        raise ValueError("Train/validation channel and context shape mismatch")
    observed = np.concatenate((train.context.reshape(-1, len(train.channels)),
                               train.target.reshape(-1, len(train.channels))), axis=0)
    mean = np.nanmean(observed, axis=0).astype(np.float32)
    scale = np.nanstd(observed, axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Training data needs nonconstant observed values in each channel")
    x_train, y_train, m_train = _prepare(train, mean, scale)
    x_val, y_val, m_val = _prepare(validation, mean, scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DLinear(train.context.shape[1], train.target.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        bad = 0
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(len(x_train), generator=generator)
            for indices in permutation.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_train[indices])
                mask = m_train[indices]
                loss = ((prediction - y_train[indices]).square() * mask).sum() / mask.sum()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                prediction = model(x_val)
                val_loss = (((prediction - y_val).square() * m_val).sum() / m_val.sum()).item()
            if val_loss < best_loss - 1e-8:
                best_loss, best_epoch = val_loss, epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return DLinearFit(model, mean, scale, best_epoch, epoch, best_loss,
                      len(x_train), len(x_val), hashlib.sha256(buffer.getvalue()).hexdigest())


def predict_dlinear(fit: DLinearFit, windows, *, batch_size: int = 256) -> np.ndarray:
    """Forecast in original channel units with train-only normalization."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    x, _, _ = _prepare(windows, fit.mean, fit.scale)
    fit.model.eval()
    with torch.inference_mode():
        prediction = torch.cat([fit.model(block) for block in x.split(batch_size)]).numpy()
    return prediction * fit.scale + fit.mean
