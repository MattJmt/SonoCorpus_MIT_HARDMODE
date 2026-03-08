#!/usr/bin/env python3
"""Train a multi-label finger-hit classifier from class-level CSV captures.

This script treats each source CSV as a stream:
- index.csv  -> target [1,0,0,0]
- middle.csv -> target [0,1,0,0]
- ring.csv   -> target [0,0,1,0]
- thumb.csv  -> target [0,0,0,1]
- noise.csv  -> target [0,0,0,0]

Windows are sampled from each stream and learned with BCE-with-logits so inference
can output multiple fingers simultaneously, or all-zero for noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FINGERS = ["index", "middle", "ring", "thumb"]
CHANNELS = ["index", "middle", "ring", "thumb"]
CLASS_FILES = ["index", "middle", "ring", "thumb", "noise"]
DEFAULT_EMA_ALPHA = 0.2


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class Chomp1d(nn.Module):
    """Trim right padding to keep causal Conv1d output length equal to input length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.net(x)
        return self.out_relu(out + residual)


class HitConvNet(nn.Module):
    """Small TCN for low-latency glove hit classification."""

    def __init__(self, in_channels: int = 4, out_dims: int = 4):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(in_channels, 24, kernel_size=3, dilation=1, dropout=0.2),
            TemporalBlock(24, 24, kernel_size=3, dilation=2, dropout=0.2),
            TemporalBlock(24, 32, kernel_size=3, dilation=4, dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32, out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a finger-hit multi-label classifier")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing index/middle/ring/thumb/noise CSV files",
    )
    p.add_argument("--window", type=int, default=32, help="Window size in timesteps")
    p.add_argument("--stride", type=int, default=2, help="Window stride")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Stop training if val loss does not improve for this many epochs",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val loss improvement to reset early stopping counter",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/finger_hit_model.pt"),
        help="Path to save the trained checkpoint",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default sigmoid threshold stored in checkpoint metadata",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="EMA alpha for preprocessing each channel before windowing",
    )
    p.add_argument(
        "--combo-aug-rate",
        type=float,
        default=0.35,
        help="Synthetic combo windows as fraction of real training windows (0 disables)",
    )
    p.add_argument(
        "--combo-mix-min",
        type=float,
        default=0.35,
        help="Minimum mix weight for first finger window in synthetic combos",
    )
    p.add_argument(
        "--combo-mix-max",
        type=float,
        default=0.65,
        help="Maximum mix weight for first finger window in synthetic combos",
    )
    p.add_argument(
        "--combo-gain-jitter",
        type=float,
        default=0.08,
        help="Random global gain jitter (+/- fraction) for synthetic combos",
    )
    p.add_argument(
        "--combo-noise-std",
        type=float,
        default=6.0,
        help="Gaussian noise std (ADC units) added to synthetic combos",
    )
    return p.parse_args()


def load_stream(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")

    try:
        arr = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(1, 2, 3, 4),
            dtype=np.float32,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load {path}: {exc}") from exc

    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 4:
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
    return arr


def one_hot_or_noise(name: str) -> np.ndarray:
    y = np.zeros((4,), dtype=np.float32)
    if name != "noise":
        y[FINGERS.index(name)] = 1.0
    return y


def apply_ema(stream: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(stream, dtype=np.float32)
    out[0] = stream[0]
    for i in range(1, len(stream)):
        out[i] = alpha * stream[i] + (1.0 - alpha) * out[i - 1]
    return out


def make_windows(stream: np.ndarray, window: int, stride: int) -> np.ndarray:
    if len(stream) < window:
        return np.empty((0, window, 4), dtype=np.float32)
    windows = []
    for start in range(0, len(stream) - window + 1, stride):
        windows.append(stream[start : start + window])
    return np.stack(windows, axis=0)


def synthesize_combo_windows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    aug_rate: float,
    mix_min: float,
    mix_max: float,
    gain_jitter: float,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if aug_rate <= 0.0:
        return x_train, y_train

    n_real = len(x_train)
    n_aug = int(n_real * aug_rate)
    if n_aug <= 0:
        return x_train, y_train

    idx_by_finger = [np.flatnonzero(y_train[:, i] > 0.5) for i in range(4)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    valid_pairs = [p for p in pairs if len(idx_by_finger[p[0]]) > 0 and len(idx_by_finger[p[1]]) > 0]
    if not valid_pairs:
        return x_train, y_train

    x_aug = np.empty((n_aug, x_train.shape[1], x_train.shape[2]), dtype=np.float32)
    y_aug = np.zeros((n_aug, y_train.shape[1]), dtype=np.float32)

    for i in range(n_aug):
        f1, f2 = valid_pairs[rng.integers(0, len(valid_pairs))]
        i1 = idx_by_finger[f1][rng.integers(0, len(idx_by_finger[f1]))]
        i2 = idx_by_finger[f2][rng.integers(0, len(idx_by_finger[f2]))]

        w1 = float(rng.uniform(mix_min, mix_max))
        w2 = 1.0 - w1
        mixed = w1 * x_train[i1] + w2 * x_train[i2]

        if gain_jitter > 0.0:
            gain = float(rng.uniform(1.0 - gain_jitter, 1.0 + gain_jitter))
            mixed *= gain
        if noise_std > 0.0:
            mixed += rng.normal(0.0, noise_std, size=mixed.shape).astype(np.float32)

        x_aug[i] = np.clip(mixed, 0.0, 1023.0)
        y_aug[i, f1] = 1.0
        y_aug[i, f2] = 1.0

    x_out = np.concatenate([x_train, x_aug], axis=0)
    y_out = np.concatenate([y_train, y_aug], axis=0)
    return x_out, y_out


def prepare_data(
    data_dir: Path,
    window: int,
    stride: int,
    val_frac: float,
    ema_alpha: float,
    combo_aug_rate: float,
    combo_mix_min: float,
    combo_mix_max: float,
    combo_gain_jitter: float,
    combo_noise_std: float,
    rng: np.random.Generator,
) -> PreparedData:
    x_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    x_val_parts: List[np.ndarray] = []
    y_val_parts: List[np.ndarray] = []

    for name in CLASS_FILES:
        arr = load_stream(data_dir / f"{name}.csv")
        arr = apply_ema(arr, alpha=ema_alpha)

        # Split by contiguous time to avoid overlap leakage from window sampling.
        split_idx = int((1.0 - val_frac) * len(arr))
        split_idx = min(max(split_idx, window), len(arr))

        train_stream = arr[:split_idx]
        val_stream = arr[split_idx - window + 1 :]

        x_train = make_windows(train_stream, window, stride)
        x_val = make_windows(val_stream, window, stride)

        y = one_hot_or_noise(name)
        y_train = np.repeat(y[None, :], repeats=len(x_train), axis=0)
        y_val = np.repeat(y[None, :], repeats=len(x_val), axis=0)

        x_train_parts.append(x_train)
        y_train_parts.append(y_train)
        x_val_parts.append(x_val)
        y_val_parts.append(y_val)

        print(
            f"{name:>6}: samples={len(arr):6d} train_windows={len(x_train):6d} "
            f"val_windows={len(x_val):6d}"
        )

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_val = np.concatenate(x_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)

    x_train, y_train = synthesize_combo_windows(
        x_train=x_train,
        y_train=y_train,
        aug_rate=combo_aug_rate,
        mix_min=combo_mix_min,
        mix_max=combo_mix_max,
        gain_jitter=combo_gain_jitter,
        noise_std=combo_noise_std,
        rng=rng,
    )
    print(
        f"train after combo augmentation: windows={len(x_train)} "
        f"(aug_rate={combo_aug_rate:.2f})"
    )

    # Normalize per channel using train split only.
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.maximum(std, 1e-6)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return PreparedData(x_train, y_train, x_val, y_val, mean, std)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.transpose(x, (0, 2, 1))).float()  # [N, C, T]
    yt = torch.from_numpy(y).float()
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def batch_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-8
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    losses = []
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(yb.cpu().numpy())

    probs = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    y_pred = (probs >= threshold).astype(np.float32)

    exact = np.mean(np.all(y_pred == y_true, axis=1))
    f1 = batch_f1(y_true, y_pred)
    return float(np.mean(losses)), float(exact), f1


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(y_train: np.ndarray) -> np.ndarray:
    # BCE pos_weight = num_negative / num_positive for each class.
    n = y_train.shape[0]
    pos = y_train.sum(axis=0).astype(np.float32)
    neg = float(n) - pos
    return neg / np.maximum(pos, 1.0)


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    if not (0.0 < args.ema_alpha <= 1.0):
        print("Error: --ema-alpha must be in (0, 1]", file=sys.stderr)
        return 2
    if args.combo_aug_rate < 0.0:
        print("Error: --combo-aug-rate must be >= 0", file=sys.stderr)
        return 2
    if not (0.0 <= args.combo_mix_min <= 1.0 and 0.0 <= args.combo_mix_max <= 1.0):
        print("Error: --combo-mix-min/max must be in [0, 1]", file=sys.stderr)
        return 2
    if args.combo_mix_min > args.combo_mix_max:
        print("Error: --combo-mix-min must be <= --combo-mix-max", file=sys.stderr)
        return 2
    if args.combo_gain_jitter < 0.0:
        print("Error: --combo-gain-jitter must be >= 0", file=sys.stderr)
        return 2
    if args.combo_noise_std < 0.0:
        print("Error: --combo-noise-std must be >= 0", file=sys.stderr)
        return 2
    if args.early_stop_patience < 1:
        print("Error: --early-stop-patience must be >= 1", file=sys.stderr)
        return 2
    if args.early_stop_min_delta < 0.0:
        print("Error: --early-stop-min-delta must be >= 0", file=sys.stderr)
        return 2

    prepared = prepare_data(
        args.data_dir,
        args.window,
        args.stride,
        args.val_frac,
        args.ema_alpha,
        args.combo_aug_rate,
        args.combo_mix_min,
        args.combo_mix_max,
        args.combo_gain_jitter,
        args.combo_noise_std,
        rng,
    )

    train_loader = to_loader(prepared.x_train, prepared.y_train, args.batch_size, shuffle=True)
    val_loader = to_loader(prepared.x_val, prepared.y_val, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HitConvNet().to(device)
    pos_weight_np = compute_pos_weight(prepared.y_train)
    pos_rate_np = prepared.y_train.mean(axis=0)
    print(
        "train positive rate: "
        + " ".join(f"{name}:{rate:.4f}" for name, rate in zip(FINGERS, pos_rate_np))
    )
    print(
        "train pos_weight: "
        + " ".join(f"{name}:{w:.2f}" for name, w in zip(FINGERS, pos_weight_np))
    )
    pos_weight = torch.from_numpy(pos_weight_np).float().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_loss, val_exact, val_f1 = evaluate(
            model, val_loader, criterion, device, threshold=args.threshold
        )
        train_loss = float(np.mean(train_losses))

        improved = (best_val_loss - val_loss) > args.early_stop_min_delta
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        f1_text = " ".join(f"{name}:{score:.3f}" for name, score in zip(FINGERS, val_f1))
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_exact={val_exact:.4f} val_f1[{f1_text}]"
        )
        if improved:
            print(f"  new best val_loss={best_val_loss:.4f} at epoch {epoch}")
        else:
            print(
                "  no val_loss improvement "
                f"({epochs_without_improve}/{args.early_stop_patience})"
            )

        if epochs_without_improve >= args.early_stop_patience:
            print(
                "Early stopping triggered: "
                f"no val loss improvement > {args.early_stop_min_delta:g} for "
                f"{args.early_stop_patience} consecutive epochs."
            )
            break

    if best_state is None:
        raise RuntimeError("Training failed: no checkpoint state was captured.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "meta": {
            "channels": CHANNELS,
            "fingers": FINGERS,
            "window": args.window,
            "stride": args.stride,
            "threshold": args.threshold,
            "ema_alpha": args.ema_alpha,
            "combo_aug_rate": args.combo_aug_rate,
            "combo_mix_min": args.combo_mix_min,
            "combo_mix_max": args.combo_mix_max,
            "combo_gain_jitter": args.combo_gain_jitter,
            "combo_noise_std": args.combo_noise_std,
            "mean": prepared.mean.tolist(),
            "std": prepared.std.tolist(),
        },
    }
    torch.save(checkpoint, args.output)

    print(f"Saved checkpoint: {args.output}")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    print("Inference rule: sigmoid(logits) >= threshold per class; all below => noise")
    print(json.dumps(checkpoint["meta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
