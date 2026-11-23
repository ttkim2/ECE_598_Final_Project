#!/usr/bin/env python3
"""
Train a 1D CNN slip detector on sliding-window /joint_states tensors
from the ROBOTIS AI Worker (FFW-SG2) dataset.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold


class SlipConv1D(nn.Module):
    """Lightweight 1D CNN over (batch, 80 channels, 50 time_steps)."""

    def __init__(self, in_channels: int = 80, num_classes: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN for ROBOTIS AI Worker slip detection.")
    parser.add_argument(
        "--dataset_npz",
        default="slip_dataset_full/slip_windows_3class.npz",
        help="Path to slip_windows_3class.npz or slip_windows_binary.npz",
    )
    parser.add_argument(
        "--out_dir",
        default="slip_dataset_full",
        help="Directory where model weights and normalization stats are saved",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fold", type=int, default=0, help="Which StratifiedGroupKFold split (0..4) to use")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset_npz)
    X = data["X"].astype(np.float32)                 # (N, 50, 80)
    y = data["y"].astype(np.int64)                   # (N,)
    groups = data["episode_idx"].astype(np.int64)    # Globally unique episode IDs
    class_names = data["class_names"]
    feature_names = data["feature_names"]
    num_classes = len(class_names)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(sgkf.split(X, y, groups))
    if not (0 <= args.fold < len(splits)):
        raise ValueError(f"--fold must be in [0, {len(splits) - 1}], got {args.fold}")
    train_idx, val_idx = splits[args.fold]

    # Fit per-channel mean/std strictly on training episodes
    mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6

    stats_path = os.path.join(args.out_dir, "norm_stats_slip.npz")
    np.savez(
        stats_path,
        mean=mean.squeeze(),
        std=std.squeeze(),
        feature_names=feature_names,
        class_names=class_names,
    )

    # Transpose (N, time_steps, channels) -> (N, channels, time_steps) for nn.Conv1d
    X_train = np.transpose((X[train_idx] - mean) / std, (0, 2, 1))
    X_val = np.transpose((X[val_idx] - mean) / std, (0, 2, 1))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y[val_idx])),
        batch_size=128,
        shuffle=False,
    )

    counts = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts * num_classes, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    model = SlipConv1D(in_channels=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    slip_class_idx = int(np.where(class_names == "slipped")[0][0]) if "slipped" in class_names else 1
    best_score = -1.0
    weights_path = os.path.join(args.out_dir, "slip_conv1d.pt")

    print(f"Device: {device} | Train windows: {len(train_idx)} | Val windows: {len(val_idx)}")
    print(f"Classes: {list(class_names)} | Train counts: {counts.astype(int).tolist()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(dim=1) == yb).sum().item()
            tr_total += len(yb)

        model.eval()
        va_correct, va_total, slip_tp, slip_fn, slip_fp = 0, 0, 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                va_correct += (preds == yb).sum().item()
                va_total += len(yb)
                slip_tp += ((preds == slip_class_idx) & (yb == slip_class_idx)).sum().item()
                slip_fn += ((preds != slip_class_idx) & (yb == slip_class_idx)).sum().item()
                slip_fp += ((preds == slip_class_idx) & (yb != slip_class_idx)).sum().item()

        val_acc = va_correct / max(va_total, 1)
        slip_recall = slip_tp / max(slip_tp + slip_fn, 1)
        slip_prec = slip_tp / max(slip_tp + slip_fp, 1)
        # Prioritize slip recall while still rewarding overall accuracy
        selection_score = 0.7 * slip_recall + 0.3 * val_acc

        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": X_train.shape[1],
                    "num_classes": num_classes,
                    "class_names": list(class_names),
                    "feature_names": list(feature_names),
                    "window_size": X.shape[1],
                },
                weights_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Loss: {tr_loss / max(tr_total, 1):.4f} | "
                f"Train Acc: {tr_correct / max(tr_total, 1):.3f} | "
                f"Val Acc: {val_acc:.3f} | "
                f"Slip Rec: {slip_recall:.3f} | Slip Prec: {slip_prec:.3f}"
            )

    print(f"Done. Saved best checkpoint to {weights_path} and stats to {stats_path}")


if __name__ == "__main__":
    main()