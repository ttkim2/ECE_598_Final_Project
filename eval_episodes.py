#!/usr/bin/env python3
"""Per-episode evaluation of saved SlipConv1D checkpoint on validation episodes."""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from train_slip_conv1d import SlipConv1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="slip_dataset_full/slip_windows_3class.npz")
    parser.add_argument("--manifest_csv", default="slip_dataset_full/manifest.csv")
    parser.add_argument("--ckpt_path", default="slip_dataset_full/slip_conv1d.pt")
    parser.add_argument("--stats_path", default="slip_dataset_full/norm_stats_slip.npz")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prob_thresh", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(args.dataset_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    ep_idx = data["episode_idx"].astype(np.int64)
    ep_ids = data["episode_ids"]
    class_names = list(data["class_names"])
    slip_idx = class_names.index("slipped")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    _, val_idx = list(sgkf.split(X, y, ep_idx))[args.fold]

    stats = np.load(args.stats_path)
    mean = stats["mean"].reshape(1, 1, -1).astype(np.float32)
    std = stats["std"].reshape(1, 1, -1).astype(np.float32)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = SlipConv1D(
        in_channels=int(ckpt["in_channels"]),
        num_classes=int(ckpt["num_classes"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_val_norm = np.transpose((X[val_idx] - mean) / std, (0, 2, 1))
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(X_val_norm)), dim=1).numpy()

    val_eps = ep_idx[val_idx]
    val_y = y[val_idx]
    slip_probs = probs[:, slip_idx]
    argmax_preds = probs.argmax(axis=1)

    manifest = pd.read_csv(args.manifest_csv).set_index("global_episode_idx")

    rows = []
    for g_ep in np.unique(val_eps):
        mask = val_eps == g_ep
        ep_y = int(val_y[mask][0])
        ep_probs = slip_probs[mask]
        ep_argmax = argmax_preds[mask]
        ep_name = ep_ids[g_ep] if g_ep < len(ep_ids) else str(g_ep)
        dur = manifest.loc[g_ep, "duration_sec"] if g_ep in manifest.index else np.nan
        rows.append({
            "ep_idx": int(g_ep),
            "episode_id": ep_name,
            "true_label": class_names[ep_y],
            "dur_s": dur,
            "n_win": int(mask.sum()),
            "argmax_slip_frac": round(float((ep_argmax == slip_idx).mean()), 3),
            f"p>={args.prob_thresh:.2f}_frac": round(float((ep_probs >= args.prob_thresh).mean()), 3),
            "max_p_slip": round(float(ep_probs.max()), 3),
            "mean_p_slip": round(float(ep_probs.mean()), 3),
        })

    out_df = pd.DataFrame(rows).sort_values(["true_label", "max_p_slip"], ascending=[True, False])
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()