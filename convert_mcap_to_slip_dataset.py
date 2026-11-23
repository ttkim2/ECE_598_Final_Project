#!/usr/bin/env python3
"""
Convert ROS 2 Jazzy MCAP rosbags (/joint_states) from ROBOTIS AI Worker (FFW-SG2)
into per-episode CSVs and sliding-window NPZ tensors for slip detection.
"""
import argparse
import glob
import os
import struct
import zipfile
import numpy as np
import pandas as pd

TARGET_JOINTS = [
    # Left 7-DOF arm
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
    # Right 7-DOF arm
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
    # Grippers
    "gripper_l_joint1", "gripper_r_joint1",
]


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def decode_joint_state_cdr(cdr_bytes: bytes):
    """Decode ROS 2 CDR-serialized sensor_msgs/msg/JointState."""
    payload = cdr_bytes[4:]  # Skip 4-byte CDR encapsulation header
    off = 0
    sec, nanosec = struct.unpack("<iI", payload[off:off + 8])
    off += 8
    flen = struct.unpack("<I", payload[off:off + 4])[0]
    off += 4 + flen

    off = _align(off, 4)
    ncount = struct.unpack("<I", payload[off:off + 4])[0]
    off += 4
    names = []
    for _ in range(ncount):
        off = _align(off, 4)
        slen = struct.unpack("<I", payload[off:off + 4])[0]
        off += 4
        names.append(payload[off:off + slen - 1].decode("utf-8") if slen > 0 else "")
        off += slen

    seqs = []
    for _ in range(3):  # position, velocity, effort
        off = _align(off, 4)
        count = struct.unpack("<I", payload[off:off + 4])[0]
        off += 4
        if count > 0:
            off = _align(off, 8)
            vals = struct.unpack(f"<{count}d", payload[off:off + 8 * count])
            off += 8 * count
        else:
            vals = ()
        seqs.append(vals)

    stamp_ns = sec * 1_000_000_000 + nanosec
    return stamp_ns, names, seqs[0], seqs[1], seqs[2]


def read_mcap_joint_states(mcap_path: str) -> pd.DataFrame | None:
    """Parse uncompressed MCAP chunks and extract /joint_states into a DataFrame."""
    with open(mcap_path, "rb") as f:
        data = f.read()

    if data.startswith(b"version https://git-lfs.github.com"):
        print(f"[WARN] Skipping Git LFS pointer (run 'git lfs pull'): {mcap_path}")
        return None

    if len(data) < 16 or data[:8] != b"\x89MCAP0\r\n":
        print(f"[WARN] Skipping non-MCAP file ({data[:24]!r}): {mcap_path}")
        return None

    pos = 8
    rows = []
    t0 = None
    col_names = None

    while pos < len(data) - 8:
        opcode = data[pos]
        length = struct.unpack("<Q", data[pos + 1:pos + 9])[0]
        body = data[pos + 9:pos + 9 + length]
        pos += 9 + length

        if opcode == 0x06:  # Chunk record
            comp_len = struct.unpack("<I", body[28:32])[0]
            comp = body[32:32 + comp_len].decode("utf-8")
            if comp != "":
                raise NotImplementedError(f"Compressed MCAP ({comp}) not supported")
            rec_len = struct.unpack("<Q", body[32 + comp_len:40 + comp_len])[0]
            rec_data = body[40 + comp_len:40 + comp_len + rec_len]

            cpos = 0
            while cpos < len(rec_data):
                cop = rec_data[cpos]
                clen = struct.unpack("<Q", rec_data[cpos + 1:cpos + 9])[0]
                cbody = rec_data[cpos + 9:cpos + 9 + clen]
                cpos += 9 + clen

                if cop == 0x05:  # Message record
                    _, _, log_time, _ = struct.unpack("<HIQQ", cbody[:22])
                    stamp_ns, names, p, v, e = decode_joint_state_cdr(cbody[22:])
                    if t0 is None:
                        t0 = log_time
                        col_names = (
                            ["t_sec", "log_time_ns", "header_stamp_ns"]
                            + [f"{n}_pos" for n in names]
                            + [f"{n}_vel" for n in names]
                            + [f"{n}_eff" for n in names]
                        )
                    t_sec = (log_time - t0) * 1e-9
                    rows.append((t_sec, log_time, stamp_ns, *p, *v, *e))

    if not rows:
        print(f"[WARN] No /joint_states messages found in: {mcap_path}")
        return None

    return pd.DataFrame(rows, columns=col_names)


def _resolve_bag_dir(path: str, extract_dir: str) -> str:
    """Return path if it is an existing directory, or extract if it is a .zip."""
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and path.endswith(".zip"):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        return extract_dir
    raise FileNotFoundError(f"Bag path not found (expected directory or .zip): {path}")


def _build_windows_with_global_ids(episodes, window_size=50, stride=10, joints=TARGET_JOINTS):
    """
    Create sliding-window tensors (N, window_size, 80) and labels (N,)
    using globally unique episode IDs to prevent train/val window leakage.
    """
    feature_cols = [f"{j}_{s}" for j in joints for s in ("pos", "vel", "eff")]
    vel_eff_cols = [f"{j}_{s}" for j in joints for s in ("vel", "eff")]

    X_list, y_list, ep_ids, t_starts = [], [], [], []
    for global_ep_idx, _, label_id, df in episodes:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"[WARN] Skipping episode {global_ep_idx} due to missing columns: {missing[:4]}")
            continue
        if len(df) < window_size:
            continue

        base = df[feature_cols].to_numpy(dtype=np.float32)
        ve = df[vel_eff_cols].to_numpy(dtype=np.float32)
        d_ve = np.diff(ve, axis=0, prepend=ve[:1])
        feat = np.concatenate([base, d_ve], axis=1)

        for start in range(0, len(feat) - window_size + 1, stride):
            X_list.append(feat[start:start + window_size])
            y_list.append(label_id)
            ep_ids.append(global_ep_idx)
            t_starts.append(df["t_sec"].iloc[start])

    derived_cols = [f"{c}_diff" for c in vel_eff_cols]
    if not X_list:
        return (
            np.empty((0, window_size, len(feature_cols) + len(derived_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            feature_cols + derived_cols,
        )

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.int64),
        np.array(ep_ids, dtype=np.int64),
        np.array(t_starts, dtype=np.float32),
        feature_cols + derived_cols,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROBOTIS AI Worker MCAP bags to per-episode CSVs and NPZ windows."
    )
    parser.add_argument(
        "--notension_dir", "--notension_zip",
        dest="notension_src",
        default="rosbags_notension",
        help="Path to notension bag folder or zip archive",
    )
    parser.add_argument(
        "--towel_dir", "--towel_zip",
        dest="towel_src",
        default="rosbags_towel",
        help="Path to towel bag folder (containing held/slipped) or zip archive",
    )
    parser.add_argument("--out_dir", default="slip_dataset", help="Output dataset directory")
    parser.add_argument("--window_size", type=int, default=50, help="Samples per window (100 Hz -> 50 = 0.5s)")
    parser.add_argument("--stride", type=int, default=10, help="Stride in samples (100 Hz -> 10 = 0.1s)")
    args = parser.parse_args()

    extract_root = os.path.join(args.out_dir, "_raw_extracted")
    os.makedirs(os.path.join(args.out_dir, "episodes"), exist_ok=True)

    notension_base = _resolve_bag_dir(args.notension_src, os.path.join(extract_root, "notension"))
    towel_base = _resolve_bag_dir(args.towel_src, os.path.join(extract_root, "towel"))

    notension_folder = (
        os.path.join(notension_base, "rosbags_notension")
        if os.path.isdir(os.path.join(notension_base, "rosbags_notension"))
        else notension_base
    )
    towel_root = (
        os.path.join(towel_base, "rosbags_towel")
        if os.path.isdir(os.path.join(towel_base, "rosbags_towel"))
        else towel_base
    )

    # Label mapping:
    # 0 = held (towel grasped & held without slipping)
    # 1 = slipped (towel slipped during pull/tension)
    # 2 = notension (free-space / no-tension baseline motion)
    sources = [
        ("held", 0, os.path.join(towel_root, "held")),
        ("slipped", 1, os.path.join(towel_root, "slipped")),
        ("notension", 2, notension_folder),
    ]

    manifest_rows, episodes = [], []
    global_ep_idx = 0

    for label_name, label_id, folder in sources:
        for mcap_path in sorted(glob.glob(os.path.join(folder, "**", "*.mcap"), recursive=True)):
            df = read_mcap_joint_states(mcap_path)
            if df is None or df.empty:
                continue

            parent_name = os.path.basename(os.path.dirname(mcap_path))
            stem_name = os.path.splitext(os.path.basename(mcap_path))[0]
            ep_tag = parent_name if parent_name not in ("held", "slipped", "rosbags_notension", "") else stem_name
            ep_id = f"{label_name}__{ep_tag}_{global_ep_idx:04d}"

            df.insert(0, "episode_id", ep_id)
            df.insert(1, "label_name", label_name)
            df.insert(2, "label_id", label_id)
            df.insert(3, "global_episode_idx", global_ep_idx)

            csv_rel = os.path.join("episodes", f"{ep_id}.csv")
            df.to_csv(os.path.join(args.out_dir, csv_rel), index=False)

            dur = float(df["t_sec"].max())
            manifest_rows.append({
                "global_episode_idx": global_ep_idx,
                "episode_id": ep_id,
                "label_name": label_name,
                "label_id": label_id,
                "source_mcap": mcap_path,
                "num_samples": len(df),
                "duration_sec": round(dur, 4),
                "sample_rate_hz": round((len(df) - 1) / max(dur, 1e-6), 2),
                "csv_path": csv_rel,
            })
            episodes.append((global_ep_idx, ep_id, label_id, df))
            global_ep_idx += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)

    for name, subset, classes in [
        ("slip_windows_binary.npz", [e for e in episodes if e[2] in (0, 1)], ["held", "slipped"]),
        ("slip_windows_3class.npz", episodes, ["held", "slipped", "notension"]),
    ]:
        X, y, ep_idx, t_start, feat_names = _build_windows_with_global_ids(
            subset, window_size=args.window_size, stride=args.stride
        )
        np.savez_compressed(
            os.path.join(args.out_dir, name),
            X=X,
            y=y,
            episode_idx=ep_idx,
            t_start=t_start,
            feature_names=np.array(feat_names),
            episode_ids=np.array([e[1] for e in subset]),
            class_names=np.array(classes),
        )

    print(f"Converted {len(episodes)} valid episodes into {args.out_dir}/")


if __name__ == "__main__":
    main()