# ROBOTIS AI Worker (FFW-SG2) Slip Detection Pipeline

Offline MCAP-to-window conversion, episode-grouped 1D CNN training, and per-episode evaluation for slip detection on the **ROBOTIS AI Worker (FFW-SG2)** using `/joint_states` (`sensor_msgs/msg/JointState`) recorded in ROS 2 Jazzy.

## Dataset Overview (`slip_dataset_full/`)

- **Episodes (`138` total):** `50` `held`, `21` `slipped`, `67` `notension`
- **Target Joints (`16`):**
  - Left arm: `arm_l_joint1` .. `arm_l_joint7`
  - Right arm: `arm_r_joint1` .. `arm_r_joint7`
  - Grippers: `gripper_l_joint1`, `gripper_r_joint1`
- **Window & Feature Layout:**
  - Sampling rate: `100 Hz`
  - Sliding window: `50` samples (`0.50 s`), stride: `10` samples (`0.10 s`)
  - Channels per sample: `80` (`48` base `pos`/`vel`/`eff` channels + `32` first-order temporal difference channels on `vel` and `eff` with `prepend=ve[:1]`)
- **Class Mapping:**
  - `0`: `held`
  - `1`: `slipped`
  - `2`: `notension`

## Scripts & Artifacts

- `convert_mcap_to_slip_dataset.py` — Parses uncompressed MCAP chunks and CDR-encoded `JointState` messages from directories or `.zip` archives, skips non-MCAP/corrupted files, assigns globally unique `episode_idx` IDs to prevent train/validation window leakage, and outputs `manifest.csv`, `episodes/*.csv`, `slip_windows_3class.npz`, and `slip_windows_binary.npz`.
- `train_slip_conv1d.py` — Trains `SlipConv1D` using episode-grouped `StratifiedGroupKFold`, computes per-channel mean/std strictly on training windows, and saves `slip_conv1d.pt` and `norm_stats_slip.npz`.
- `eval_episodes.py` — Evaluates a saved checkpoint per validation episode (`argmax_slip_frac`, `p>=thresh_frac`, `max_p_slip`, `mean_p_slip`).

## Commands

### 1. Environment Setup

~~~bash
cd ~/robotis_slip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
~~~

### 2. Convert MCAP Bags to Sliding Windows

~~~bash
python3 convert_mcap_to_slip_dataset.py \
  --notension_dir path/to/rosbags_notension \
  --towel_dir path/to/rosbags_towel \
  --out_dir slip_dataset_full \
  --window_size 50 \
  --stride 10
~~~

### 3. Train `SlipConv1D`

~~~bash
python3 train_slip_conv1d.py \
  --dataset_npz slip_dataset_full/slip_windows_3class.npz \
  --out_dir slip_dataset_full \
  --epochs 20 \
  --batch_size 64
~~~

### 4. Per-Episode Evaluation

~~~bash
python3 eval_episodes.py --prob_thresh 0.75
~~~

### 5. Cleanup & Artifact Verification Before Pausing

~~~bash
cd ~/robotis_slip
rm -rf ~/robotis_slip/raw_zips
ls -lh ~/robotis_slip/*.py ~/robotis_slip/slip_dataset_full/manifest.csv \
  ~/robotis_slip/slip_dataset_full/slip_windows_3class.npz \
  ~/robotis_slip/slip_dataset_full/slip_windows_binary.npz \
  ~/robotis_slip/slip_dataset_full/slip_conv1d.pt \
  ~/robotis_slip/slip_dataset_full/norm_stats_slip.npz
~~~

## Status & Next Steps (When Hardware Returns)

- **Completed offline:** Full 138-episode dataset conversion, `SlipConv1D` training (`slip_conv1d.pt` and `norm_stats_slip.npz`), per-episode evaluation, and 5-fold cross-check (`0.828–0.966` accuracy, `0.863–1.000` slip recall). Per-episode checks showed high-effort outliers on select bags (such as `ep_idx=3` and `ep_idx=71`), motivating probability thresholding plus multi-window debouncing in the controller.
- **Pending (requires FFW-SG2 / ROS 2 Jazzy access):** Build and test the real-time `/joint_states` ROS 2 Jazzy inference node using a `50`-sample rolling buffer, `norm_stats_slip.npz`, `torch.load(..., weights_only=False)`, and consecutive-window debouncing before triggering grip or impedance actions.