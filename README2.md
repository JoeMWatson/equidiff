# Running Bimanual Training with Apptainer

## Overview

Training uses `equidiff_new.sif` (Ubuntu 22.04 + CUDA 11.8 base). Packages that are either
missing from the container or need to be overridden for the local GPU are installed into
`.container_pkgs/` on the host and prepended to `PYTHONPATH` inside the container.

## Prerequisites

### 1. Local package overrides (`.container_pkgs/`)

These are installed once on the host with `pip install --target=.container_pkgs ...`.
The directory already exists in the repo; you should not need to redo this unless the
environment is rebuilt.

What's in there:

| Package | Reason |
|---|---|
| `mink==1.1.0`, `osqp`, `qpsolvers`, `daqp` | IK solver — not in container |
| `robomimic==0.3.0` (no deps) | `SpatialSoftmax` — not in container |
| `gymnasium==0.29.1` (no deps) | used by `bimanual_image_runner` — not in container |
| `torch==2.11.0+cu128`, `torchvision`, `torchaudio` | RTX 5070 Ti (sm_120/Blackwell) needs cu128; container ships cu118 which doesn't support sm_120 |
| `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`, etc. | CUDA 12 runtime libraries required by cu128 torch |

> **Note for cluster use:** The cluster has cu118-compatible drivers, so cu128 torch is not
> needed there. Build a container from `equidiff_py310_new.def` instead — it bakes in mink,
> robomimic, and gymnasium. You will still need to bind-mount `bimanual_suite/`.

### 2. `bimanual_suite/` on PYTHONPATH

`bimanual_suite/` is a local package (not pip-installed) that must be on `PYTHONPATH`
inside the container. It is passed via `APPTAINERENV_PYTHONPATH` in the launch command.

---

## Launch Commands (local, RTX 5070 Ti)

### Bimanual three-cubes

```bash
APPTAINERENV_WANDB_MODE=disabled \
APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
APPTAINERENV_MUJOCO_GL=osmesa \
apptainer exec --nv /home/joe/Research/code/equidiff/equidiff_new.sif \
  python train.py \
    --config-name=train_equi_diffusion_unet_bimanual_abs \
    dataset_path=/home/joe/Research/code/equidiff/bimanual_three_cubes.hdf5 \
    n_demo=853 \
    training.rollout_every=5 \
    logging.mode=disabled
```

### Stacking

```bash
APPTAINERENV_WANDB_MODE=disabled \
APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
APPTAINERENV_MUJOCO_GL=osmesa \
apptainer exec --nv equidiff_new.sif \
  python train.py \
    --config-name=train_equi_diffusion_unet_bimanual_abs \
    dataset_path=/home/joe/Research/code/equidiff/stacking_easy_1044.hdf5 \
    n_demo=1000 \
    training.rollout_every=1
```

### Key parameters to tune

| Parameter | Default | Notes |
|---|---|---|
| `n_demo` | 200 | number of demos to train on; dataset has 853 |
| `training.rollout_every` | `null` (disabled) | set to e.g. `5` to enable env rollouts every N epochs |
| `logging.mode` | `online` | set to `disabled` to skip wandb |
| `training.num_epochs` | `50000 / n_demo` | computed automatically |

---

## Environment Variables Explained

| Variable / Flag | Purpose |
|---|---|
| `APPTAINERENV_WANDB_MODE=disabled` | Disables wandb inside the container (plain `WANDB_MODE` does not propagate) |
| `APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs"` | Puts the container's GL libs before the host libs injected by `--nv`; without this, `libGLdispatch.so.0` from the host requires GLIBC 2.38 but the container only has 2.35. **Local machine only** — clusters running Ubuntu 22.04 hosts don't have this GLIBC mismatch. |
| `APPTAINERENV_PYTHONPATH=...` | Makes `.container_pkgs`, `bimanual_suite`, and the repo root visible inside the container |
| `APPTAINERENV_MUJOCO_GL=osmesa` | Software (headless) rendering for MuJoCo; avoids needing a display |
| `--nv` | Passes through the host NVIDIA GPU |
| `--bind /tmp/escnn_cache:/.../escnn/group/_cache` | escnn writes a joblib cache on import; the container squashfs is read-only so the write fails. Bind-mounting a host directory gives it a writable location. Only needed for `plot_predictions.py` (training doesn't import escnn directly at the top level). |

---

## Running on the Cluster

The local launch command has two machine-specific workarounds that are not needed on the cluster:

1. **cu128 torch + `nvidia-*-cu12` in `.container_pkgs/`** — the RTX 5070 Ti (Blackwell, sm_120) requires cu128. Any pre-Blackwell cluster GPU (A100, V100, H100, etc.) works with the cu118 torch already in the container.
2. **`APPTAINERENV_LD_LIBRARY_PATH` GL fix** — only needed locally due to GLIBC version mismatch between host (2.39) and container (2.35).

On the cluster, build the container from `equidiff_py310_new.def` (which bakes in mink, robomimic, and gymnasium), then:

```bash
APPTAINERENV_PYTHONPATH=/path/to/bimanual_suite:/path/to/equidiff:/opt/repos \
APPTAINERENV_MUJOCO_GL=osmesa \
apptainer exec --nv equidiff_py310_new.sif \
  python train.py \
    --config-name=train_equi_diffusion_unet_bimanual_abs \
    dataset_path=/home/joe/Research/code/equidiff/bimanual_three_cubes.hdf5 \
    n_demo=1000 \
    training.rollout_every=5
```

`bimanual_suite/` is not baked into the container and must always be provided via `APPTAINERENV_PYTHONPATH` (bind-mount or copy to the cluster alongside the repo).

---

## Quick Smoke Test

To verify the full pipeline (training + rollout) without a long run:

```bash
APPTAINERENV_WANDB_MODE=disabled \
APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
APPTAINERENV_MUJOCO_GL=osmesa \
apptainer exec --nv /home/joe/Research/code/equidiff/equidiff_new.sif \
  python train.py \
    --config-name=train_equi_diffusion_unet_bimanual_abs \
    dataset_path=/home/joe/Research/code/equidiff/bimanual_three_cubes.hdf5 \
    n_demo=853 \
    training.rollout_every=1 \
    training.debug=True \
    logging.mode=disabled \
    task.env_runner.max_steps=10 \
    task.env_runner.n_train=1 \
    task.env_runner.n_test=1 \
    task.env_runner.n_envs=1
```

A successful run completes two training epochs (each followed by a rollout) and exits cleanly.

---

## Plotting Action Predictions

`plot_predictions.py` queries the policy on one training episode and plots stitched predicted
actions vs ground-truth in a 2×8 grid (left arm / right arm columns; `pos_x/y/z`,
`quat_w/x/y/z`, `gripper` rows). GT is stored as 14D axis-angle; predictions are 20D rot6d —
both are converted to wxyz quaternion for comparison.

The script reads `n_demo` from the saved config and caps it to the number of demos actually
present in the HDF5, so you can point it at a different dataset than the one used for training.

```bash
mkdir -p /tmp/escnn_cache && \
APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
APPTAINERENV_MUJOCO_GL=osmesa \
apptainer exec --nv \
  --bind /tmp/escnn_cache:/opt/venv/lib/python3.10/site-packages/escnn/group/_cache \
  equidiff_new.sif python plot_predictions.py \
  --ckpt_dir data/outputs/2026.05.21/13.00.26_equi_diff_bimanual_three_cubes \
  --hdf5 bimanual_three_cubes_fixed.hdf5 \
  --demo 0 \
  --out predictions.png
```

| Argument | Description |
|---|---|
| `--ckpt_dir` | Run directory (contains `checkpoints/latest.ckpt` and `.hydra/config.yaml`) |
| `--hdf5` | HDF5 dataset to load the episode from |
| `--demo` | Demo index (default: 0) |
| `--out` | Output image path (default: `predictions.png`) |

---

## Observation Space

The policy uses image + EEF proprioception only — no object state:

| Key | Shape | Source |
|---|---|---|
| `agentview_image` | `[3, 84, 84]` | overhead camera, resized from 144×256 |
| `robot0_left_eye_in_hand_image` | `[3, 84, 84]` | left wrist camera |
| `robot0_right_eye_in_hand_image` | `[3, 84, 84]` | right wrist camera |
| `robot0_left_eef_pos` | `[3]` | left EEF position |
| `robot0_left_eef_quat` | `[4]` | left EEF quaternion |
| `robot0_right_eef_pos` | `[3]` | right EEF position |
| `robot0_right_eef_quat` | `[4]` | right EEF quaternion |
| `robot0_gripper_qpos` | `[2]` | left and right gripper positions |

**Action:** 20D — `[pos(3) + rot_6d(6) + gripper(1)]` × 2 arms. The env receives 14D axis-angle
actions; `BimanualImageRunner.undo_transform_action()` handles the conversion.
