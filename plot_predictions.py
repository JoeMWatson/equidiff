#!/usr/bin/env python3
"""
Plot policy action predictions vs ground-truth for one training episode.

Usage (inside container):
  python plot_predictions.py \
    --ckpt_dir data/outputs/2026.05.21/13.00.26_equi_diff_bimanual_three_cubes \
    --hdf5 bimanual_three_cubes_fixed.hdf5 \
    --demo 0 \
    --out predictions.png
"""
import sys
import os
import pathlib
import argparse
import numpy as np
import torch
import h5py
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import hydra

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, os.path.join(ROOT_DIR, '.container_pkgs'))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("get_max_steps", lambda name: 700, replace=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROW_LABELS = ['pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z', 'gripper']


def resize_image(img):
    out = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return np.moveaxis(out, -1, 0)


def build_obs(demo_obs, t, n_obs_steps):
    def get_slice(arr):
        start = max(0, t - n_obs_steps + 1)
        frames = arr[start:t + 1]
        if len(frames) < n_obs_steps:
            pad = np.repeat(frames[:1], n_obs_steps - len(frames), axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        return frames

    obs = {}
    for k in ['agentview_image', 'robot0_left_eye_in_hand_image', 'robot0_right_eye_in_hand_image']:
        frames = get_slice(demo_obs[k])
        processed = np.stack([resize_image(f) for f in frames])
        obs[k] = torch.from_numpy(processed).unsqueeze(0).to(DEVICE)
    for k in ['robot0_left_eef_pos', 'robot0_left_eef_quat', 'robot0_right_eef_pos',
              'robot0_right_eef_quat', 'robot0_gripper_qpos']:
        frames = get_slice(demo_obs[k]).astype(np.float32)
        obs[k] = torch.from_numpy(frames).unsqueeze(0).to(DEVICE)
    return obs


def aa_batch_to_wxyz(aa):
    """(T, 3) axis-angle → (T, 4) wxyz quaternion"""
    xyzw = Rotation.from_rotvec(aa).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def to_8d_per_arm(actions_14):
    """(T, 14) aa-format → (T, 2, 8) wxyz-quat format"""
    T = actions_14.shape[0]
    out = np.zeros((T, 2, 8))
    for i, start in enumerate([0, 7]):
        out[:, i, :3] = actions_14[:, start:start + 3]
        out[:, i, 3:7] = aa_batch_to_wxyz(actions_14[:, start + 3:start + 6])
        out[:, i, 7] = actions_14[:, start + 6]
    return out


def pred20_to_8d_per_arm(actions_20, rot_transformer):
    """(T, 20) rot6d-format → (T, 2, 8) wxyz-quat format"""
    T = actions_20.shape[0]
    out = np.zeros((T, 2, 8))
    for i, start in enumerate([0, 10]):
        pos   = actions_20[:, start:start + 3]
        rot6d = actions_20[:, start + 3:start + 9]
        grip  = actions_20[:, start + 9]
        aa = rot_transformer.inverse(rot6d)  # (T, 3)
        out[:, i, :3] = pos
        out[:, i, 3:7] = aa_batch_to_wxyz(aa)
        out[:, i, 7] = grip
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--hdf5', required=True)
    parser.add_argument('--demo', type=int, default=0)
    parser.add_argument('--out', default='predictions.png')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, 'checkpoints', 'latest.ckpt')
    cfg_path  = os.path.join(args.ckpt_dir, '.hydra', 'config.yaml')

    cfg = OmegaConf.load(cfg_path)
    cfg.dataset_path = args.hdf5
    with h5py.File(args.hdf5, 'r') as _f:
        n_demo_available = len(_f['data'].keys())
    if cfg.n_demo > n_demo_available:
        cfg.n_demo = n_demo_available

    from equi_diffpo.workspace.train_equi_workspace import TrainEquiWorkspace
    from equi_diffpo.model.common.rotation_transformer import RotationTransformer

    workspace = TrainEquiWorkspace(cfg)
    workspace.load_checkpoint(path=ckpt_path)
    print(f"Loaded checkpoint at epoch {workspace.epoch}")

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    policy = workspace.ema_model
    policy.set_normalizer(normalizer)
    policy.eval()
    policy.to(DEVICE)

    rot_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    with h5py.File(args.hdf5, 'r') as f:
        demo = f[f'data/demo_{args.demo}']
        T = demo['actions'].shape[0]
        print(f"Demo {args.demo}: {T} steps")
        demo_obs = {
            'agentview_image':               demo['obs']['agentview_image'][:],
            'robot0_left_eye_in_hand_image':  demo['obs']['robot0_left_eye_in_hand_image'][:],
            'robot0_right_eye_in_hand_image': demo['obs']['robot0_right_eye_in_hand_image'][:],
            'robot0_left_eef_pos':            demo['obs']['robot0_left_eef_pos'][:].astype(np.float32),
            'robot0_left_eef_quat':           demo['obs']['robot0_left_eef_quat'][:].astype(np.float32),
            'robot0_right_eef_pos':           demo['obs']['robot0_right_eef_pos'][:].astype(np.float32),
            'robot0_right_eef_quat':          demo['obs']['robot0_right_eef_quat'][:].astype(np.float32),
            'robot0_gripper_qpos':            demo['obs']['robot0_gripper_qpos'][:].astype(np.float32),
        }
        gt_actions_14 = demo['actions'][:].astype(np.float32)  # (T, 14)

    gt = to_8d_per_arm(gt_actions_14)  # (T, 2, 8)

    n_action_steps = cfg.n_action_steps
    n_obs_steps    = cfg.n_obs_steps
    pred = np.full((T, 2, 8), np.nan)

    chunk_starts = list(range(0, T, n_action_steps))
    print(f"Running inference for {len(chunk_starts)} chunks...")
    for idx, t in enumerate(chunk_starts):
        obs = build_obs(demo_obs, t, n_obs_steps)
        with torch.no_grad():
            result = policy.predict_action(obs)
        pred_20 = result['action_pred'].squeeze(0).cpu().numpy()  # (horizon, 20)
        n = min(n_action_steps, T - t)
        pred[t:t + n] = pred20_to_8d_per_arm(pred_20[:n], rot_transformer)
        if (idx + 1) % 10 == 0 or idx + 1 == len(chunk_starts):
            print(f"  {idx + 1}/{len(chunk_starts)}")

    fig, axes = plt.subplots(8, 2, figsize=(12, 20), sharex=True)
    t_range = np.arange(T)

    for col, arm_name in enumerate(['Left arm', 'Right arm']):
        for row, label in enumerate(ROW_LABELS):
            ax = axes[row, col]
            ax.plot(t_range, gt[:, col, row], 'k-', linewidth=1.0, label='GT')
            ax.plot(t_range, pred[:, col, row], 'r--', linewidth=0.8, alpha=0.8, label='Pred')
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(arm_name, fontsize=10)
                ax.legend(fontsize=7, loc='upper right')
            if row == 7:
                ax.set_xlabel('timestep', fontsize=8)

    plt.suptitle(f"Demo {args.demo} — Stitched Predictions vs GT", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
