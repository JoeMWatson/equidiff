#!/usr/bin/env python3
"""Replay a demonstration from HDF5 through the IK-controlled environment.

Produces per-demo (index-stamped) outputs:
  replay_demo_{idx}.mp4         — user_camera composite view
  proprio_demo_{idx}.png        — proprioception comparison (HDF5 vs MuJoCo replay)
  pixel_error_demo_{idx}.mp4    — |hdf5_image − mj_render| for three views side-by-side
  training_data_demo_{idx}.mp4  — raw HDF5 images, three views side-by-side
"""
import argparse
import os

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np

from equi_diffpo.env_runner.bimanual_env import ThreeCubesEnvironment

IMG_H, IMG_W = 72, 128


def _env_obs_to_policy_obs(env_obs: dict) -> dict:
    """Mirror BimanualImageRunner._env_obs_to_policy_obs image resize (uint8, pre-normalisation)."""
    def _resize(img):
        return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return {
        "overhead": _resize(env_obs["overhead_camera"]),
        "left":     _resize(env_obs["left_camera"]),
        "right":    _resize(env_obs["right_camera"]),
    }


def _load_demo(hdf5_path: str, demo_idx: int):
    with h5py.File(hdf5_path, "r") as f:
        grp = f[f"data/demo_{demo_idx}"]
        actions = grp["actions"][:]  # (T, 14)
        seed = int(grp.attrs.get("seed", demo_idx))
        obs = grp["obs"]
        proprio = {
            "left_eef_pos":    obs["robot0_left_eef_pos"][:],
            "left_eef_quat":   obs["robot0_left_eef_quat"][:],
            "right_eef_pos":   obs["robot0_right_eef_pos"][:],
            "right_eef_quat":  obs["robot0_right_eef_quat"][:],
            "gripper_qpos":    obs["robot0_gripper_qpos"][:],
            "joint_left_pos":  obs["robot0_joint_left_pos"][:],
            "joint_right_pos": obs["robot0_joint_right_pos"][:],
        }
        images = {
            "overhead": obs["agentview_image"][:],
            "left":     obs["robot0_left_eye_in_hand_image"][:],
            "right":    obs["robot0_right_eye_in_hand_image"][:],
        }
    return actions, proprio, images, seed


def _extract_mj(env_obs: dict):
    gripper = np.array([env_obs["left_pos"][7], env_obs["right_pos"][7]], dtype=np.float32)
    proprio = {
        "left_eef_pos":    np.asarray(env_obs["left_ee_pos"],   dtype=np.float32),
        "left_eef_quat":   np.asarray(env_obs["left_ee_quat"],  dtype=np.float32),
        "right_eef_pos":   np.asarray(env_obs["right_ee_pos"],  dtype=np.float32),
        "right_eef_quat":  np.asarray(env_obs["right_ee_quat"], dtype=np.float32),
        "gripper_qpos":    gripper,
        "joint_left_pos":  np.asarray(env_obs["left_pos"][:7],  dtype=np.float32),
        "joint_right_pos": np.asarray(env_obs["right_pos"][:7], dtype=np.float32),
    }
    images = _env_obs_to_policy_obs(env_obs)
    return proprio, images, env_obs["user_camera"]


def replay_demo(
    hdf5_path: str,
    demo_idx: int,
    out_dir: str,
    no_video: bool = False,
    no_proprio: bool = False,
    no_pixel: bool = False,
    no_data_video: bool = False,
):
    actions, hdf5_proprio, hdf5_images, seed = _load_demo(hdf5_path, demo_idx)
    T = len(actions)

    env = ThreeCubesEnvironment(seed=seed)
    raw_obs = env.reset()

    mj_p_lists = {k: [] for k in hdf5_proprio}
    mj_i_lists = {k: [] for k in hdf5_images}
    replay_frames = []

    def _append(obs_dict):
        p, imgs, user = _extract_mj(obs_dict)
        for k in mj_p_lists:
            mj_p_lists[k].append(p[k])
        for k in mj_i_lists:
            mj_i_lists[k].append(imgs[k])
        replay_frames.append(user)

    _append(raw_obs)

    total_reward = 0.0
    success = False
    for t in range(T - 1):
        raw_obs, reward, done, _ = env.step(actions[t])
        _append(raw_obs)
        total_reward += reward
        if done:
            success = True
            print(f"  Success at step {t + 1}!")
            break

    env.close()
    print(f"Demo {demo_idx}: {len(replay_frames)} frames, reward={total_reward:.2f}, success={success}")

    n = min(T, len(replay_frames))
    mj_p = {k: np.stack(v[:n]) for k, v in mj_p_lists.items()}
    mj_i = {k: np.stack(v[:n]) for k, v in mj_i_lists.items()}

    if not no_video:
        path = os.path.join(out_dir, f"replay_demo_{demo_idx}.mp4")
        _write_video(path, replay_frames)
        print(f"  Replay → {path}")

    if not no_proprio:
        path = os.path.join(out_dir, f"proprio_demo_{demo_idx}.png")
        _plot_proprio(path, hdf5_proprio, mj_p, n, demo_idx)
        print(f"  Proprio → {path}")

    if not no_pixel:
        path = os.path.join(out_dir, f"pixel_error_demo_{demo_idx}.mp4")
        _write_pixel_error_video(path, hdf5_images, mj_i, n)
        print(f"  Pixel error → {path}")

    if not no_data_video:
        path = os.path.join(out_dir, f"training_data_demo_{demo_idx}.mp4")
        _write_data_video(path, hdf5_images)
        print(f"  Training data → {path}")


def _write_video(path: str, frames: list):
    writer = imageio.get_writer(path, fps=20)
    for f in frames:
        writer.append_data(np.asarray(f, dtype=np.uint8))
    writer.close()


def _plot_proprio(path: str, hdf5: dict, mj: dict, T: int, demo_idx: int):
    """4-per-row grid, one subplot per scalar channel, demo (solid) vs replay (dashed)."""
    labels_xyz  = ["x", "y", "z"]
    labels_quat = ["w", "x", "y", "z"]

    channels = []
    for side in ("left", "right"):
        for i, l in enumerate(labels_xyz):
            channels.append((f"{side}_eef_pos_{l}",
                             hdf5[f"{side}_eef_pos"][:T, i],
                             mj[f"{side}_eef_pos"][:T, i]))
        for i, l in enumerate(labels_quat):
            channels.append((f"{side}_eef_quat_{l}",
                             hdf5[f"{side}_eef_quat"][:T, i],
                             mj[f"{side}_eef_quat"][:T, i]))
    for i, l in enumerate(["L", "R"]):
        channels.append((f"gripper_{l}",
                         hdf5["gripper_qpos"][:T, i],
                         mj["gripper_qpos"][:T, i]))
    for side in ("left", "right"):
        for i in range(7):
            channels.append((f"{side}_j{i + 1}",
                             hdf5[f"joint_{side}_pos"][:T, i],
                             mj[f"joint_{side}_pos"][:T, i]))

    ncols = 7
    nrows = (len(channels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.2))
    axes_flat = axes.flatten()
    t = np.arange(T)

    for ax, (label, demo_vals, mj_vals) in zip(axes_flat, channels):
        ax.plot(t, demo_vals, lw=0.9, label="demo")
        ax.plot(t, mj_vals,  lw=0.9, ls="--", label="replay")
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=5)

    axes_flat[0].legend(fontsize=6, loc="best")

    for ax in axes_flat[len(channels):]:
        ax.set_visible(False)

    fig.suptitle(f"Proprioception — demo {demo_idx}  (solid=HDF5, dashed=MuJoCo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_data_video(path: str, hdf5_imgs: dict):
    """Raw HDF5 images, three views (overhead, left, right) side-by-side."""
    T = hdf5_imgs["overhead"].shape[0]
    writer = imageio.get_writer(path, fps=20)
    for t in range(T):
        frame = np.concatenate(
            [hdf5_imgs[v][t] for v in ("overhead", "left", "right")], axis=1
        )
        writer.append_data(frame)
    writer.close()


def _write_pixel_error_video(path: str, hdf5_imgs: dict, mj_imgs: dict, T: int):
    """Each frame: |hdf5 − mj| amplified ×4, three views (overhead, left, right) side-by-side."""
    writer = imageio.get_writer(path, fps=20)
    for t in range(T):
        strips = []
        for v in ("overhead", "left", "right"):
            h_f = hdf5_imgs[v][t].astype(np.int16)
            m_f = mj_imgs[v][t].astype(np.int16)
            err = np.clip(np.abs(h_f - m_f) * 4, 0, 255).astype(np.uint8)
            strips.append(err)
        writer.append_data(np.concatenate(strips, axis=1))  # (84, 252, 3)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5",       default="stacking_easy_1000.hdf5",
                        help="HDF5 dataset path")
    parser.add_argument("--demo",       type=int, default=0,
                        help="Demo index to replay")
    parser.add_argument("--out_dir",    default=".",
                        help="Output directory (files named by demo index)")
    parser.add_argument("--no_video",   action="store_true",
                        help="Skip replay video")
    parser.add_argument("--no_proprio", action="store_true",
                        help="Skip proprioception comparison plot")
    parser.add_argument("--no_pixel",      action="store_true",
                        help="Skip pixel error video")
    parser.add_argument("--no_data_video", action="store_true",
                        help="Skip training data video")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    replay_demo(
        args.hdf5,
        args.demo,
        args.out_dir,
        no_video=args.no_video,
        no_proprio=args.no_proprio,
        no_pixel=args.no_pixel,
        no_data_video=args.no_data_video,
    )
