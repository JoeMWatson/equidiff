#!/usr/bin/env python3
"""
create_hdf5.py  —  Build a robomimic-compatible HDF5 dataset for stack_three.

This script shows the EXACT schema used by the existing dataset so you can
plug in your own trajectories (from a simulator, motion planner, human tele-op,
data augmentation pipeline, etc.).

Usage:
    python create_hdf5.py \
        --out  my_dataset.hdf5 \
        --model_file  /path/to/existing_demo.hdf5   # to copy the MuJoCo XML from
        --n_demos 10   # how many synthetic demos to generate (default: uses YOUR data)

Structure written
─────────────────
data/
  demo_0/
    actions          (T, 7)   float64   [dx dy dz dax day daz gripper]  absolute
    obs/
      agentview_image          (T, 84, 84, 3)  uint8
      robot0_eye_in_hand_image (T, 84, 84, 3)  uint8
      object                   (T, 39)          float64
      robot0_eef_pos           (T, 3)           float64
      robot0_eef_quat          (T, 4)           float64
      robot0_eef_vel_ang       (T, 3)           float64
      robot0_eef_vel_lin       (T, 3)           float64
      robot0_gripper_qpos      (T, 2)           float64
      robot0_gripper_qvel      (T, 2)           float64
      robot0_joint_pos         (T, 7)           float64
      robot0_joint_pos_cos     (T, 7)           float64
      robot0_joint_pos_sin     (T, 7)           float64
      robot0_joint_vel         (T, 7)           float64
    rewards          (T,)     float64
    dones            (T,)     int64
    states           (T, 58)  float64
    [attrs]  model_file=<mujoco ...>  num_samples=T
  demo_1/ ...
"""

import argparse
import os
import numpy as np
import h5py
import pathlib
from scipy.spatial.transform import Rotation
import cv2



# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  —  edit these to match YOUR data source
# ─────────────────────────────────────────────────────────────────────────────

# Image size (must match what your renderer produces)
IMG_H, IMG_W, IMG_C = 84, 84, 3

# Action dimensionality  (7 = [eef_pos(3) + eef_rot(3) + gripper(1)])
ACTION_DIM = 7

# Object observation dim  (3 cubes × 13 = 39)
#   Per cube: pos(3) + quat(4) + vel_lin(3) + vel_ang(3) = 13
OBJECT_DIM = 39

# MuJoCo state dim  (qpos + qvel for the whole scene = 58)
STATE_DIM = 58

# Robot proprioception dims
EEF_POS_DIM    = 3
EEF_QUAT_DIM   = 4
EEF_VEL_ANG    = 3
EEF_VEL_LIN    = 3
GRIPPER_QPOS   = 2
GRIPPER_QVEL   = 2
JOINT_POS_DIM  = 7   # also used for cos/sin variants
JOINT_VEL_DIM  = 7


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: load MuJoCo XML string from an existing reference HDF5
# ─────────────────────────────────────────────────────────────────────────────

def load_model_xml(reference_hdf5_path: str) -> str:
    """Read the model_file attribute from demo_0 of an existing dataset."""
    with h5py.File(reference_hdf5_path, "r") as f:
        demo_key = sorted(f["data"].keys())[0]
        xml = f["data"][demo_key].attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
    return xml
    
def mp4_to_frames(video_path: str) -> np.ndarray:
    """
    Read an MP4 file and return all frames scaled to (84, 84, 3) as uint8.

    Args:
        video_path: Path to the .mp4 file.

    Returns:
        np.ndarray of shape (N, 84, 84, 3), dtype=uint8, values in [0, 255].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cv2 reads BGR → convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to (IMG_W, IMG_H) — note cv2 takes (width, height)
        frame_resized = cv2.resize(frame_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        frames.append(frame_resized)

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    # Stack into (N, 84, 84, 3), ensure uint8
    return np.stack(frames, axis=0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  REPLACE THIS FUNCTION WITH YOUR REAL DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "/home/joe/Research/code/data/stacking1031"

def collect_trajectory(demo_idx: int) -> dict:
    """
    Return one trajectory as a dict of numpy arrays.

    Replace the body of this function with whatever generates your data:
      • A robosuite/MimicGen rollout
      • A motion-planning episode
      • Loaded from your own format
      • Augmented from an existing trajectory
      • etc.

    Returns
    -------
    dict with keys:
        actions              np.ndarray  (T, 7)          float64
        obs_agentview        np.ndarray  (T, 84, 84, 3)  uint8
        obs_eye_in_hand      np.ndarray  (T, 84, 84, 3)  uint8
        obs_object           np.ndarray  (T, 39)         float64
        obs_eef_pos          np.ndarray  (T, 3)          float64
        obs_eef_quat         np.ndarray  (T, 4)          float64
        obs_eef_vel_ang      np.ndarray  (T, 3)          float64
        obs_eef_vel_lin      np.ndarray  (T, 3)          float64
        obs_gripper_qpos     np.ndarray  (T, 2)          float64
        obs_gripper_qvel     np.ndarray  (T, 2)          float64
        obs_joint_pos        np.ndarray  (T, 7)          float64
        obs_joint_pos_cos    np.ndarray  (T, 7)          float64
        obs_joint_pos_sin    np.ndarray  (T, 7)          float64
        obs_joint_vel        np.ndarray  (T, 7)          float64
        rewards              np.ndarray  (T,)            float64
        dones                np.ndarray  (T,)            int64
        states               np.ndarray  (T, 58)         float64
    """
    data_path = pathlib.Path(DATA_PATH) / f"stacking_{demo_idx}.npz"
    data = np.load(data_path)
    n = data["timesteps"].shape[0] - 1

    action_left_gripper = data["action_14"][:-1, None]
    action_right_gripper = data["action_15"][:-1, None]
    action_left_pos = data["left_ee_pos"][1:, :]
    # mujoco is w x y z
    action_left_ori = Rotation.from_quat(data["left_ee_quat"][1:, [1, 2, 3, 0]]).as_rotvec()
    action_right_pos = data["right_ee_pos"][1:, :]
    action_right_ori = Rotation.from_quat(data["right_ee_quat"][1:, [1, 2, 3, 0]]).as_rotvec()
    actions = np.concatenate((
        action_left_pos, action_left_ori, action_left_gripper, action_right_pos, action_right_ori, action_right_gripper
    ), axis=1)

    left_eye_in_hand = mp4_to_frames(str(pathlib.Path(DATA_PATH) / f"stacking_{demo_idx}_left.mp4"))
    right_eye_in_hand = mp4_to_frames(str(pathlib.Path(DATA_PATH) / f"stacking_{demo_idx}_right.mp4"))
    agent_view = mp4_to_frames(str(pathlib.Path(DATA_PATH) / f"stacking_{demo_idx}_overhead.mp4"))
    # 7 and 15 are grippers
    left_pos = np.concatenate([data[f"robot_state_{i}"][:, None] for i in range(0, 7)], axis=1)
    right_pos = np.concatenate([data[f"robot_state_{i}"][:, None] for i in range(8, 15)], axis=1)
    left_vel = np.concatenate([data[f"robot_state_{i}"][:, None] for i in range(16, 23)], axis=1)
    right_vel = np.concatenate([data[f"robot_state_{i}"][:, None] for i in range(23, 30)], axis=1)
    gripper = np.concatenate([data[f"robot_state_{i}"][:, None] for i in [7, 15]], axis=1)
    traj = {
        "actions":                 actions,
        "obs_agentview":           agent_view,
        "obs_left_eye_in_hand":    right_eye_in_hand,
        "obs_right_eye_in_hand":   left_eye_in_hand,
        "obs_blue_object":        np.concatenate((data["blue_pos"][:-1, :], data["blue_quat"][:-1, :]), axis=1),
        "obs_orange_object":        np.concatenate((data["orange_pos"][:-1, :], data["black_quat"][:-1, :]), axis=1),
        "obs_black_object":        np.concatenate((data["black_pos"][:-1, :], data["black_quat"][:-1, :]), axis=1),
        "obs_left_eef_pos":       data["left_ee_pos"][:-1, :],
        "obs_left_eef_quat":      data["left_ee_quat"][:-1, :],
        "obs_right_eef_pos":      data["right_ee_pos"][:-1, :],
        "obs_right_eef_quat":     data["right_ee_quat"][:-1, :],
        # "obs_left_eef_vel_ang":   rng.uniform(-1, 1, (T, EEF_VEL_ANG)).astype(np.float64),
        # "obs_right_eef_vel_ang":   rng.uniform(-1, 1, (T, EEF_VEL_ANG)).astype(np.float64),
        # "obs_left_eef_vel_lin":   np. (T, EEF_VEL_LIN)).astype(np.float64),
        # "obs_right_eef_vel_lin":   rng.uniform(-1, 1, (T, EEF_VEL_LIN)).astype(np.float64),
        "obs_gripper_qpos":  gripper,
        # "obs_gripper_qvel":  np.zeros((n, 2)),
        "obs_joint_left_pos":     left_pos,
        "obs_joint_right_pos":     right_pos,
        "obs_joint_left_pos_cos": np.cos(left_pos).astype(np.float64),
        "obs_joint_left_pos_sin": np.sin(left_pos).astype(np.float64),
        "obs_joint_right_pos_cos": np.cos(right_pos).astype(np.float64),
        "obs_joint_right_pos_sin": np.sin(right_pos).astype(np.float64),
        "obs_joint_left_vel":     left_vel,
        "obs_joint_right_vel":     right_vel,
        "rewards":           np.zeros(n, dtype=np.float64),
        "dones":             np.zeros(n, dtype=np.int64),
        # "states":            rng.uniform(-1, 1, (T, STATE_DIM)).astype(np.float64),
    }
    # mark last step as success
    traj["rewards"][-1] = 1.0
    traj["dones"][-1]   = 1

    return traj


# ─────────────────────────────────────────────────────────────────────────────
#  WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_demo(data_group: h5py.Group, demo_idx: int, traj: dict, model_xml: str):
    """Write one trajectory into the HDF5 data group."""
    demo_key = f"demo_{demo_idx}"
    grp = data_group.create_group(demo_key)

    T = traj["actions"].shape[0]

    # ── flat arrays ──────────────────────────────────────────────────────────
    grp.create_dataset("actions", data=traj["actions"],  dtype=np.float64)
    grp.create_dataset("rewards", data=traj["rewards"],  dtype=np.float64)
    grp.create_dataset("dones",   data=traj["dones"],    dtype=np.int64)
    # grp.create_dataset("states",  data=traj["states"],   dtype=np.float64)

    # ── observations ─────────────────────────────────────────────────────────
    obs = grp.create_group("obs")
    obs.create_dataset("agentview_image",          data=traj["obs_agentview"],    dtype=np.uint8)
    obs.create_dataset("robot0_left_eye_in_hand_image", data=traj["obs_left_eye_in_hand"],  dtype=np.uint8)
    obs.create_dataset("robot0_right_eye_in_hand_image", data=traj["obs_right_eye_in_hand"],  dtype=np.uint8)
    obs.create_dataset("orange_object",                   data=traj["obs_orange_object"],       dtype=np.float64)
    obs.create_dataset("blue_object",                   data=traj["obs_blue_object"],       dtype=np.float64)
    obs.create_dataset("black_object",                   data=traj["obs_black_object"],       dtype=np.float64)
    obs.create_dataset("robot0_left_eef_pos",           data=traj["obs_left_eef_pos"],      dtype=np.float64)
    obs.create_dataset("robot0_right_eef_pos",           data=traj["obs_right_eef_pos"],      dtype=np.float64)
    obs.create_dataset("robot0_left_eef_quat",          data=traj["obs_left_eef_quat"],     dtype=np.float64)
    obs.create_dataset("robot0_right_eef_quat",          data=traj["obs_right_eef_quat"],     dtype=np.float64)
    # obs.create_dataset("robot0_eef_vel_ang",       data=traj["obs_eef_vel_ang"],  dtype=np.float64)
    # obs.create_dataset("robot0_eef_vel_lin",       data=traj["obs_eef_vel_lin"],  dtype=np.float64)
    obs.create_dataset("robot0_gripper_qpos",      data=traj["obs_gripper_qpos"], dtype=np.float64)
    # obs.create_dataset("robot0_gripper_qvel",      data=traj["obs_gripper_qvel"], dtype=np.float64)
    obs.create_dataset("robot0_joint_left_pos",         data=traj["obs_joint_left_pos"],    dtype=np.float64)
    obs.create_dataset("robot0_joint_left_pos_cos",     data=traj["obs_joint_left_pos_cos"],dtype=np.float64)
    obs.create_dataset("robot0_joint_left_pos_sin",     data=traj["obs_joint_left_pos_sin"],dtype=np.float64)
    obs.create_dataset("robot0_joint_left_vel",         data=traj["obs_joint_left_vel"],    dtype=np.float64)
    #
    obs.create_dataset("robot0_joint_right_pos",         data=traj["obs_joint_right_pos"],    dtype=np.float64)
    obs.create_dataset("robot0_joint_right_pos_cos",     data=traj["obs_joint_right_pos_cos"],dtype=np.float64)
    obs.create_dataset("robot0_joint_right_pos_sin",     data=traj["obs_joint_right_pos_sin"],dtype=np.float64)
    obs.create_dataset("robot0_joint_right_vel",         data=traj["obs_joint_left_vel"],    dtype=np.float64)

    # ── per-demo attributes (robomimic reads these) ───────────────────────────
    grp.attrs["model_file"]  = model_xml
    grp.attrs["num_samples"] = T

    print(f"  wrote {demo_key}  T={T}")


def create_dataset(
    out_path:          str,
    n_demos:           int,
    # reference_hdf5:    str | None,
    train_ratio:       float = 0.9,
):
    # ── get MuJoCo XML ───────────────────────────────────────────────────────
    # if reference_hdf5 and os.path.exists(reference_hdf5):
    #     print(f"Loading model XML from: {reference_hdf5}")
    #     model_xml = load_model_xml(reference_hdf5)
    # else:
    #     print("Warning: no reference HDF5 given — model_file attr will be empty string.")
    model_xml = ""

    print(f"\nCreating {out_path}  ({n_demos} demos) …\n")

    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")

        for i in range(n_demos):
            try:
                traj = collect_trajectory(i)
                write_demo(data, i, traj, model_xml)
            except FileNotFoundError:
                print(f"No data for {i}")
                pass

        # ── optional: train/valid split mask ────────────────────────────────
        n_train = int(n_demos * train_ratio)
        all_keys = [f"demo_{i}" for i in range(n_demos)]
        train_keys = all_keys[:n_train]
        valid_keys = all_keys[n_train:]

        mask = f.create_group("mask")
        mask.create_dataset(
            "train",
            data=np.array(train_keys, dtype=h5py.special_dtype(vlen=str))
        )
        mask.create_dataset(
            "valid",
            data=np.array(valid_keys, dtype=h5py.special_dtype(vlen=str))
        )

    # ── quick verification ───────────────────────────────────────────────────
    print("\n── Verification ──")
    with h5py.File(out_path, "r") as f:
        demos = list(f["data"].keys())
        print(f"  Total demos : {len(demos)}")
        d0 = f["data"]["demo_0"]
        print(f"  demo_0 keys : {list(d0.keys())}")
        print(f"  actions     : {d0['actions'].shape}")
        print(f"  obs keys    : {list(d0['obs'].keys())}")
        # print(f"  states      : {d0['states'].shape}")
        print(f"  rewards     : unique={np.unique(d0['rewards'][:])}")
        print(f"  mask/train  : {len(f['mask']['train'])} demos")
        print(f"  mask/valid  : {len(f['mask']['valid'])} demos")

    print(f"\nDone! → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create a robomimic HDF5 dataset.")
    parser.add_argument("--out",        default="my_dataset.hdf5",
                        help="Output HDF5 path (default: my_dataset.hdf5)")
    # parser.add_argument("--model_file", default=None,
    #                     help="Path to an existing dataset HDF5 to copy the MuJoCo XML from")
    parser.add_argument("--n_demos",    type=int, default=10,
                        help="Number of demos to write (default: 10)")
    parser.add_argument("--train_ratio",type=float, default=0.9,
                        help="Fraction of demos in train split (default: 0.9)")
    args = parser.parse_args()

    create_dataset(
        out_path       = args.out,
        n_demos        = args.n_demos,
        # reference_hdf5 = args.model_file,
        train_ratio    = args.train_ratio,
    )


if __name__ == "__main__":
    main()
