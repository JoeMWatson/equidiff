#!/usr/bin/env python3
"""Replay a demonstration from HDF5 through the IK-controlled environment.

NOTE: the HDF5 dataset has a known bug in write_stacking_dataset.py where
action_right_pos/ori were accidentally set to left_ee_pos/quat instead of
right_ee_pos/quat. This script reconstructs correct right-arm targets from
the obs fields (robot0_right_eef_pos / robot0_right_eef_quat) so the IK
controller can be tested properly.
"""
import argparse
import h5py
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from equi_diffpo.env_runner.bimanual_env import ThreeCubesEnvironment


def _load_actions(hdf5_path: str, demo_idx: int) -> np.ndarray:
    """Load 14D EE-space actions, correcting the right-arm bug in the dataset.

    The stored actions[:, 7:13] contain left-arm data (bug in
    write_stacking_dataset.py). We replace them with targets derived from the
    right-arm obs: for step t the target is the EE pose at obs[t+1].
    """
    with h5py.File(hdf5_path, "r") as f:
        grp     = f[f"data/demo_{demo_idx}"]
        actions = grp["actions"][:]                              # (T, 14)
        r_pos   = grp["obs/robot0_right_eef_pos"][:]            # (T, 3) wxyz
        r_quat  = grp["obs/robot0_right_eef_quat"][:]           # (T, 4) wxyz

    T = actions.shape[0]
    # obs[t] = state reached after action[t-1], so the target for action[t]
    # is obs[t+1]. Shift by 1; hold the last frame for the final step.
    right_pos_targets = np.vstack([r_pos[1:], r_pos[-1:]])      # (T, 3)
    right_quat_wxyz   = np.vstack([r_quat[1:], r_quat[-1:]])    # (T, 4) wxyz
    # wxyz -> xyzw for scipy, then convert to axis-angle
    right_aa = SciRotation.from_quat(
        right_quat_wxyz[:, [1, 2, 3, 0]]
    ).as_rotvec()                                                # (T, 3)

    corrected = actions.copy()
    corrected[:, 7:10] = right_pos_targets
    corrected[:, 10:13] = right_aa
    return corrected


def replay_demo(hdf5_path: str, demo_idx: int, output_video: str):
    actions = _load_actions(hdf5_path, demo_idx)

    env = ThreeCubesEnvironment(seed=demo_idx)
    obs, _ = env.reset()

    images = [obs["user_camera"]]
    total_reward = 0.0
    success = False

    for t, action in enumerate(actions):
        obs, reward, terminated, _, info = env.step(action)
        images.append(obs["user_camera"])
        total_reward += reward
        if terminated:
            success = True
            print(f"Success at step {t + 1}!")
            break

    print(f"Demo {demo_idx}: {len(images) - 1} steps, total_reward={total_reward:.2f}, success={success}")

    writer = imageio.get_writer(output_video, fps=20)
    for img in images:
        writer.append_data(img.astype(np.uint8))
    writer.close()
    print(f"Video saved to {output_video}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default="bimanual_three_cubes.hdf5")
    parser.add_argument("--demo", type=int, default=0)
    parser.add_argument("--output", default="replay.mp4")
    args = parser.parse_args()
    replay_demo(args.hdf5, args.demo, args.output)
