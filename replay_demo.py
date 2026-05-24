#!/usr/bin/env python3
"""Replay a demonstration from HDF5 through the IK-controlled environment."""
import argparse
import h5py
import imageio
import numpy as np

from equi_diffpo.env_runner.bimanual_env import ThreeCubesEnvironment


def _load_actions(hdf5_path: str, demo_idx: int) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        return f[f"data/demo_{demo_idx}/actions"][:]


def replay_demo(hdf5_path: str, demo_idx: int, output_video: str):
    actions = _load_actions(hdf5_path, demo_idx)

    env = ThreeCubesEnvironment(seed=demo_idx)
    obs = env.reset()

    images = [obs["user_camera"]]
    total_reward = 0.0
    success = False

    for t, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        images.append(obs["user_camera"])
        total_reward += reward
        if done:
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
