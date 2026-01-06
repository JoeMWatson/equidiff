#!/usr/bin/env python3
"""
Script to load the MimicGen Stack Three task and run random actions.
"""
import numpy as np
from mimicgen.envs.robosuite.stack import *
import robosuite as suite
from robosuite.controllers import load_controller_config


def main():
    # Configuration options
    options = {
        "env_name": "StackThree_D1",  # MimicGen Stack Three task
        "robots": "Panda",  # Using Panda robot
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
    }

    # Create the environment
    print(f"Initializing environment: {options['env_name']}")
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    # Reset environment
    env.reset()

    # Set camera view
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(f"Action space: low={low}, high={high}")
    print(f"Action dimension: {len(low)}")

    # Run random actions
    print("\nRunning random actions...")
    for i in range(10000):
        # Sample random action
        action = np.random.uniform(low, high)

        # Step the environment
        obs, reward, done, info = env.step(action)

        # Render
        env.render()

        # Print progress every 100 steps
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/10000 - Reward: {reward:.4f}")

    print("\nSimulation complete!")
    env.close()


if __name__ == "__main__":
    main()