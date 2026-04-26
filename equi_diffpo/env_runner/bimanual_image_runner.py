import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import gymnasium
import wandb.sdk.data_types.video as wv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner

from equi_diffpo.env_runner.bimanual_env import ThreeCubesEnvironment  # noqa: F401


class BimanualImageRunner(BaseImageRunner):
    """
    Environment runner for the bimanual_three_cubes task.

    The dataset loading and action transformation are fully implemented.
    Environment creation (env_fn / dummy_env_fn) requires a gym-compatible
    wrapper around bimanual_suite and is left as TODO / NotImplementedError.

    Action convention (after undo_transform_action):
        (N_envs, T, 14) float32
        [left_eef_pos(3), left_rot_aa(3), left_gripper(1),
         right_eef_pos(3), right_rot_aa(3), right_gripper(1)]
    """

    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta: dict,
            n_train=6,
            n_train_vis=2,
            train_start_idx=0,
            n_test=10,
            n_test_vis=4,
            test_start_seed=100000,
            max_steps=700,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=True,
            tqdm_interval_sec=5.0,
            n_envs=None,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = os.path.expanduser(dataset_path)

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # ------------------------------------------------------------------
        # TODO: implement a gym-compatible wrapper for bimanual_suite and
        # replace the NotImplementedError below with real env construction.
        #
        # The wrapper must expose:
        #   - observation_space / action_space (gym.spaces)
        #   - reset() -> obs_dict   (keys matching shape_meta['obs'])
        #   - step(action) -> (obs_dict, reward, done, info)
        #     where action is (14,) float32 in axis-angle convention
        #   - seed(seed)
        #   - render() -> np.ndarray  (H, W, 3) uint8

        def env_fn():
          env = ThreeCubesEnvironment(seed=0)
          return MultiStepWrapper(
              VideoRecordingWrapper(
                  env,
                  video_recoder=VideoRecorder.create_h264(...),
                  file_path=None,
                  steps_per_render=steps_per_render,
              ),
              n_obs_steps=n_obs_steps,
              n_action_steps=n_action_steps,
              max_episode_steps=max_steps,
          )

        # ---- everything below this line is ready once env_fn is implemented ----

        env_fns = [env_fn] * n_envs
        env_seeds = []
        env_prefixs = []
        env_init_fn_dills = []

        # train rollouts – replay initial states from the dataset
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                        enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    # TODO: adapt init_state reset for bimanual_suite wrapper
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test rollouts – random seeds
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
        env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {p: 0 for p in set(env_prefixs)}

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval bimanual_three_cubes {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = {}
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path)

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix + 'max_score'] = self.max_rewards[prefix]

        return log_data

    def undo_transform_action(self, action):
        """
        Convert network output (N, T, 20) back to environment action (N, T, 14).
        20 = left[pos3 + rot6d6 + gripper1] + right[pos3 + rot6d6 + gripper1]
        14 = left[pos3 + rot_aa3 + gripper1] + right[pos3 + rot_aa3 + gripper1]
        """
        raw_shape = action.shape
        assert raw_shape[-1] == 20, f"Expected action dim 20, got {raw_shape[-1]}"
        # reshape to (..., 2, 10) – one row per arm
        action = action.reshape(*raw_shape[:-1], 2, 10)

        d_rot = action.shape[-1] - 4  # 10 - 4 = 6
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        # flatten back to (..., 14)
        uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction
