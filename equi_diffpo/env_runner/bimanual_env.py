import mujoco
import numpy as np
import gym
from scipy.spatial.transform import Rotation as SciRotation

import bimanual_suite.mjc as suite

INITIAL_POSE = {
    "ewellix_lift_top_joint": 0.4000,
    "ptu_pan": 0.0000,
    "ptu_tilt": -1.0000,
    "left_kinova_arm_joint_1": -np.pi + 2.9376,
    "left_kinova_arm_joint_2": 0.9546,
    "left_kinova_arm_joint_3": -2.5865,
    "left_kinova_arm_joint_4": -1.9366,
    "left_kinova_arm_joint_5": -1.5037,
    "left_kinova_arm_joint_6": -2.0908,
    "left_kinova_arm_joint_7": 1.2453,
    "right_kinova_arm_joint_1": -2.9376,
    "right_kinova_arm_joint_2": -0.9546,
    "right_kinova_arm_joint_3": 2.5865,
    "right_kinova_arm_joint_4": 1.9366,
    "right_kinova_arm_joint_5": 1.5037,
    "right_kinova_arm_joint_6": 2.0908,
    "right_kinova_arm_joint_7": -1.2453,
}

class ThreeCubesEnvironment(gym.Env):
    """
    Gymnasium wrapper for CubeAssembleEnvironment.

    Observation dict keys:
        left_pos       (8,)   left arm joint positions + gripper
        right_pos      (8,)   right arm joint positions + gripper
        left_vel       (7,)   left arm joint velocities
        right_vel      (7,)   right arm joint velocities
        left_ee_pos    (3,)   left end-effector Cartesian position
        left_ee_quat   (4,)   left end-effector orientation (w,x,y,z)
        right_ee_pos   (3,)
        right_ee_quat  (4,)
        black_pos      (3,)   black cube Cartesian position
        black_quat     (4,)   black cube orientation
        blue_pos       (3,)
        blue_quat      (4,)
        orange_pos     (3,)
        orange_quat    (4,)
        overhead_camera  (H, W, 3)  uint8 RGB
        left_camera      (H, W, 3)
        right_camera     (H, W, 3)
        user_camera      (2H, 2W, 3)  composite view

    Action space:
        Box(14,)
        [left_ee_pos(3), left_rot_aa(3), left_gripper(1),
         right_ee_pos(3), right_rot_aa(3), right_gripper(1)]
        Grippers in [0, 1]; all other dims unconstrained.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed: int = 0,
        render_height: int = 144,
        render_width: int = 256,
        render_mode: str = "rgb_array",
        task: str = "assemble",   # "assemble" | "disassemble"
    ):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed

        self._max_dq = suite.control.MAX_VEL_NORM * suite.control.DT

        if task == "assemble":
            self._env = suite.CubeAssembleEnvironment(seed, render_height, render_width, initial_pose=INITIAL_POSE)
        elif task == "disassemble":
            self._env = suite.CubeDissassembleEnvironment(seed, render_height, render_width)
        else:
            raise ValueError(f"Unknown task '{task}'. Choose 'assemble' or 'disassemble'.")

        H, W = render_height, render_width

        # ── observation space ───────────────────────────────────────────────
        img_space   = gym.spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8)
        img_wide    = gym.spaces.Box(0, 255, shape=(2*H, 2*W, 3), dtype=np.uint8)
        vec         = lambda n: gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        unit_quat   = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "left_pos":       vec(8),
            "right_pos":      vec(8),
            "left_vel":       vec(7),
            "right_vel":      vec(7),
            "left_ee_pos":    vec(3),
            "left_ee_quat":   unit_quat,
            "right_ee_pos":   vec(3),
            "right_ee_quat":  unit_quat,
            "overhead_cam_pos":  vec(3),
            "overhead_cam_quat": unit_quat,
            "left_cam_pos":      vec(3),
            "left_cam_quat":     unit_quat,
            "right_cam_pos":     vec(3),
            "right_cam_quat":    unit_quat,
            "black_pos":      vec(3),
            "black_quat":     unit_quat,
            "blue_pos":       vec(3),
            "blue_quat":      unit_quat,
            "orange_pos":     vec(3),
            "orange_quat":    unit_quat,
            "overhead_camera": img_space,
            "left_camera":     img_space,
            "right_camera":    img_space,
            "user_camera":     img_wide,
        })

        # ── action space ────────────────────────────────────────────────────
        # EE targets: [left_pos(3), left_aa(3), left_gripper(1),
        #              right_pos(3), right_aa(3), right_gripper(1)]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space.low[6]  = 0.0
        self.action_space.high[6] = 1.0
        self.action_space.low[13]  = 0.0
        self.action_space.high[13] = 1.0

        # ── IK setup ────────────────────────────────────────────────────────
        self._left_ee_id  = self._env.model.site("left_ee").id
        self._right_ee_id = self._env.model.site("right_ee").id

    # ── core API ────────────────────────────────────────────────────────────

    def seed(self, seed=None):
        self._seed = seed

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._seed = seed
        obs_raw = self._env.reset()
        return self._convert_obs(obs_raw)

    def step(self, action: np.ndarray):
        # action: [left_ee_pos(3), left_rot_aa(3), left_gripper(1),
        #          right_ee_pos(3), right_rot_aa(3), right_gripper(1)] = 14D
        action = np.asarray(action, dtype=np.float64)

        left_quat  = SciRotation.from_rotvec(action[3:6]).as_quat(scalar_first=True)
        right_quat = SciRotation.from_rotvec(action[10:13]).as_quat(scalar_first=True)

        dq_left  = self._ik_delta(action[0:3],  left_quat,  self._left_ee_id,  self._env.left_joint_state_to_vel)
        dq_right = self._ik_delta(action[7:10], right_quat, self._right_ee_id, self._env.right_joint_state_to_vel)

        dq_left  = self._max_dq * dq_left  / max(self._max_dq, np.linalg.norm(dq_left))
        dq_right = self._max_dq * dq_right / max(self._max_dq, np.linalg.norm(dq_right))

        left_joints  = self._env.data.qpos[self._env.left_joint_state_to_pos]
        right_joints = self._env.data.qpos[self._env.right_joint_state_to_pos]
        action_joint   = np.concatenate((left_joints + dq_left, right_joints + dq_right))
        action_gripper = action[[6, 13]]
        joint_action   = np.concatenate((action_joint, action_gripper))

        obs_raw    = self._env.step(joint_action)
        reward     = float(self._env.success())
        done       = bool(self._env.success())
        return self._convert_obs(obs_raw), reward, done, {"success": done}

    def _ik_delta(self, target_pos, target_quat_wxyz, site_id, joint_dof_ids):
        """Damped least-squares IK for one arm. Returns joint position delta (7,)."""
        model, data = self._env.model, self._env.data
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, joint_dof_ids]  # (6, 7)

        pos_err = target_pos - data.site_xpos[site_id]
        R_cur = SciRotation.from_matrix(data.site_xmat[site_id].reshape(3, 3))
        # target_quat_wxyz → xyzw for scipy
        R_tgt = SciRotation.from_quat(target_quat_wxyz[[1, 2, 3, 0]])
        rot_err = (R_tgt * R_cur.inv()).as_rotvec()

        err = np.concatenate([pos_err, rot_err])
        return J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)

    def render(self, mode=None):
        return self._env.observation["user_camera"]

    def close(self):
        self._env.close()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _convert_obs(obs: dict) -> dict:
        """Cast all float arrays to float32 (gym convention)."""
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v.astype(np.float32) if v.dtype.kind == "f" else v
            else:
                out[k] = v
        return out
