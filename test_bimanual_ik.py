import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRotation

from equi_diffpo.env_runner.bimanual_env import ThreeCubesEnvironment


@pytest.fixture(scope="module")
def env():
    e = ThreeCubesEnvironment(seed=0)
    e.reset()
    yield e
    e.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _site_pos(env, site_id):
    return env._env.data.site_xpos[site_id].copy()


def _site_quat_wxyz(env, site_id):
    mat = env._env.data.site_xmat[site_id].reshape(3, 3)
    return SciRotation.from_matrix(mat).as_quat(scalar_first=True)


def _site_aa(env, site_id):
    mat = env._env.data.site_xmat[site_id].reshape(3, 3)
    return SciRotation.from_matrix(mat).as_rotvec()


def _hold_action(env):
    """14D action commanding current EE poses with grippers open."""
    return np.concatenate([
        _site_pos(env, env._left_ee_id),
        _site_aa(env, env._left_ee_id),
        [0.0],
        _site_pos(env, env._right_ee_id),
        _site_aa(env, env._right_ee_id),
        [0.0],
    ])


# ── _ik_delta tests ───────────────────────────────────────────────────────────

def test_ik_delta_left_shape(env):
    dq = env._ik_delta(
        _site_pos(env, env._left_ee_id),
        _site_quat_wxyz(env, env._left_ee_id),
        env._left_ee_id,
        env._env.left_joint_state_to_vel,
    )
    assert dq.shape == (7,)


def test_ik_delta_right_shape(env):
    dq = env._ik_delta(
        _site_pos(env, env._right_ee_id),
        _site_quat_wxyz(env, env._right_ee_id),
        env._right_ee_id,
        env._env.right_joint_state_to_vel,
    )
    assert dq.shape == (7,)


def test_ik_delta_zero_error_gives_zero_delta(env):
    """When target == current EE pose, delta should be zero (J.T @ 0 = 0)."""
    dq = env._ik_delta(
        _site_pos(env, env._left_ee_id),
        _site_quat_wxyz(env, env._left_ee_id),
        env._left_ee_id,
        env._env.left_joint_state_to_vel,
    )
    assert np.allclose(dq, 0, atol=1e-10)


def test_ik_delta_reduces_left_pos_error(env):
    """Applying the left-arm delta should move the EE closer to target."""
    env.reset()
    target_pos = _site_pos(env, env._left_ee_id) + np.array([0.05, 0.0, 0.0])
    quat = _site_quat_wxyz(env, env._left_ee_id)
    err_before = np.linalg.norm(target_pos - _site_pos(env, env._left_ee_id))

    qpos_saved = env._env.data.qpos.copy()
    dq = env._ik_delta(target_pos, quat, env._left_ee_id, env._env.left_joint_state_to_vel)
    env._env.data.qpos[env._env.left_joint_state_to_pos] += dq
    mujoco.mj_forward(env._env.model, env._env.data)
    err_after = np.linalg.norm(target_pos - _site_pos(env, env._left_ee_id))

    # Restore state
    env._env.data.qpos[:] = qpos_saved
    mujoco.mj_forward(env._env.model, env._env.data)

    assert err_after < err_before


def test_ik_delta_reduces_right_pos_error(env):
    """Applying the right-arm delta should move the EE closer to target."""
    env.reset()
    target_pos = _site_pos(env, env._right_ee_id) + np.array([0.0, -0.05, 0.0])
    quat = _site_quat_wxyz(env, env._right_ee_id)
    err_before = np.linalg.norm(target_pos - _site_pos(env, env._right_ee_id))

    qpos_saved = env._env.data.qpos.copy()
    dq = env._ik_delta(target_pos, quat, env._right_ee_id, env._env.right_joint_state_to_vel)
    env._env.data.qpos[env._env.right_joint_state_to_pos] += dq
    mujoco.mj_forward(env._env.model, env._env.data)
    err_after = np.linalg.norm(target_pos - _site_pos(env, env._right_ee_id))

    env._env.data.qpos[:] = qpos_saved
    mujoco.mj_forward(env._env.model, env._env.data)

    assert err_after < err_before


# ── step tests ────────────────────────────────────────────────────────────────

def test_step_returns_five_tuple(env):
    env.reset()
    result = env.step(_hold_action(env))
    assert len(result) == 5


def test_step_obs_has_expected_keys(env):
    env.reset()
    obs, *_ = env.step(_hold_action(env))
    for key in ["left_pos", "right_pos", "left_ee_pos", "right_ee_pos",
                "left_ee_quat", "right_ee_quat", "overhead_camera"]:
        assert key in obs, f"Missing key: {key}"


def test_step_obs_dtypes(env):
    """Float obs should be float32; image obs should be uint8."""
    env.reset()
    obs, *_ = env.step(_hold_action(env))
    assert obs["left_pos"].dtype == np.float32
    assert obs["overhead_camera"].dtype == np.uint8


def test_step_reward_and_info_types(env):
    env.reset()
    _, reward, terminated, truncated, info = env.step(_hold_action(env))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in info


def test_step_left_ee_moves_toward_target(env):
    """Left EE should be closer to target after 10 steps than at the start."""
    env.reset()
    target_left = _site_pos(env, env._left_ee_id) + np.array([0.1, 0.0, 0.0])
    err_before = np.linalg.norm(target_left - _site_pos(env, env._left_ee_id))

    action = np.concatenate([
        target_left,
        _site_aa(env, env._left_ee_id),
        [0.0],
        _site_pos(env, env._right_ee_id),
        _site_aa(env, env._right_ee_id),
        [0.0],
    ])
    for _ in range(10):
        env.step(action)

    err_after = np.linalg.norm(target_left - _site_pos(env, env._left_ee_id))
    assert err_after < err_before


def test_step_right_ee_moves_toward_target(env):
    """Right EE should be closer to target after 10 steps than at the start."""
    env.reset()
    target_right = _site_pos(env, env._right_ee_id) + np.array([0.0, -0.1, 0.0])
    err_before = np.linalg.norm(target_right - _site_pos(env, env._right_ee_id))

    action = np.concatenate([
        _site_pos(env, env._left_ee_id),
        _site_aa(env, env._left_ee_id),
        [0.0],
        target_right,
        _site_aa(env, env._right_ee_id),
        [0.0],
    ])
    for _ in range(10):
        env.step(action)

    err_after = np.linalg.norm(target_right - _site_pos(env, env._right_ee_id))
    assert err_after < err_before
