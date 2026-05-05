"""Return-to-Go / augmented state translator for pushT.

Converts pushT (planning task) into LL-like reactive task by enriching the
raw 7D state with explicit planning signals:
  - goal_rel_xy     (2)  goal_block_xy - current_block_xy
  - goal_angle_sin/cos (2)  absolute goal block angle
  - block_angle_sin/cos (2)  absolute current block angle
  - time_remaining  (1)  (T_expected - t) / T_expected
  - t_norm          (1)  t / T_expected
  - prev_action     (2)  a_{t-1} (or 0 at t=0)
  - agent_to_block_xy (2)  relative
  - block_to_goal_dist (1)  scalar distance in pixel/512 units

Total: 15D augmented state vs 7D raw.

Output policy: simple MLP or diffusion head.
"""
from __future__ import annotations
import numpy as np

AUG_STATE_DIM = 15              # legacy with prev_action
AUG_STATE_DIM_NOPREV = 13       # without prev_action
AUG_STATE_DIM_DENSE = 16        # 13 + block_vel(2) + angle_closeness(1)
# 16 + per_axis_err(2) + angle_err_sincos(2) + align_flags(3) + total_score(1) = 24
AUG_STATE_DIM_3GRAV = 24         # 3-gravity design: x, y, theta each has own alignment signal
RAW_STATE_DIM = 7
# success thresholds (from swm/pusht env):
POS_THRESHOLD = 20.0   # pixels
ANGLE_THRESHOLD = float(np.pi / 9)  # ~20 deg


def _safe_angle(a: float) -> float:
    """Normalize radians to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def augment_state(raw_state: np.ndarray,
                  goal_state: np.ndarray,
                  t: int,
                  T_expected: int,
                  prev_action: np.ndarray | None = None,
                  prev_state: np.ndarray | None = None,
                  pos_scale: float = 512.0,
                  include_prev_action: bool = True,
                  include_dense_reward: bool = False,
                  include_3grav: bool = False) -> np.ndarray:
    """Build augmented state. Inputs are in pushT raw units (pixels, radians).

    raw_state:  (7,) = [agent_x, agent_y, block_x, block_y, block_theta, ?, ?]
    goal_state: (7,) same format (target configuration)
    t:          current step index
    T_expected: expected episode length (for normalization; 100 default)
    prev_action: (2,) previous action or None
    """
    assert raw_state.shape[-1] >= 7 and goal_state.shape[-1] >= 7
    s = raw_state
    g = goal_state

    agent_xy = s[:2] / pos_scale - 0.5                       # center 0
    block_xy = s[2:4] / pos_scale - 0.5
    block_theta = _safe_angle(float(s[4]))
    goal_block_xy = g[2:4] / pos_scale - 0.5
    goal_theta = _safe_angle(float(g[4]))

    goal_rel = goal_block_xy - block_xy                      # (2,) where block needs to go
    agent_to_block = block_xy - agent_xy                     # (2,)
    block_to_goal_dist = np.linalg.norm(goal_rel)            # scalar

    time_remaining = max(T_expected - t, 0) / T_expected     # 1 at start, 0 at end
    t_norm = min(t / T_expected, 1.0)                         # 0 at start, 1 at end

    prev_a = np.zeros(2, dtype=np.float32) if prev_action is None \
             else np.clip(prev_action, -1.0, 1.0)

    parts = [
        goal_rel,                                             # 2
        np.array([np.sin(goal_theta), np.cos(goal_theta)]),   # 2
        np.array([np.sin(block_theta), np.cos(block_theta)]), # 2
        np.array([time_remaining], dtype=np.float32),         # 1
        np.array([t_norm], dtype=np.float32),                 # 1
    ]
    if include_prev_action:
        parts.append(prev_a)                                  # 2
    parts.extend([
        agent_to_block,                                       # 2
        np.array([block_to_goal_dist], dtype=np.float32),     # 1
        agent_xy,                                             # 2
    ])
    if include_dense_reward or include_3grav:
        # velocity + angle closeness
        if prev_state is not None:
            prev_block_xy = prev_state[2:4] / pos_scale - 0.5
            block_vel = block_xy - prev_block_xy
        else:
            block_vel = np.zeros(2, dtype=np.float32)
        angle_closeness = float(np.cos(block_theta - goal_theta))
        parts.extend([
            block_vel.astype(np.float32),                     # 2
            np.array([angle_closeness], dtype=np.float32),    # 1
        ])
    if include_3grav:
        # 3-gravity design: X, Y, theta each has own alignment signal
        raw_pos_err_x = float(goal_state[2] - raw_state[2])  # pixel
        raw_pos_err_y = float(goal_state[3] - raw_state[3])
        raw_angle_err = _safe_angle(goal_theta - block_theta)
        # normalized per-axis errors
        pos_err_x = raw_pos_err_x / pos_scale
        pos_err_y = raw_pos_err_y / pos_scale
        # alignment flags (1 = within threshold, 0 = not)
        align_x = float(abs(raw_pos_err_x) < POS_THRESHOLD)
        align_y = float(abs(raw_pos_err_y) < POS_THRESHOLD)
        align_t = float(abs(raw_angle_err) < ANGLE_THRESHOLD)
        total_score = (align_x + align_y + align_t) / 3.0
        parts.extend([
            np.array([pos_err_x, pos_err_y], dtype=np.float32),                  # 2
            np.array([np.sin(raw_angle_err), np.cos(raw_angle_err)], dtype=np.float32),  # 2
            np.array([align_x, align_y, align_t], dtype=np.float32),             # 3
            np.array([total_score], dtype=np.float32),                           # 1
        ])
    aug = np.concatenate(parts, dtype=np.float32)
    # expected dim calculation
    exp_dim = AUG_STATE_DIM if include_prev_action else AUG_STATE_DIM_NOPREV
    if include_dense_reward or include_3grav:
        exp_dim += 3
    if include_3grav:
        exp_dim += 8
    assert aug.shape[0] == exp_dim, f"got {aug.shape[0]}, expected {exp_dim}"
    return aug


def augment_trajectory(states: np.ndarray,
                        actions: np.ndarray,
                        goal_state: np.ndarray,
                        T_expected: int | None = None,
                        include_prev_action: bool = True,
                        include_dense_reward: bool = False,
                        include_3grav: bool = False) -> np.ndarray:
    """Vectorized augmentation."""
    T = actions.shape[0]
    if T_expected is None:
        T_expected = T
    prev_acts = np.zeros((T, 2), dtype=np.float32)
    prev_acts[1:] = actions[:-1]
    prev_states = np.zeros_like(states[:T])
    prev_states[1:] = states[:T-1]
    prev_states[0] = states[0]
    dim = AUG_STATE_DIM if include_prev_action else AUG_STATE_DIM_NOPREV
    if include_dense_reward or include_3grav:
        dim += 3
    if include_3grav:
        dim += 8
    aug = np.empty((T, dim), dtype=np.float32)
    for t in range(T):
        aug[t] = augment_state(states[t], goal_state, t, T_expected,
                                prev_action=prev_acts[t],
                                prev_state=prev_states[t] if t > 0 else None,
                                include_prev_action=include_prev_action,
                                include_dense_reward=include_dense_reward,
                                include_3grav=include_3grav)
    return aug


# ── smoke test ──
if __name__ == "__main__":
    # Simulate 5-step trajectory
    T = 5
    raw_states = np.array([
        [100, 100, 200, 200, 0.5, 0, 0],
        [105, 105, 200, 200, 0.5, 0, 0],
        [110, 110, 205, 205, 0.6, 0, 0],
        [120, 110, 210, 210, 0.7, 0, 0],
        [130, 110, 220, 220, 0.8, 0, 0],
        [140, 110, 250, 250, 1.0, 0, 0],
    ], dtype=np.float32)
    actions = np.array([
        [0.5, 0.5], [0.5, 0.5], [0.3, 0.3], [0.2, 0.2], [0.1, 0.1],
    ], dtype=np.float32)
    goal = np.array([300, 300, 400, 400, 1.5, 0, 0], dtype=np.float32)

    aug = augment_trajectory(raw_states, actions, goal, T_expected=100)
    print(f"[smoke] aug shape: {aug.shape}")
    print(f"        first row:  {aug[0]}")
    print(f"        last row:   {aug[-1]}")
    print(f"        min/max: {aug.min():.3f} / {aug.max():.3f}")
    assert aug.shape == (T, AUG_STATE_DIM)
    print("OK")
