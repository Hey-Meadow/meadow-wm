"""2-link arm analytical IK + PD controller. Clean implementation."""
import numpy as np

L1 = L2 = 0.12            # dm_control reacher arm lengths
R_MAX = L1 + L2           # workspace outer radius


def forward_kin(q1, q2):
    """(q1, q2) → (finger_x, finger_y)."""
    fx = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    fy = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return fx, fy


def ik(tx, ty, elbow_up=True):
    """(target_x, target_y) → (q1*, q2*). Returns (nan, nan) if unreachable."""
    r2 = tx*tx + ty*ty
    if r2 > R_MAX * R_MAX * 0.999:     # clip just inside reachable
        scale = R_MAX * 0.999 / np.sqrt(r2)
        tx *= scale; ty *= scale; r2 = tx*tx + ty*ty
    c2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1 - c2*c2))
    if not elbow_up: s2 = -s2
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(ty, tx) - np.arctan2(L2*s2, L1 + L2*c2)
    return q1, q2


def ik_both(tx, ty):
    """Return both elbow branches as (q_up, q_down) tuples."""
    return ik(tx, ty, True), ik(tx, ty, False)


def pick_closest_branch(tx, ty, qpos_now, wrist_limit=2.7925):
    """Pick elbow branch with smaller joint-space travel.
    Shoulder: wrap distance. Wrist: direct distance (has limits, can't wrap)."""
    up = ik(tx, ty, True)
    down = ik(tx, ty, False)
    def wrap_d(a, b): return abs((a - b + np.pi) % (2*np.pi) - np.pi)
    def direct_d(a, b): return abs(np.clip(a, -wrist_limit, wrist_limit) - b)
    d_up = wrap_d(up[0], qpos_now[0]) + direct_d(up[1], qpos_now[1])
    d_down = wrap_d(down[0], qpos_now[0]) + direct_d(down[1], qpos_now[1])
    return up if d_up <= d_down else down


def pd_controller(q_star, qpos, qvel, k_p=100.0, k_v=10.0,
                    wrist_limit=2.7925):
    """PD torque to drive qpos → q_star.
    Shoulder (q1) is unconstrained → wrap error to shortest path.
    Wrist (q2) has limits [-wrist_limit, wrist_limit] → use direct error
        to avoid running into joint wall."""
    # shoulder (wrap OK)
    e1 = (q_star[0] - qpos[0] + np.pi) % (2*np.pi) - np.pi
    # wrist: clip q_star to reachable range first
    q2_tgt = np.clip(q_star[1], -wrist_limit, wrist_limit)
    e2 = q2_tgt - qpos[1]     # direct, no wrap — must respect limits
    torque = np.array([
        k_p * e1 - k_v * qvel[0],
        k_p * e2 - k_v * qvel[1],
    ], dtype=np.float32)
    return np.clip(torque, -1.0, 1.0)


def controller(info, k_p=100.0, k_v=10.0):
    """Full controller: target → torque."""
    tx, ty = info['target_pos']
    qpos = info['qpos']; qvel = info['qvel']
    q_star = pick_closest_branch(tx, ty, qpos)
    return pd_controller(q_star, qpos, qvel, k_p=k_p, k_v=k_v)


if __name__ == "__main__":
    # Self-test: IK should be consistent with forward kinematics
    np.random.seed(0)
    errs = []
    for _ in range(1000):
        # random reachable target
        r = np.random.uniform(0.02, R_MAX * 0.95)
        theta = np.random.uniform(0, 2*np.pi)
        tx, ty = r*np.cos(theta), r*np.sin(theta)
        for elbow in [True, False]:
            q1, q2 = ik(tx, ty, elbow_up=elbow)
            fx, fy = forward_kin(q1, q2)
            err = np.sqrt((fx-tx)**2 + (fy-ty)**2)
            errs.append(err)
    errs = np.array(errs)
    print(f"IK self-test: mean={errs.mean():.6f}  max={errs.max():.6f}")
    assert errs.max() < 1e-6, "IK formula inconsistent with forward kinematics!"
    print("✓ IK formula verified")
