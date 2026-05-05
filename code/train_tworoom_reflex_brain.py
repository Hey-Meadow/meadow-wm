"""Train a first TwoRoom Reflex Physics Brain prototype.

This is a standalone MLX testbed for the architecture described in
MEADOW_REFLEX_PHYSICS_BRAIN_DEVNOTE_2026-04-22.md.

The key constraint is intentional: hidden physical parameters are NOT given to
the network. The network receives a short interaction history and must infer a
latent z_phys that helps predict the next state under a new action.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


WORLD = 224.0
WALL_X = 112.0
WALL_HALF = 4.0
DOOR_CENTER = 98.0
DOOR_HALF = 22.0
AGENT_R = 5.4
MAX_SPEED = 5.0

STATE_DIM = 6          # internal simulator state: x, y, vx, vy, goal_x, goal_y
OBS_DIM = 4            # model observation: x, y, goal_x, goal_y
ACTION_DIM = 2
PRED_DIM = 2           # next normalized x, y delta
HIST_STEPS = 5
HIST_FEAT_DIM = OBS_DIM + ACTION_DIM + PRED_DIM
HIST_DIM = HIST_STEPS * HIST_FEAT_DIM
Z_PHYS_DIM = 64


@dataclass
class HiddenPhysics:
    mass: float
    damping: float
    friction: float
    actuator_gain: float
    door_drag: float
    wind_x: float
    wind_y: float
    bounce: float


def sample_physics(rng: np.random.Generator) -> HiddenPhysics:
    return HiddenPhysics(
        mass=float(rng.uniform(1.0, 3.2)),
        damping=float(rng.uniform(0.03, 0.16)),
        friction=float(rng.uniform(0.00, 0.10)),
        actuator_gain=float(rng.uniform(0.55, 1.15)),
        door_drag=float(rng.uniform(0.50, 0.92)),
        wind_x=float(rng.uniform(-0.035, 0.035)),
        wind_y=float(rng.uniform(-0.035, 0.035)),
        bounce=float(rng.uniform(0.45, 0.88)),
    )


def in_door(y: float) -> bool:
    return DOOR_CENTER - DOOR_HALF <= y <= DOOR_CENTER + DOOR_HALF


def clip_speed(v: np.ndarray, max_speed: float = MAX_SPEED) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > max_speed:
        return v / n * max_speed
    return v


def step_state(state: np.ndarray, action: np.ndarray, p: HiddenPhysics) -> np.ndarray:
    pos = state[:2].copy()
    vel = state[2:4].copy()
    goal = state[4:6].copy()
    act = np.clip(action.astype(np.float32), -1.0, 1.0)

    speed = float(np.linalg.norm(vel))
    if speed > 1e-6:
        friction = p.friction * vel / speed
    else:
        friction = np.zeros(2, dtype=np.float32)

    force = p.actuator_gain * act + np.array([p.wind_x, p.wind_y], dtype=np.float32)
    accel = (force - p.damping * vel - friction) / max(p.mass, 1e-6)
    vel = clip_speed(vel + accel).astype(np.float32)
    nxt = pos + vel

    # Border bounce.
    lo = 18.0 + AGENT_R
    hi = WORLD - 18.0 - AGENT_R
    for axis in (0, 1):
        if nxt[axis] < lo:
            nxt[axis] = lo
            vel[axis] = abs(vel[axis]) * p.bounce
        elif nxt[axis] > hi:
            nxt[axis] = hi
            vel[axis] = -abs(vel[axis]) * p.bounce

    # Middle wall bounce or door drag.
    crossed = (pos[0] - WALL_X) * (nxt[0] - WALL_X) <= 0
    near_wall = abs(nxt[0] - WALL_X) < WALL_HALF + AGENT_R
    if crossed or near_wall:
        if in_door(float(nxt[1])):
            vel *= p.door_drag
            nxt = pos + vel
        else:
            if pos[0] < WALL_X:
                nxt[0] = WALL_X - WALL_HALF - AGENT_R
                vel[0] = -abs(vel[0]) * p.bounce
            else:
                nxt[0] = WALL_X + WALL_HALF + AGENT_R
                vel[0] = abs(vel[0]) * p.bounce

    out = state.copy()
    out[:2] = nxt.astype(np.float32)
    out[2:4] = vel.astype(np.float32)
    out[4:6] = goal
    return out.astype(np.float32)


def normalize_state(s: np.ndarray) -> np.ndarray:
    out = s.astype(np.float32).copy()
    out[[0, 1, 4, 5]] = out[[0, 1, 4, 5]] / WORLD - 0.5
    out[2:4] = out[2:4] / MAX_SPEED
    return out


def normalize_obs(s: np.ndarray) -> np.ndarray:
    return np.array([
        s[0] / WORLD - 0.5,
        s[1] / WORLD - 0.5,
        s[4] / WORLD - 0.5,
        s[5] / WORLD - 0.5,
    ], dtype=np.float32)


def action_policy(state: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
    pos = state[:2]
    goal = state[4:6]
    if t < 7 or rng.random() < 0.35:
        a = rng.uniform(-1.0, 1.0, size=2)
    else:
        waypoint = np.array([WALL_X, DOOR_CENTER], dtype=np.float32)
        if pos[0] > WALL_X + 12.0:
            waypoint = goal
        d = waypoint - pos
        n = float(np.linalg.norm(d))
        a = d / max(n, 1e-6) + rng.normal(0.0, 0.25, size=2)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def make_dataset(n_episodes: int, ep_len: int, seed: int):
    rng = np.random.default_rng(seed)
    histories = []
    currents = []
    actions = []
    targets = []
    meta = []

    for ep in range(n_episodes):
        phys = sample_physics(rng)
        state = np.array([
            rng.uniform(32.0, 72.0),
            rng.uniform(132.0, 184.0),
            0.0,
            0.0,
            rng.uniform(166.0, 194.0),
            rng.uniform(42.0, 76.0),
        ], dtype=np.float32)
        trans_feats = []
        for t in range(ep_len):
            action = action_policy(state, t, rng)
            next_state = step_state(state, action, phys)
            obs_n = normalize_obs(state)
            next_obs_n = normalize_obs(next_state)
            delta = (next_obs_n[:2] - obs_n[:2]).astype(np.float32)
            trans_feats.append(np.concatenate([obs_n, action, delta], dtype=np.float32))

            if t >= HIST_STEPS:
                hist = np.concatenate(trans_feats[t - HIST_STEPS:t], dtype=np.float32)
                histories.append(hist)
                currents.append(obs_n)
                actions.append(action)
                targets.append(delta)
                meta.append([ep, t, phys.mass, phys.damping, phys.actuator_gain,
                             phys.wind_x, phys.wind_y])
            state = next_state

    return {
        "history": np.asarray(histories, dtype=np.float32),
        "current": np.asarray(currents, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "target_delta": np.asarray(targets, dtype=np.float32),
        "meta": np.asarray(meta, dtype=np.float32),
    }


class ReflexPhysicsBrain(nn.Module):
    def __init__(self, hidden: int = 256, z_dim: int = Z_PHYS_DIM):
        super().__init__()
        self.phys_encoder = nn.Sequential(
            nn.Linear(HIST_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(OBS_DIM + ACTION_DIM + z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, PRED_DIM),
        )

    def __call__(self, history, current, action):
        z = self.phys_encoder(history)
        x = mx.concatenate([current, action, z], axis=-1)
        return self.dynamics(x)


def mse(a, b):
    return mx.mean((a - b) ** 2)


def eval_loss(model, data, idx_np):
    pred = model(mx.array(data["history"][idx_np]),
                 mx.array(data["current"][idx_np]),
                 mx.array(data["action"][idx_np]))
    loss = mse(pred, mx.array(data["target_delta"][idx_np]))
    mx.eval(loss)
    return float(loss.item())


def cold_loss(model, data, idx_np):
    hist = np.zeros_like(data["history"][idx_np])
    pred = model(mx.array(hist),
                 mx.array(data["current"][idx_np]),
                 mx.array(data["action"][idx_np]))
    loss = mse(pred, mx.array(data["target_delta"][idx_np]))
    mx.eval(loss)
    return float(loss.item())


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args) | {
        "state_dim_internal": STATE_DIM,
        "obs_dim_model": OBS_DIM,
        "history_dim": HIST_DIM,
        "z_phys_dim": Z_PHYS_DIM,
        "hardware_note": "MLX GPU full training; ANE not used",
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("[device]", mx.default_device())
    print("[data] generating train/val rollouts...")
    t_data = time.time()
    train_data = make_dataset(args.train_episodes, args.ep_len, args.seed)
    val_data = make_dataset(args.val_episodes, args.ep_len, args.seed + 999)
    print(f"[data] train={len(train_data['history'])} val={len(val_data['history'])} "
          f"elapsed={time.time() - t_data:.1f}s")

    np.savez_compressed(os.path.join(args.out_dir, "train_samples.npz"), **train_data)
    np.savez_compressed(os.path.join(args.out_dir, "val_samples.npz"), **val_data)

    model = ReflexPhysicsBrain(hidden=args.hidden, z_dim=args.z_dim)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)

    def loss_fn(m, h, c, a, y):
        pred = m(h, c, a)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)
    n_train = len(train_data["history"])
    n_val = len(val_data["history"])
    log = {
        "iters": [],
        "train_loss": [],
        "val_loss_warm": [],
        "val_loss_cold_zero_history": [],
        "ms_per_step": [],
    }
    t0 = time.time()
    last_report_t = time.time()
    for it in range(args.iters):
        idx = rng.integers(0, n_train, size=args.batch)
        h = mx.array(train_data["history"][idx])
        c = mx.array(train_data["current"][idx])
        a = mx.array(train_data["action"][idx])
        y = mx.array(train_data["target_delta"][idx])

        step_t = time.time()
        loss, grads = grad_fn(model, h, c, a, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        step_ms = (time.time() - step_t) * 1000.0

        if it % args.log_every == 0 or it == args.iters - 1:
            vidx = rng.choice(n_val, size=min(args.eval_batch, n_val), replace=False)
            warm = eval_loss(model, val_data, vidx)
            cold = cold_loss(model, val_data, vidx)
            train_l = float(loss.item())
            dt = time.time() - last_report_t
            last_report_t = time.time()
            print(f"[{it:5d}] train={train_l:.6f} val_warm={warm:.6f} "
                  f"val_cold={cold:.6f} step={step_ms:.2f}ms "
                  f"elapsed={time.time()-t0:.1f}s report_dt={dt:.1f}s")
            log["iters"].append(it)
            log["train_loss"].append(train_l)
            log["val_loss_warm"].append(warm)
            log["val_loss_cold_zero_history"].append(cold)
            log["ms_per_step"].append(step_ms)

    ckpt = os.path.join(args.out_dir, "tworoom_reflex_brain.npz")
    mx.savez(ckpt, **dict(tree_flatten(model.parameters())))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)

    summary = {
        "ckpt": ckpt,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "final_train_loss": log["train_loss"][-1],
        "final_val_loss_warm": log["val_loss_warm"][-1],
        "final_val_loss_cold_zero_history": log["val_loss_cold_zero_history"][-1],
        "warm_vs_cold_ratio": log["val_loss_warm"][-1] / max(log["val_loss_cold_zero_history"][-1], 1e-12),
        "median_reported_ms_per_step": float(np.median(log["ms_per_step"])),
        "real_training": True,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[save]", ckpt)
    print("[summary]", json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_episodes", type=int, default=900)
    ap.add_argument("--val_episodes", type=int, default=180)
    ap.add_argument("--ep_len", type=int, default=80)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=Z_PHYS_DIM)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--out_dir", default="meadow/tworoom_reflex_brain")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
