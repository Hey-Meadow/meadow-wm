"""TwoRoom causal student v2 with shared physics/engine backbone."""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import record_tworoom_reflex_brain_video as viz
from meadow_student_backbone import (
    LossWeights,
    SharedPhysicsStudent,
    batch_task_vector,
    eval_components,
    load_model,
    loss_fn,
    pad_history,
    save_model,
)
from train_tworoom_causal_student import (
    MICRO_RADIUS,
    SLOW_RADIUS,
    STOP_RADIUS,
    SUCCESS_RADIUS,
    inertia_brake_action,
    micro_align_action,
    random_layout,
    rollout_start_state,
    select_teacher_path,
    set_layout,
    teacher_action,
    waypoint_for_state,
)
from train_tworoom_reflex_brain import HiddenPhysics, normalize_obs, sample_physics


DATASET_VERSION = "tworoom_causal_tree_student_v2_backbone"
TASK_NAME = "tworoom"
SCENE_DIM = 20
ACTION_DIM = 2
HIST_STEPS = 4
HIST_ROW_DIM = 8


def scene_feature(layout, state):
    door_center, door_half, _, _, _, start_y = layout
    waypoint, phase = waypoint_for_state(layout, state)
    wp_delta = (waypoint - state[:2]) / viz.WORLD
    goal_delta = (state[4:6] - state[:2]) / viz.WORLD
    wp_dist = float(np.linalg.norm(waypoint - state[:2]))
    goal_dist = float(np.linalg.norm(state[4:6] - state[:2]))
    speed = float(np.linalg.norm(state[2:4]))
    wp_dir = (waypoint - state[:2]) / max(wp_dist, 1e-6)
    progress_speed = float(np.dot(state[2:4], wp_dir))
    return np.array(
        [
            state[0] / viz.WORLD - 0.5,
            state[1] / viz.WORLD - 0.5,
            state[2] / viz.MAX_SPEED,
            state[3] / viz.MAX_SPEED,
            state[4] / viz.WORLD - 0.5,
            state[5] / viz.WORLD - 0.5,
            door_center / viz.WORLD - 0.5,
            door_half / 48.0,
            start_y / viz.WORLD - 0.5,
            float(state[0] > viz.WALL_X),
            wp_delta[0],
            wp_delta[1],
            phase[0],
            phase[1],
            phase[2],
            float(abs(state[1] - door_center) <= max(1.0, door_half - viz.AGENT_R)),
            goal_delta[0],
            goal_delta[1],
            speed / viz.MAX_SPEED,
            progress_speed / viz.MAX_SPEED,
        ],
        dtype=np.float32,
    )


def history_row(prev_state, action, next_state):
    obs = normalize_obs(prev_state)
    next_obs = normalize_obs(next_state)
    delta = (next_obs[:2] - obs[:2]).astype(np.float32)
    return np.concatenate([obs, action.astype(np.float32), delta], axis=0).astype(np.float32)


def guide_action(layout, state):
    target, phase = waypoint_for_state(layout, state)
    desired = viz.steer_to(state[:2], target, strength=1.0)
    return inertia_brake_action(state, desired, target, enabled=bool(phase[2] > 0.5 or np.linalg.norm(target - state[:2]) < SLOW_RADIUS))


def phys_targets(phys: HiddenPhysics):
    phys_vec = np.array(
        [
            phys.mass / 3.5,
            0.0,
            phys.damping / 0.20,
            0.0,
            phys.friction / 0.15,
            phys.bounce,
            phys.actuator_gain / 1.20,
            phys.door_drag,
            phys.wind_x / 0.05,
            phys.wind_y / 0.05,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    engine_vec = np.array(
        [
            phys.mass,
            phys.mass / 2.0,
            phys.damping * 8.0,
            0.0,
            phys.friction,
            phys.bounce,
            phys.wind_x * 280.0,
            phys.wind_y * 280.0,
            1.0,
            0.05 + phys.friction * 0.2,
            0.0,
            phys.door_drag,
        ],
        dtype=np.float32,
    )
    return phys_vec, engine_vec


def build_dataset(n_layouts, candidates, max_steps, seed, augment_states=10):
    rng = np.random.default_rng(seed)
    scenes, histories, actions, phys_rows, engine_rows = [], [], [], [], []
    layouts = []
    teacher_success = []
    task = batch_task_vector(TASK_NAME, 1)[0]
    for lid in range(n_layouts):
        layout = random_layout(rng)
        phys = sample_physics(rng)
        phys_row, engine_row = phys_targets(phys)
        set_layout(*layout)
        state, _ = rollout_start_state(phys)
        path, ok, _ = select_teacher_path(state, phys, candidates)
        hist_rows: list[np.ndarray] = []
        layouts.append(
            {
                "door_center": layout[0],
                "door_half": layout[1],
                "goal_x": layout[2],
                "goal_y": layout[3],
                "start_x": layout[4],
                "start_y": layout[5],
                "phys": {
                    "mass": phys.mass,
                    "damping": phys.damping,
                    "friction": phys.friction,
                    "actuator_gain": phys.actuator_gain,
                    "door_drag": phys.door_drag,
                    "wind_x": phys.wind_x,
                    "wind_y": phys.wind_y,
                    "bounce": phys.bounce,
                },
            }
        )
        teacher_success.append(int(ok))
        for _ in range(max_steps):
            if np.linalg.norm(state[:2] - state[4:6]) <= SUCCESS_RADIUS:
                break
            teacher_a = teacher_action(layout, state, path)
            base_a = guide_action(layout, state)
            scenes.append(scene_feature(layout, state))
            histories.append(pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM))
            actions.append(np.clip(teacher_a - base_a, -1.0, 1.0).astype(np.float32))
            phys_rows.append(phys_row)
            engine_rows.append(engine_row)
            next_state, collided = viz.step_state_stop_on_collision(state, teacher_a, phys)
            hist_rows.append(history_row(state, teacher_a, next_state))
            state = next_state
            if collided:
                path, _, _ = select_teacher_path(state, phys, candidates)
        for _ in range(augment_states):
            aug_state = np.array(
                [
                    rng.uniform(28.0, viz.WORLD - 28.0),
                    rng.uniform(28.0, viz.WORLD - 28.0),
                    rng.uniform(-0.5, 0.5) * viz.MAX_SPEED,
                    rng.uniform(-0.5, 0.5) * viz.MAX_SPEED,
                    layout[2],
                    layout[3],
                ],
                dtype=np.float32,
            )
            scenes.append(scene_feature(layout, aug_state))
            histories.append(np.zeros(HIST_STEPS * HIST_ROW_DIM, dtype=np.float32))
            actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
            phys_rows.append(phys_row)
            engine_rows.append(engine_row)
    return {
        "scene": np.asarray(scenes, dtype=np.float32),
        "history": np.asarray(histories, dtype=np.float32),
        "task": np.repeat(task[None, :], len(actions), axis=0).astype(np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "phys": np.asarray(phys_rows, dtype=np.float32),
        "engine": np.asarray(engine_rows, dtype=np.float32),
        "layouts": layouts,
        "teacher_success_rate": float(np.mean(teacher_success)) if teacher_success else 0.0,
    }


def rollout_student(model, layout, phys, max_steps=140):
    set_layout(*layout)
    state, _ = rollout_start_state(phys)
    hist_rows: list[np.ndarray] = []
    task = mx.array(batch_task_vector(TASK_NAME, 1))
    best_dist = float(np.linalg.norm(state[:2] - state[4:6]))
    hit_step = None
    settle_count = 0
    for t in range(max_steps + 72):
        dist = float(np.linalg.norm(state[:2] - state[4:6]))
        if hit_step is None and dist <= SUCCESS_RADIUS:
            hit_step = t
        if hit_step is not None:
            speed = float(np.linalg.norm(state[2:4]))
            if settle_count >= 24 and dist <= MICRO_RADIUS and speed < 0.06:
                break
            if settle_count >= 72:
                break
            a = micro_align_action(state)
            settle_count += 1
        else:
            scene = mx.array(scene_feature(layout, state)[None])
            history = mx.array(pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM)[None])
            action, _, _ = model(scene, history, task)
            mx.eval(action)
            residual = np.array(action, dtype=np.float32)[0]
            base_a = guide_action(layout, state)
            a = viz.clip_action(base_a + 0.75 * residual)
        next_state, _ = viz.step_state_stop_on_collision(state, a, phys)
        hist_rows.append(history_row(state, a, next_state))
        state = next_state
        best_dist = min(best_dist, float(np.linalg.norm(state[:2] - state[4:6])))
    final_dist = float(np.linalg.norm(state[:2] - state[4:6]))
    return {
        "success": bool(hit_step is not None or final_dist <= SUCCESS_RADIUS or best_dist <= SUCCESS_RADIUS),
        "final_dist": final_dist,
        "best_dist": best_dist,
    }


def evaluate(model, n_layouts, seed, max_steps):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_layouts):
        rows.append(rollout_student(model, random_layout(rng), sample_physics(rng), max_steps=max_steps))
    return {
        "success": int(sum(r["success"] for r in rows)),
        "n": int(len(rows)),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "final_dist_mean": float(np.mean([r["final_dist"] for r in rows])),
        "best_dist_median": float(np.median([r["best_dist"] for r in rows])),
    }


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args) | {
        "dataset_version": DATASET_VERSION,
        "task": TASK_NAME,
        "history_steps": HIST_STEPS,
        "history_row_dim": HIST_ROW_DIM,
        "teacher": "TwoRoom causal-tree teacher with varying hidden physics",
        "student": "shared physics/engine backbone student",
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    train_data = build_dataset(args.train_layouts, args.candidates, args.max_steps, args.seed, args.augment_states)
    val_data = build_dataset(args.val_layouts, args.candidates, args.max_steps, args.seed + 1000, args.augment_states)
    np.savez_compressed(
        os.path.join(args.out_dir, "train_samples.npz"),
        scene=train_data["scene"],
        history=train_data["history"],
        task=train_data["task"],
        action=train_data["action"],
        phys=train_data["phys"],
        engine=train_data["engine"],
    )
    np.savez_compressed(
        os.path.join(args.out_dir, "val_samples.npz"),
        scene=val_data["scene"],
        history=val_data["history"],
        task=val_data["task"],
        action=val_data["action"],
        phys=val_data["phys"],
        engine=val_data["engine"],
    )
    with open(os.path.join(args.out_dir, "train_layouts.json"), "w") as f:
        json.dump(train_data["layouts"], f, indent=2)
    with open(os.path.join(args.out_dir, "val_layouts.json"), "w") as f:
        json.dump(val_data["layouts"], f, indent=2)
    print(
        f"[data] train={len(train_data['action'])} val={len(val_data['action'])} "
        f"teacher_train={train_data['teacher_success_rate']:.3f} "
        f"teacher_val={val_data['teacher_success_rate']:.3f} elapsed={time.time()-t0:.1f}s"
    )

    model = SharedPhysicsStudent(
        scene_dim=SCENE_DIM,
        history_dim=HIST_STEPS * HIST_ROW_DIM,
        action_dim=ACTION_DIM,
        hidden=args.hidden,
        latent=args.latent,
    )
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    weights = LossWeights(action=1.0, phys=args.phys_weight, engine=args.engine_weight)
    grad_fn = nn.value_and_grad(model, lambda m, s, h, t, a, p, e: loss_fn(m, s, h, t, a, p, e, weights))
    rng = np.random.default_rng(args.seed)
    n_train = len(train_data["action"])
    n_val = len(val_data["action"])
    log = []
    for it in range(args.iters):
        idx = rng.integers(0, n_train, size=args.batch)
        loss, grads = grad_fn(
            model,
            mx.array(train_data["scene"][idx]),
            mx.array(train_data["history"][idx]),
            mx.array(train_data["task"][idx]),
            mx.array(train_data["action"][idx]),
            mx.array(train_data["phys"][idx]),
            mx.array(train_data["engine"][idx]),
        )
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if it % args.log_every == 0 or it == args.iters - 1:
            vidx = rng.choice(n_val, size=min(args.eval_batch, n_val), replace=False)
            parts = eval_components(
                model,
                mx.array(val_data["scene"][vidx]),
                mx.array(val_data["history"][vidx]),
                mx.array(val_data["task"][vidx]),
                mx.array(val_data["action"][vidx]),
                mx.array(val_data["phys"][vidx]),
                mx.array(val_data["engine"][vidx]),
            )
            row = {"iter": it, "train_loss": float(loss.item()), **parts}
            log.append(row)
            print(
                f"[{it:5d}] train={row['train_loss']:.6f} "
                f"act={row['action_loss']:.6f} phys={row['phys_loss']:.6f} "
                f"eng={row['engine_loss']:.6f}"
            )

    ckpt = os.path.join(args.out_dir, "tworoom_causal_student_v2.npz")
    save_model(model, ckpt)
    eval_row = evaluate(model, args.eval_layouts, args.seed + 2000, args.max_steps)
    summary = {
        "ckpt": ckpt,
        "dataset_version": DATASET_VERSION,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "teacher_train_success_rate": train_data["teacher_success_rate"],
        "teacher_val_success_rate": val_data["teacher_success_rate"],
        "student_eval": eval_row,
        "final": log[-1],
        "real_training": True,
        "shared_backbone": True,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_layouts", type=int, default=220)
    ap.add_argument("--val_layouts", type=int, default=60)
    ap.add_argument("--eval_layouts", type=int, default=80)
    ap.add_argument("--candidates", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=140)
    ap.add_argument("--augment_states", type=int, default=18)
    ap.add_argument("--iters", type=int, default=1400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--latent", type=int, default=160)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--phys_weight", type=float, default=0.20)
    ap.add_argument("--engine_weight", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out_dir", default="meadow/tworoom_causal_student_v2")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
