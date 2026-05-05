"""Distill a real OGBench Cube controller into a shared-backbone student."""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from meadow_student_backbone import (
    LossWeights,
    SharedPhysicsStudent,
    batch_task_vector,
    eval_components,
    loss_fn,
    pad_history,
    save_model,
)
from train_ogbench_cube_neural_causal_scorer import DATASET_NAME, SUCCESS_POS


DATASET_VERSION = "ogbench_cube_causal_student_v1_backbone"
TASK_NAME = "ogbench_cube"
SCENE_DIM = 21
ACTION_DIM = 5
HIST_STEPS = 4
HIST_ROW_DIM = 19


def scene_feature(info_now, goal):
    eff = np.asarray(info_now["proprio/effector_pos"], dtype=np.float32)
    yaw = float(np.asarray(info_now["proprio/effector_yaw"]).reshape(-1)[0])
    grip = float(np.asarray(info_now["proprio/gripper_opening"]).reshape(-1)[0])
    contact = float(np.asarray(info_now["proprio/gripper_contact"]).reshape(-1)[0])
    block = np.asarray(info_now["privileged/block_0_pos"], dtype=np.float32)
    block_yaw = float(np.asarray(info_now["privileged/block_0_yaw"]).reshape(-1)[0])
    eff_to_block = block - eff
    block_to_goal = goal - block
    return np.concatenate(
        [
            eff / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32),
            np.array([grip, contact], dtype=np.float32),
            block / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            np.array([np.cos(block_yaw), np.sin(block_yaw)], dtype=np.float32),
            goal / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            eff_to_block / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            block_to_goal / np.array([0.6, 0.6, 0.4], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def history_row(info_now, action, next_info):
    eff = np.asarray(info_now["proprio/effector_pos"], dtype=np.float32)
    block = np.asarray(info_now["privileged/block_0_pos"], dtype=np.float32)
    next_eff = np.asarray(next_info["proprio/effector_pos"], dtype=np.float32)
    next_block = np.asarray(next_info["privileged/block_0_pos"], dtype=np.float32)
    grip = float(np.asarray(info_now["proprio/gripper_opening"]).reshape(-1)[0])
    contact = float(np.asarray(info_now["proprio/gripper_contact"]).reshape(-1)[0])
    return np.concatenate(
        [
            eff / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            block / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            action.astype(np.float32),
            (next_eff - eff) / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            (next_block - block) / np.array([0.6, 0.6, 0.4], dtype=np.float32),
            np.array([grip, contact], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def phys_targets(env):
    model = env.unwrapped.model
    dof_damping = np.asarray(model.dof_damping, dtype=np.float32)
    dof_frictionloss = np.asarray(model.dof_frictionloss, dtype=np.float32)
    body_mass = np.asarray(model.body_mass, dtype=np.float32)
    body_inertia = np.asarray(model.body_inertia, dtype=np.float32)
    geom_friction = np.asarray(model.geom_friction, dtype=np.float32)
    geom_solref = np.asarray(model.geom_solref, dtype=np.float32)
    geom_solimp = np.asarray(model.geom_solimp, dtype=np.float32)
    phys_vec = np.array(
        [
            float(body_mass.sum()),
            float(body_inertia.sum()) * 1e2,
            float(dof_damping.mean()),
            float(dof_frictionloss.mean()),
            float(geom_friction[:, 0].mean()),
            float(geom_solref[:, 0].mean()),
            1.0,
            1.0,
            0.0,
            0.0,
            float(geom_solimp[:, 0].mean()),
            float(geom_solimp[:, 1].mean()),
        ],
        dtype=np.float32,
    )
    engine_vec = np.array(
        [
            float(body_mass.sum()),
            1.0,
            float(dof_damping.mean()) * 10.0,
            float(dof_frictionloss.mean()) * 10.0,
            float(geom_friction[:, 0].mean()),
            float(geom_solref[:, 0].mean()),
            0.0,
            0.0,
            float(body_inertia.sum()) * 1e2,
            0.02,
            45.0,
            float(geom_solimp[:, 0].mean()) * 10.0,
        ],
        dtype=np.float32,
    )
    return phys_vec, engine_vec


def oracle_info(info_now, goal):
    out = dict(info_now)
    out["privileged/target_block"] = 0
    out["privileged/target_block_pos"] = np.asarray(goal, dtype=np.float32).copy()
    out["privileged/target_block_yaw"] = np.array([0.0], dtype=np.float32)
    return out


def collect_cases(case_count, seed, dataset_name):
    import ogbench
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    rows = {k: [] for k in ("scene", "history", "task", "action", "phys", "engine")}
    task_rows = []
    cases = []
    for i in range(case_count):
        task_id = i % 5 + 1
        ob, info = env.reset(seed=seed + i, options={"task_id": task_id, "render_goal": True})
        goal = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][0], dtype=np.float32)
        oracle = CubeMarkovOracle(env=env)
        oracle.reset(ob, oracle_info(info, goal))
        phys_row, engine_row = phys_targets(env)
        task_vec = batch_task_vector(TASK_NAME, 1)[0]
        hist_rows: list[np.ndarray] = []
        steps = 0
        while steps < 140:
            info_now = env.unwrapped.compute_ob_info()
            action = np.asarray(oracle.select_action(ob, oracle_info(info_now, goal)), dtype=np.float32)
            rows["scene"].append(scene_feature(info_now, goal))
            rows["history"].append(pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM))
            rows["task"].append(task_vec)
            rows["action"].append(action)
            rows["phys"].append(phys_row)
            rows["engine"].append(engine_row)
            ob, _, _, _, _ = env.step(action)
            next_info = env.unwrapped.compute_ob_info()
            hist_rows.append(history_row(info_now, action, next_info))
            steps += 1
            if oracle.done:
                break
        cases.append({"task_id": int(task_id), "seed": int(seed + i), "steps": int(steps)})
    env.close()
    for key in rows:
        rows[key] = np.asarray(rows[key], dtype=np.float32)
    rows["cases"] = cases
    return rows


def run_student_case(model, seed, task_id, dataset_name):
    import ogbench

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    ob, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
    goal = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][0], dtype=np.float32)
    task = mx.array(batch_task_vector(TASK_NAME, 1))
    hist_rows: list[np.ndarray] = []
    steps = 0
    try:
        while steps < 140:
            info_now = env.unwrapped.compute_ob_info()
            block = np.asarray(info_now["privileged/block_0_pos"], dtype=np.float32)
            if float(np.linalg.norm(block - goal)) <= SUCCESS_POS:
                break
            scene = mx.array(scene_feature(info_now, goal)[None])
            history = mx.array(pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM)[None])
            action, _, _ = model(scene, history, task)
            mx.eval(action)
            act = np.array(action, dtype=np.float32)[0]
            if float(np.linalg.norm(block - goal)) < 0.03:
                act = act * 0.30
            ob, _, _, _, _ = env.step(np.clip(act, -1.0, 1.0))
            next_info = env.unwrapped.compute_ob_info()
            hist_rows.append(history_row(info_now, act, next_info))
            steps += 1
        final_info = env.unwrapped.compute_ob_info()
        final_block = np.asarray(final_info["privileged/block_0_pos"], dtype=np.float32)
        final_dist = float(np.linalg.norm(final_block - goal))
        max_lift = float(np.max([row[2] for row in [np.asarray(final_info["privileged/block_0_pos"], dtype=np.float32)]]))
        return {
            "success": bool(final_dist <= SUCCESS_POS),
            "final_dist": final_dist,
            "steps": int(steps),
            "max_cube_z": max_lift,
        }
    finally:
        env.close()


def evaluate(model, cases, dataset_name):
    rows = [run_student_case(model, row["seed"], row["task_id"], dataset_name) for row in cases]
    return {
        "success": int(sum(r["success"] for r in rows)),
        "n": int(len(rows)),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "final_dist_mean": float(np.mean([r["final_dist"] for r in rows])),
        "final_dist_median": float(np.median([r["final_dist"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
    }


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args) | {
        "dataset_version": DATASET_VERSION,
        "task": TASK_NAME,
        "history_steps": HIST_STEPS,
        "teacher": "OGBench CubeMarkovOracle",
        "student": "shared physics/engine backbone student",
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    train_data = collect_cases(args.train_cases, args.seed, args.dataset_name)
    val_data = collect_cases(args.val_cases, args.seed + 1000, args.dataset_name)
    np.savez_compressed(os.path.join(args.out_dir, "train_samples.npz"), **{k: train_data[k] for k in ("scene", "history", "task", "action", "phys", "engine")})
    np.savez_compressed(os.path.join(args.out_dir, "val_samples.npz"), **{k: val_data[k] for k in ("scene", "history", "task", "action", "phys", "engine")})
    with open(os.path.join(args.out_dir, "train_cases.json"), "w") as f:
        json.dump(train_data["cases"], f, indent=2)
    with open(os.path.join(args.out_dir, "val_cases.json"), "w") as f:
        json.dump(val_data["cases"], f, indent=2)
    print(f"[data] train={len(train_data['action'])} val={len(val_data['action'])} elapsed={time.time()-t0:.1f}s")

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
                f"act={row['action_loss']:.6f} phys={row['phys_loss']:.6f} eng={row['engine_loss']:.6f}"
            )

    ckpt = os.path.join(args.out_dir, "ogbench_cube_causal_student.npz")
    save_model(model, ckpt)
    eval_row = evaluate(model, val_data["cases"], args.dataset_name)
    summary = {
        "ckpt": ckpt,
        "dataset_version": DATASET_VERSION,
        "dataset_name": args.dataset_name,
        "train_cases": train_data["cases"],
        "val_cases": val_data["cases"],
        "train_samples": int(n_train),
        "val_samples": int(n_val),
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
    ap.add_argument("--out_dir", default="meadow/ogbench_cube_causal_student")
    ap.add_argument("--dataset_name", default=DATASET_NAME)
    ap.add_argument("--train_cases", type=int, default=24)
    ap.add_argument("--val_cases", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval_batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=224)
    ap.add_argument("--latent", type=int, default=160)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--phys_weight", type=float, default=0.20)
    ap.add_argument("--engine_weight", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
