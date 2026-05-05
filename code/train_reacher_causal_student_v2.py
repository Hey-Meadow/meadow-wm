"""Distill the Reacher causal-tree teacher into a chunked BC-style student."""

from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import stable_worldmodel.envs  # noqa: F401

from ik_reacher import pd_controller, pick_closest_branch
from meadow_student_backbone import (
    LossWeights,
    SharedPhysicsStudent,
    batch_task_vector,
    eval_components,
    loss_fn,
    pad_history,
    save_model,
)
from record_reacher_causal_tree import (
    ENV_ID,
    TARGET_THRESHOLD,
    candidate_action,
    current_info,
    plan_once,
    settle_action,
)


DATASET_VERSION = "reacher_causal_student_v2_chunked_bc"
TASK_NAME = "reacher"
SCENE_DIM = 20
ACTION_COMPONENT_DIM = 2
HIST_STEPS = 4
HIST_ROW_DIM = 12
PHASE_AGGRESSIVE = 0
PHASE_NOMINAL = 1
PHASE_TERMINAL = 2
PHASE_NAMES = ("aggressive", "nominal", "terminal")


def phase_index_from_dist(dist: float) -> int:
    if dist < 0.045:
        return PHASE_TERMINAL
    if dist > 0.085:
        return PHASE_AGGRESSIVE
    return PHASE_NOMINAL


def phase_one_hot(phase_idx: int) -> np.ndarray:
    one_hot = np.zeros(len(PHASE_NAMES), dtype=np.float32)
    one_hot[phase_idx] = 1.0
    return one_hot


def guide_action(info: dict[str, np.ndarray]) -> np.ndarray:
    target = info["target_pos"]
    q_star = pick_closest_branch(target[0], target[1], info["qpos"])
    action = pd_controller(q_star, info["qpos"], info["qvel"], k_p=58.0, k_v=9.0)
    dist = float(np.linalg.norm(info["finger_pos"] - info["target_pos"]))
    if dist < TARGET_THRESHOLD * 1.4:
        action = settle_action(info)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def scene_feature(info: dict[str, np.ndarray]) -> np.ndarray:
    qpos = info["qpos"].astype(np.float32)
    qvel = info["qvel"].astype(np.float32)
    finger = info["finger_pos"].astype(np.float32)
    target = info["target_pos"].astype(np.float32)
    delta = target - finger
    dist = float(np.linalg.norm(delta))
    phase_idx = phase_index_from_dist(dist)
    guide = guide_action(info)
    return np.concatenate(
        [
            np.sin(qpos).astype(np.float32),
            np.cos(qpos).astype(np.float32),
            qvel / 8.0,
            finger / 0.25,
            target / 0.25,
            delta / 0.25,
            np.array([dist / 0.25], dtype=np.float32),
            guide,
            phase_one_hot(phase_idx),
            np.array(
                [
                    np.linalg.norm(qvel) / 8.0,
                    np.linalg.norm(guide),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)


def history_row(prev_info: dict[str, np.ndarray], action: np.ndarray, next_info: dict[str, np.ndarray]) -> np.ndarray:
    prev_qpos = prev_info["qpos"].astype(np.float32)
    prev_qvel = prev_info["qvel"].astype(np.float32)
    prev_delta = prev_info["target_pos"] - prev_info["finger_pos"]
    next_delta = next_info["target_pos"] - next_info["finger_pos"]
    finger_delta = (next_info["finger_pos"] - prev_info["finger_pos"]) / 0.25
    dist_delta = (np.linalg.norm(next_delta) - np.linalg.norm(prev_delta)) / 0.25
    finger_speed = np.linalg.norm(finger_delta)
    return np.concatenate(
        [
            np.sin(prev_qpos).astype(np.float32),
            np.cos(prev_qpos).astype(np.float32),
            prev_qvel / 8.0,
            action.astype(np.float32),
            finger_delta.astype(np.float32),
            np.array([dist_delta, finger_speed], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def phys_targets(env):
    model = env.unwrapped.env.physics.model
    damping = np.asarray(model.dof_damping, dtype=np.float32)
    frictionloss = np.asarray(model.dof_frictionloss, dtype=np.float32)
    body_mass = np.asarray(model.body_mass, dtype=np.float32)
    body_inertia = np.asarray(model.body_inertia, dtype=np.float32)
    geom_friction = np.asarray(model.geom_friction, dtype=np.float32)
    geom_solref = np.asarray(model.geom_solref, dtype=np.float32)
    geom_solimp = np.asarray(model.geom_solimp, dtype=np.float32)
    phys_vec = np.array(
        [
            float(body_mass.sum()),
            float(body_inertia.sum()) * 1e3,
            float(damping.mean()),
            0.0,
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
            float(damping.mean()) * 10.0,
            float(frictionloss.mean()) * 10.0,
            float(geom_friction[:, 0].mean()),
            float(geom_solref[:, 0].mean()),
            0.0,
            0.0,
            float(body_inertia.sum()) * 1e3,
            0.02,
            25.0,
            float(geom_solimp[:, 0].mean()) * 10.0,
        ],
        dtype=np.float32,
    )
    return phys_vec, engine_vec


def parse_csv_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_floats(spec: str, expected: int) -> list[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated values, got {len(values)} from {spec!r}")
    return values


def build_action_chunk(actions: list[np.ndarray], start_idx: int, chunk_horizon: int) -> np.ndarray:
    if not actions:
        raise RuntimeError("no teacher actions available to build chunk")
    chunk = []
    for offset in range(chunk_horizon):
        idx = min(start_idx + offset, len(actions) - 1)
        chunk.append(actions[idx].astype(np.float32))
    return np.concatenate(chunk, axis=0).astype(np.float32)


def sample_weight(dist: float, phase_idx: int, args: argparse.Namespace) -> float:
    weight = 1.0
    if phase_idx == PHASE_AGGRESSIVE:
        weight *= args.aggressive_weight
    if phase_idx == PHASE_TERMINAL:
        weight *= args.terminal_weight
    if dist < TARGET_THRESHOLD * args.micro_weight_radius:
        weight *= args.micro_weight
    return float(weight)


def rollout_teacher(seed: int, args: argparse.Namespace) -> dict[str, object]:
    live_env = gym.make(ENV_ID)
    oracle_env = gym.make(ENV_ID)
    live_env.reset(seed=seed)
    oracle_env.reset(seed=seed)
    task = batch_task_vector(TASK_NAME, 1)[0]
    hist_rows: list[np.ndarray] = []
    prev_commitment = None
    phys_row, engine_row = phys_targets(live_env)
    records: list[dict[str, object]] = []
    actions: list[np.ndarray] = []
    total_steps = 0
    stable_steps = 0
    best_dist = 9e9
    prev_finger = current_info(live_env)["finger_pos"].copy()
    try:
        while total_steps < args.max_steps:
            info = current_info(live_env)
            dist = float(np.linalg.norm(info["finger_pos"] - info["target_pos"]))
            finger_speed = float(np.linalg.norm(info["finger_pos"] - prev_finger))
            qvel_norm = float(np.linalg.norm(info["qvel"]))
            best_dist = min(best_dist, dist)
            if dist < TARGET_THRESHOLD and finger_speed < 0.0018 and qvel_norm < 0.20:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= 6:
                break

            planned = plan_once(
                live_env,
                oracle_env,
                oracle_horizon=args.oracle_horizon,
                top_k=args.top_k,
                prev_commitment=prev_commitment,
            )
            best = planned["best"]
            prev_commitment = planned["commitment"]
            local_exec_horizon = (
                2
                if planned["terminal_mode"]
                else (args.exec_horizon + 2 if planned["aggressive_mode"] else args.exec_horizon)
            )
            for plan_step in range(local_exec_horizon):
                info = current_info(live_env)
                dist = float(np.linalg.norm(info["finger_pos"] - info["target_pos"]))
                phase_idx = phase_index_from_dist(dist)
                if dist < TARGET_THRESHOLD * 1.15:
                    action = settle_action(info)
                else:
                    action = candidate_action(info, best["candidate"], plan_step)
                records.append(
                    {
                        "scene": scene_feature(info),
                        "history": pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM),
                        "task": task,
                        "phys": phys_row,
                        "engine": engine_row,
                        "dist": dist,
                        "phase_idx": phase_idx,
                    }
                )
                actions.append(action.astype(np.float32))
                _, _, term, trunc, _ = live_env.step(action)
                next_info = current_info(live_env)
                hist_rows.append(history_row(info, action, next_info))
                prev_finger = info["finger_pos"].copy()
                total_steps += 1
                if term or trunc or total_steps >= args.max_steps:
                    break
            if total_steps >= args.max_steps:
                break

        for _ in range(args.settle_pad):
            info = current_info(live_env)
            dist = float(np.linalg.norm(info["finger_pos"] - info["target_pos"]))
            phase_idx = phase_index_from_dist(dist)
            action = settle_action(info)
            records.append(
                {
                    "scene": scene_feature(info),
                    "history": pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM),
                    "task": task,
                    "phys": phys_row,
                    "engine": engine_row,
                    "dist": dist,
                    "phase_idx": phase_idx,
                }
            )
            actions.append(action.astype(np.float32))
            _, _, term, trunc, _ = live_env.step(action)
            next_info = current_info(live_env)
            hist_rows.append(history_row(info, action, next_info))
            prev_finger = info["finger_pos"].copy()
            total_steps += 1
            if term or trunc:
                break

        final_info = current_info(live_env)
        final_dist = float(np.linalg.norm(final_info["finger_pos"] - final_info["target_pos"]))
        return {
            "seed": int(seed),
            "records": records,
            "actions": actions,
            "success": bool(final_dist < TARGET_THRESHOLD),
            "best_dist": float(best_dist),
            "final_dist": final_dist,
            "steps": int(len(actions)),
        }
    finally:
        live_env.close()
        oracle_env.close()


def build_dataset(seeds: list[int], args: argparse.Namespace) -> dict[str, object]:
    rollouts = [rollout_teacher(seed, args) for seed in seeds]
    rows = {k: [] for k in ("scene", "history", "task", "action", "phys", "engine", "weight")}
    phase_counts = {name: 0 for name in PHASE_NAMES}
    for rollout in rollouts:
        actions = rollout["actions"]
        records = rollout["records"]
        for i, rec in enumerate(records):
            phase_idx = int(rec["phase_idx"])
            dist = float(rec["dist"])
            rows["scene"].append(rec["scene"])
            rows["history"].append(rec["history"])
            rows["task"].append(rec["task"])
            rows["action"].append(build_action_chunk(actions, i, args.chunk_horizon))
            rows["phys"].append(rec["phys"])
            rows["engine"].append(rec["engine"])
            rows["weight"].append(sample_weight(dist, phase_idx, args))
            phase_counts[PHASE_NAMES[phase_idx]] += 1
    if not rows["action"]:
        raise RuntimeError("no causal-teacher samples collected for Reacher v2")
    out = {
        "scene": np.asarray(rows["scene"], dtype=np.float32),
        "history": np.asarray(rows["history"], dtype=np.float32),
        "task": np.asarray(rows["task"], dtype=np.float32),
        "action": np.asarray(rows["action"], dtype=np.float32),
        "phys": np.asarray(rows["phys"], dtype=np.float32),
        "engine": np.asarray(rows["engine"], dtype=np.float32),
        "weight": np.asarray(rows["weight"], dtype=np.float64),
        "seeds": [int(s) for s in seeds],
        "phase_counts": phase_counts,
        "teacher_eval": {
            "success": int(sum(r["success"] for r in rollouts)),
            "n": int(len(rollouts)),
            "success_rate": float(np.mean([r["success"] for r in rollouts])),
            "best_dist_mean": float(np.mean([r["best_dist"] for r in rollouts])),
            "final_dist_mean": float(np.mean([r["final_dist"] for r in rollouts])),
            "steps_mean": float(np.mean([r["steps"] for r in rollouts])),
        },
        "teacher_rollouts": [
            {
                "seed": int(r["seed"]),
                "success": bool(r["success"]),
                "best_dist": float(r["best_dist"]),
                "final_dist": float(r["final_dist"]),
                "steps": int(r["steps"]),
            }
            for r in rollouts
        ],
    }
    return out


def run_student_episode(model: SharedPhysicsStudent, seed: int, args: argparse.Namespace, guide_blend: list[float]) -> dict[str, float]:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    task = mx.array(batch_task_vector(TASK_NAME, 1))
    hist_rows: list[np.ndarray] = []
    total_steps = 0
    stable_steps = 0
    best_dist = 9e9
    prev_finger = current_info(env)["finger_pos"].copy()
    chunk = None
    chunk_i = args.chunk_exec
    phase_sig = None
    try:
        while total_steps < args.max_steps:
            info = current_info(env)
            dist = float(np.linalg.norm(info["finger_pos"] - info["target_pos"]))
            finger_speed = float(np.linalg.norm(info["finger_pos"] - prev_finger))
            qvel_norm = float(np.linalg.norm(info["qvel"]))
            best_dist = min(best_dist, dist)
            if dist < TARGET_THRESHOLD and finger_speed < 0.0018 and qvel_norm < 0.20:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= 6:
                break

            phase_idx = phase_index_from_dist(dist)
            if chunk is None or chunk_i >= args.chunk_exec or phase_idx != phase_sig:
                scene = mx.array(scene_feature(info)[None])
                history = mx.array(pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM)[None])
                action, _, _ = model(scene, history, task)
                mx.eval(action)
                chunk = np.array(action, dtype=np.float32)[0].reshape(args.chunk_horizon, ACTION_COMPONENT_DIM)
                chunk = np.clip(chunk, -1.0, 1.0)
                chunk_i = 0
                phase_sig = phase_idx

            pred = chunk[min(chunk_i, args.chunk_horizon - 1)]
            guide = guide_action(info)
            act = (1.0 - guide_blend[phase_idx]) * pred + guide_blend[phase_idx] * guide
            if phase_idx == PHASE_TERMINAL or dist < TARGET_THRESHOLD * args.micro_settle_radius:
                settle = settle_action(info)
                act = (1.0 - args.settle_mix) * act + args.settle_mix * settle
            act = np.clip(act, -1.0, 1.0).astype(np.float32)
            _, _, term, trunc, _ = env.step(act)
            next_info = current_info(env)
            hist_rows.append(history_row(info, act, next_info))
            prev_finger = info["finger_pos"].copy()
            total_steps += 1
            chunk_i += 1
            if term or trunc:
                break

        final_info = current_info(env)
        final_dist = float(np.linalg.norm(final_info["finger_pos"] - final_info["target_pos"]))
        return {
            "success": bool(final_dist < TARGET_THRESHOLD),
            "final_dist": final_dist,
            "best_dist": float(best_dist),
            "steps": int(total_steps),
        }
    finally:
        env.close()


def evaluate(model: SharedPhysicsStudent, seeds: list[int], args: argparse.Namespace, guide_blend: list[float]) -> dict[str, object]:
    rows = [run_student_episode(model, seed, args, guide_blend) for seed in seeds]
    return {
        "success": int(sum(r["success"] for r in rows)),
        "n": int(len(rows)),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "final_dist_mean": float(np.mean([r["final_dist"] for r in rows])),
        "final_dist_median": float(np.median([r["final_dist"] for r in rows])),
        "best_dist_mean": float(np.mean([r["best_dist"] for r in rows])),
        "best_dist_median": float(np.median([r["best_dist"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "rows": rows,
    }


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    guide_blend = parse_csv_floats(args.guide_blend, len(PHASE_NAMES))
    cfg = vars(args) | {
        "dataset_version": DATASET_VERSION,
        "task": TASK_NAME,
        "history_steps": HIST_STEPS,
        "history_row_dim": HIST_ROW_DIM,
        "teacher": "Reacher true-env causal tree",
        "student": "chunked BC solving student with detached physics/engine probe heads",
        "physics_aux_gradient": "detached_from_solving_encoder"
        if not args.couple_physics_grad
        else "coupled_to_solving_encoder",
        "scene_dim": SCENE_DIM,
        "chunk_action_dim": args.chunk_horizon * ACTION_COMPONENT_DIM,
        "phase_names": list(PHASE_NAMES),
        "guide_blend_parsed": guide_blend,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_seeds = parse_csv_ints(args.train_seeds)
    val_seeds = parse_csv_ints(args.val_seeds)
    t0 = time.time()
    train_data = build_dataset(train_seeds, args)
    val_data = build_dataset(val_seeds, args)
    np.savez_compressed(
        os.path.join(args.out_dir, "train_samples.npz"),
        scene=train_data["scene"],
        history=train_data["history"],
        task=train_data["task"],
        action=train_data["action"],
        phys=train_data["phys"],
        engine=train_data["engine"],
        weight=train_data["weight"].astype(np.float32),
    )
    np.savez_compressed(
        os.path.join(args.out_dir, "val_samples.npz"),
        scene=val_data["scene"],
        history=val_data["history"],
        task=val_data["task"],
        action=val_data["action"],
        phys=val_data["phys"],
        engine=val_data["engine"],
        weight=val_data["weight"].astype(np.float32),
    )
    with open(os.path.join(args.out_dir, "train_cases.json"), "w") as f:
        json.dump({"seeds": train_seeds}, f, indent=2)
    with open(os.path.join(args.out_dir, "val_cases.json"), "w") as f:
        json.dump({"seeds": val_seeds}, f, indent=2)
    with open(os.path.join(args.out_dir, "train_teacher_rollouts.json"), "w") as f:
        json.dump(train_data["teacher_rollouts"], f, indent=2)
    with open(os.path.join(args.out_dir, "val_teacher_rollouts.json"), "w") as f:
        json.dump(val_data["teacher_rollouts"], f, indent=2)
    print(
        f"[data] train={len(train_data['action'])} val={len(val_data['action'])} "
        f"teacher_train={train_data['teacher_eval']['success_rate']:.3f} "
        f"teacher_val={val_data['teacher_eval']['success_rate']:.3f} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    model = SharedPhysicsStudent(
        scene_dim=SCENE_DIM,
        history_dim=HIST_STEPS * HIST_ROW_DIM,
        action_dim=args.chunk_horizon * ACTION_COMPONENT_DIM,
        hidden=args.hidden,
        latent=args.latent,
        couple_physics_grad=args.couple_physics_grad,
    )
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    weights = LossWeights(action=1.0, phys=args.phys_weight, engine=args.engine_weight)
    grad_fn = nn.value_and_grad(
        model,
        lambda m, s, h, t, a, p, e: loss_fn(m, s, h, t, a, p, e, weights),
    )
    rng = np.random.default_rng(args.seed)
    n_train = len(train_data["action"])
    n_val = len(val_data["action"])
    train_probs = train_data["weight"] / train_data["weight"].sum()
    log = []
    for it in range(args.iters):
        replace = n_train < args.batch
        idx = rng.choice(n_train, size=args.batch, replace=replace, p=train_probs)
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

    ckpt = os.path.join(args.out_dir, "reacher_causal_student_v2.npz")
    save_model(model, ckpt)
    eval_row = evaluate(model, val_seeds, args, guide_blend)
    summary = {
        "ckpt": ckpt,
        "dataset_version": DATASET_VERSION,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "train_phase_counts": train_data["phase_counts"],
        "val_phase_counts": val_data["phase_counts"],
        "teacher_train_eval": train_data["teacher_eval"],
        "teacher_val_eval": val_data["teacher_eval"],
        "student_eval": eval_row,
        "final": log[-1],
        "real_training": True,
        "shared_backbone": True,
        "chunked_bc_contract": True,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="meadow/reacher_causal_student_v2")
    ap.add_argument("--train_seeds", default="0,1,2,3,6,7")
    ap.add_argument("--val_seeds", default="4,5")
    ap.add_argument("--max_steps", type=int, default=90)
    ap.add_argument("--oracle_horizon", type=int, default=16)
    ap.add_argument("--exec_horizon", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--chunk_horizon", type=int, default=6)
    ap.add_argument("--chunk_exec", type=int, default=2)
    ap.add_argument("--settle_pad", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval_batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=192)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--phys_weight", type=float, default=0.10)
    ap.add_argument("--engine_weight", type=float, default=0.06)
    ap.add_argument("--couple_physics_grad", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--aggressive_weight", type=float, default=1.15)
    ap.add_argument("--terminal_weight", type=float, default=2.40)
    ap.add_argument("--micro_weight", type=float, default=2.00)
    ap.add_argument("--micro_weight_radius", type=float, default=1.45)
    ap.add_argument("--guide_blend", default="0.10,0.18,0.42")
    ap.add_argument("--settle_mix", type=float, default=0.80)
    ap.add_argument("--micro_settle_radius", type=float, default=1.35)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
