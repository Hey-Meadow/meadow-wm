"""PushT causal student v2: chunked BC-style distillation from the causal tree.

This keeps the teacher data causal-tree-derived, but changes the student
contract from a one-step residual to a closed-loop chunk predictor:

  object-pose-aware state + short contact history
      -> short action chunk
      -> future block-pose delta
      -> future contact mask

At rollout time the student replans every step and executes only the first
action from the predicted chunk, which is closer to the old BC-style contract
that learned better on PushT while still using the stronger causal teacher.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import stable_worldmodel.envs  # noqa: F401
from mlx.utils import tree_flatten, tree_unflatten

from meadow_student_backbone import pad_history
from record_pusht_true_env_causal_tree import (
    ANGLE_MICRO,
    DEFAULT_DATASET,
    ENV_ID,
    POS_MICRO,
    choose_guarded_plan,
    generate_candidates,
    get_full_snapshot,
    pad_chunk,
    pose_error,
    score_candidate,
)
from record_pusht_true_env_teacher import (
    current_state,
    episode_slice,
    load_compact_dataset,
    set_episode,
)
from rtg_translator import augment_state


DATASET_VERSION = "pusht_causal_student_v2_chunk_pose"
ACTION_DIM = 2
POSE_DIM = 3
HIST_STEPS = 4
HIST_ROW_DIM = 8
CONTACT_GOAL_POS = 20.0
CONTACT_GOAL_ANGLE = float(np.pi / 9)


def wrap_angle(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))


def norm_state(s):
    s = np.asarray(s, dtype=np.float32).copy()
    out = np.zeros(7, dtype=np.float32)
    out[0:4] = s[0:4] / 512.0 - 0.5
    out[4] = wrap_angle(float(s[4])) / np.pi
    out[5:7] = s[5:7] / 64.0
    return out


def guide_action(state, goal_state):
    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    goal = goal_state[2:4].astype(np.float32)
    push_vec = goal - block
    n = float(np.linalg.norm(push_vec))
    push_dir = push_vec / max(n, 1e-6)
    contact_dist = 34.0
    ideal = block - push_dir * contact_dist
    if float(np.linalg.norm(agent - ideal)) > 18.0:
        action = (ideal - agent) / 100.0
    else:
        action = push_dir * 0.18
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def history_row(prev_state, action, next_state, contact_flag):
    agent_delta = (next_state[:2] - prev_state[:2]) / 64.0
    block_delta = (next_state[2:4] - prev_state[2:4]) / 64.0
    angle_delta = wrap_angle(float(next_state[4] - prev_state[4])) / np.pi
    return np.concatenate(
        [
            np.clip(action, -1.0, 1.0).astype(np.float32),
            agent_delta.astype(np.float32),
            block_delta.astype(np.float32),
            np.array([angle_delta, float(contact_flag)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def scene_feature(state, goal_state, t, total_steps, prev_action, prev_state, prev_contact):
    aug = augment_state(
        state,
        goal_state,
        t=t,
        T_expected=max(total_steps, 1),
        prev_action=prev_action,
        prev_state=prev_state,
        include_prev_action=True,
        include_3grav=True,
    )
    pos, ang = pose_error(state, goal_state)
    return np.concatenate(
        [
            aug.astype(np.float32),
            norm_state(state),
            guide_action(state, goal_state),
            np.array(
                [
                    min(pos / 128.0, 4.0),
                    ang / np.pi,
                    float(prev_contact),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)


def build_input_feature(state, goal_state, t, total_steps, prev_action, prev_state, prev_contact, hist_rows):
    scene = scene_feature(state, goal_state, t, total_steps, prev_action, prev_state, prev_contact)
    history = pad_history(hist_rows, HIST_STEPS, HIST_ROW_DIM)
    return np.concatenate([scene, history], axis=0).astype(np.float32)


def discount_weights(horizon: int) -> np.ndarray:
    base = np.array([0.72 ** i for i in range(horizon)], dtype=np.float32)
    return base / max(float(base.mean()), 1e-6)


def future_pose_target(states, start_idx, horizon):
    end_idx = min(start_idx + horizon, len(states) - 1)
    cur = states[start_idx]
    fut = states[end_idx]
    return np.array(
        [
            (fut[2] - cur[2]) / 64.0,
            (fut[3] - cur[3]) / 64.0,
            wrap_angle(float(fut[4] - cur[4])) / np.pi,
        ],
        dtype=np.float32,
    )


def relax_success(state, goal_state):
    pos, ang = pose_error(state, goal_state)
    return bool(pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE)


def signed_pose_error(state, goal_state):
    dx = float(goal_state[2] - state[2])
    dy = float(goal_state[3] - state[3])
    dtheta = float(wrap_angle(float(goal_state[4] - state[4])))
    return dx, dy, dtheta


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32), 0.0
    return (v / n).astype(np.float32), n


def contact_push_chunk(state, push_dir, horizon, magnitude, lever, contact_dist, rng):
    """Relative-action chunk that approaches a contact side, then micro-pushes.

    swm/PushT-v1 uses relative actions: action * action_scale becomes the
    pusher's local target displacement. The first few commands move toward a
    contact point; later commands keep contact and apply a small push.
    """
    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    push_dir, n = _unit(push_dir)
    if n < 1e-6:
        return np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    perp = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)
    contact = block - push_dir * float(contact_dist) + perp * float(lever)
    to_contact = np.clip((contact - agent) / 100.0, -1.0, 1.0).astype(np.float32)
    dist = float(np.linalg.norm(contact - agent))
    approach_steps = int(np.clip(np.ceil(dist / 24.0), 1, min(5, horizon)))
    actions = []
    for i in range(horizon):
        if i < approach_steps:
            decay = max(0.22, 1.0 - 0.20 * i)
            a = to_contact * decay + rng.normal(0, 0.008, 2).astype(np.float32)
        else:
            # Keep the contact point loaded while pushing in the correction direction.
            preload = np.clip(to_contact, -0.12, 0.12) * 0.20
            a = push_dir * float(magnitude) + preload + rng.normal(0, 0.006, 2).astype(np.float32)
        actions.append(np.clip(a, -1.0, 1.0).astype(np.float32))
    return np.asarray(actions, dtype=np.float32)


def terminal_precision_candidates(state, goal_state, k, horizon, rng):
    dx, dy, dtheta = signed_pose_error(state, goal_state)
    residual = np.array([dx, dy], dtype=np.float32)
    res_dir, res_norm = _unit(residual)
    dirs = []
    labels = []
    if res_norm > 1e-4:
        dirs.append(res_dir)
        labels.append("residual")
    for axis, name in (
        (np.array([np.sign(dx) if abs(dx) > 1e-4 else 0.0, 0.0], dtype=np.float32), "axis_x"),
        (np.array([0.0, np.sign(dy) if abs(dy) > 1e-4 else 0.0], dtype=np.float32), "axis_y"),
    ):
        axis_dir, axis_norm = _unit(axis)
        if axis_norm > 0:
            dirs.append(axis_dir)
            labels.append(name)
    if not dirs:
        dirs.append(np.array([1.0, 0.0], dtype=np.float32))
        labels.append("settle")

    chunks = []
    out_labels = []
    theta_sign = np.sign(dtheta) if abs(dtheta) > 1e-4 else 0.0
    mags = [0.030, 0.045, 0.065, 0.085]
    contact_dists = [28.0, 32.0, 36.0]
    lever_base = [0.0, 12.0, 22.0, 32.0]
    for di, (push_dir, name) in enumerate(zip(dirs, labels)):
        for mag in mags:
            for cd in contact_dists:
                chunks.append(contact_push_chunk(state, push_dir, horizon, mag, 0.0, cd, rng))
                out_labels.append(f"precision_translate_{name}_{mag:.3f}_{cd:.0f}")
                if theta_sign != 0:
                    # Positive dtheta needs positive torque. With contact at
                    # block - push_dir*d + perp*lever, torque sign is roughly
                    # -sign(lever), hence the negative below.
                    for lever_mag in lever_base[1:]:
                        lever = -theta_sign * lever_mag
                        chunks.append(contact_push_chunk(state, push_dir, horizon, mag, lever, cd, rng))
                        out_labels.append(f"precision_blend_{name}_{mag:.3f}_{cd:.0f}_{lever:.0f}")
                if len(chunks) >= k:
                    return np.asarray(chunks[:k], dtype=np.float32), out_labels[:k]

    # Pure rotation probes from cardinal push directions. These are useful when
    # position is already close but theta is still outside the micro threshold.
    cardinals = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
    ]
    while len(chunks) < k and theta_sign != 0:
        push_dir = cardinals[len(chunks) % len(cardinals)]
        mag = mags[len(chunks) % len(mags)]
        lever = -theta_sign * lever_base[1 + (len(chunks) % (len(lever_base) - 1))]
        chunks.append(contact_push_chunk(state, push_dir, horizon, mag, lever, 32.0, rng))
        out_labels.append(f"precision_rotate_{mag:.3f}_{lever:.0f}")

    while len(chunks) < k:
        chunks.append(np.zeros((horizon, ACTION_DIM), dtype=np.float32))
        out_labels.append("precision_settle")
    return np.asarray(chunks[:k], dtype=np.float32), out_labels[:k]


def attach_terminal_pose_score(row, goal_state, args, current_pos, current_ang, current_axis):
    path = row.get("path")
    if path is None or len(path) == 0:
        row["axis_error"] = float("inf")
        row["terminal_error"] = float("inf")
        return row
    final_pose = np.asarray(path[-1], dtype=np.float32)
    dx = float(goal_state[2] - final_pose[0])
    dy = float(goal_state[3] - final_pose[1])
    axis = max(abs(dx), abs(dy))
    terminal_error = (
        args.term_pos_weight * float(row["pos"])
        + args.term_axis_weight * axis
        + args.term_angle_weight * float(row["angle"])
    )
    current_error = (
        args.term_pos_weight * float(current_pos)
        + args.term_axis_weight * float(current_axis)
        + args.term_angle_weight * float(current_ang)
    )
    row["axis_error"] = float(axis)
    row["terminal_error"] = float(terminal_error)
    row["micro_success"] = bool(row["pos"] <= POS_MICRO and row["angle"] <= ANGLE_MICRO)
    row["terminal_improve_vs_current"] = float(current_error - terminal_error)
    row["score"] = float(row["score"] - args.term_score_weight * terminal_error)
    if row["micro_success"]:
        row["score"] += float(args.term_micro_bonus)
    if str(row.get("label", "")).startswith("precision_"):
        row["score"] += float(args.term_precision_prior)
    return row


def terminal_error_value(row, args):
    if "terminal_error" in row:
        return float(row["terminal_error"])
    return (
        args.term_pos_weight * float(row["pos"])
        + args.term_axis_weight * float(row.get("axis_error", row["pos"]))
        + args.term_angle_weight * float(row["angle"])
    )


def choose_terminal_precision_plan(scored, args, current_pos, current_ang, current_axis):
    best = scored[0]
    expert = next((r for r in scored if r["label"] == "expert_tail"), best)
    micro = [r for r in scored if r.get("micro_success", False)]
    if micro:
        chosen = max(micro, key=lambda r: r["score"])
        chosen = dict(chosen)
        chosen["guard_used"] = False
        chosen["expert_score"] = float(expert["score"])
        chosen["terminal_choice"] = "micro"
        return chosen

    current_error = (
        args.term_pos_weight * current_pos
        + args.term_axis_weight * current_axis
        + args.term_angle_weight * current_ang
    )
    best_error = terminal_error_value(best, args)
    expert_error = terminal_error_value(expert, args)
    improves_current = best_error <= current_error - args.term_min_current_improve
    improves_expert = best_error <= expert_error - args.term_min_expert_improve
    stable = best["pos"] <= args.accept_candidate_max_pos and best["angle"] <= args.accept_candidate_max_angle
    if stable and (improves_current or improves_expert):
        chosen = dict(best)
        chosen["guard_used"] = False
        chosen["expert_score"] = float(expert["score"])
        chosen["terminal_choice"] = "incremental"
        return chosen

    chosen = dict(expert)
    chosen["guard_used"] = True
    chosen["raw_best_label"] = best["label"]
    chosen["raw_best_score"] = float(best["score"])
    chosen["expert_score"] = float(expert["score"])
    chosen["terminal_choice"] = "expert_guard"
    return chosen


def choose_phase_plan(scored, args, pos, ang, axis=None):
    """Use precision acceptance only inside the terminal pose basin.

    Global precision gating can select locally attractive candidates before the
    T pose is controllable. This keeps the original guarded teacher in the
    approach/contact phase, then allows strict per-axis precision candidates
    only after the object is already near the target.
    """
    if not args.precision_phase_only:
        return choose_guarded_plan(scored, args)
    precision_mode = pos < args.precision_phase_pos and ang < args.precision_phase_angle
    if precision_mode:
        if args.terminal_precision_teacher:
            return choose_terminal_precision_plan(scored, args, pos, ang, float(pos if axis is None else axis))
        return choose_guarded_plan(scored, args)
    strict_args = argparse.Namespace(**vars(args))
    strict_args.accept_candidate_pos = 0.0
    strict_args.accept_candidate_angle = 0.0
    strict_args.accept_candidate_max_pos = 0.0
    strict_args.accept_candidate_max_angle = 0.0
    return choose_guarded_plan(scored, strict_args)


def _raw_block_velocity(raw):
    try:
        return np.asarray(raw.block.velocity, dtype=np.float32).copy()
    except Exception:
        return np.zeros(2, dtype=np.float32)


def student_terminal_hold_action(state, goal_state, raw, args):
    """Small active hold used only during student evaluation.

    Once the pusher reaches the goal basin, zero action can let residual object
    momentum drift the T back outside the micro threshold. This controller is a
    deployment safety layer, not a replacement for the causal teacher: it uses
    signed pose residual plus measured block velocity to keep the object loaded
    gently toward the target.
    """
    pos, ang = pose_error(state, goal_state)
    if pos > args.student_hold_pos or ang > args.student_hold_angle:
        return np.zeros(2, dtype=np.float32)

    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    goal = goal_state[2:4].astype(np.float32)
    residual = goal - block
    block_vel = _raw_block_velocity(raw)
    desired = args.student_hold_pos_gain * residual - args.student_hold_vel_gain * block_vel
    push_dir, n = _unit(desired)
    if n < 1e-5:
        return np.zeros(2, dtype=np.float32)

    contact = block - push_dir * args.student_hold_contact_dist
    to_contact = contact - agent
    dist = float(np.linalg.norm(to_contact))
    if dist > args.student_hold_contact_tol:
        action = to_contact / 100.0
    else:
        mag = min(args.student_hold_max_action, args.student_hold_min_action + n / args.student_hold_norm_scale)
        action = push_dir * mag
    return np.clip(action, -args.student_hold_clip, args.student_hold_clip).astype(np.float32)


def student_terminal_release_action(state, goal_state, args):
    pos, ang = pose_error(state, goal_state)
    if pos > args.student_release_pos or ang > args.student_release_angle:
        return np.zeros(2, dtype=np.float32)
    agent = state[:2].astype(np.float32)
    block = state[2:4].astype(np.float32)
    away, n = _unit(agent - block)
    if n < 1e-5:
        return np.zeros(2, dtype=np.float32)
    return np.clip(away * args.student_release_action, -1.0, 1.0).astype(np.float32)


def student_terminal_settle_action(state, goal_state, raw, args):
    if args.student_hold_mode == "release":
        return student_terminal_release_action(state, goal_state, args)
    if args.student_hold_mode == "active":
        return student_terminal_hold_action(state, goal_state, raw, args)
    return np.zeros(2, dtype=np.float32)


def causal_rollout_episode(data, ep_idx, args):
    rng = np.random.default_rng(args.seed + ep_idx)
    live_env = gym.make(ENV_ID, render_mode=None)
    oracle_env = gym.make(ENV_ID, render_mode=None)
    ep_states, ep_actions = episode_slice(data, ep_idx)
    goal_state = ep_states[-1].astype(np.float32)
    obs = set_episode(live_env, ep_states[0], goal_state)
    raw = live_env.unwrapped

    hist_rows: list[np.ndarray] = []
    feature_rows = []
    action_rows = []
    contact_rows = []
    states = [current_state(obs).astype(np.float32)]
    t = 0
    control_budget = len(ep_actions) + (args.extra_precision_steps if args.terminal_precision_teacher else 0)
    control_steps = min(args.max_steps, control_budget)
    total_steps = control_steps + args.hold_frames
    first_success_step = None
    micro_streak = 0
    prev_action = np.zeros(2, dtype=np.float32)
    prev_state = None
    prev_contact = 0.0
    current_chunk = None
    chunk_i = args.chunk_exec

    try:
        while t < control_steps:
            state = current_state(obs).astype(np.float32)
            pos, ang = pose_error(state, goal_state)
            if first_success_step is None and pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE:
                first_success_step = t
            if pos <= POS_MICRO and ang <= ANGLE_MICRO:
                micro_streak += 1
            else:
                micro_streak = 0
            if (
                micro_streak >= args.micro_hold_steps
                and t >= max(4, (first_success_step or 0) + args.min_tail_steps)
            ):
                break

            if current_chunk is None or chunk_i >= args.chunk_exec:
                saved_snapshot = get_full_snapshot(raw)
                chunks, labels = generate_candidates(
                    ep_actions, t, state, goal_state, args.candidates, args.horizon, rng
                )
                dx, dy, dtheta = signed_pose_error(state, goal_state)
                axis = max(abs(dx), abs(dy))
                precision_mode = (
                    args.terminal_precision_teacher
                    and pos < args.precision_phase_pos
                    and ang < args.precision_phase_angle
                )
                if precision_mode:
                    p_chunks, p_labels = terminal_precision_candidates(
                        state,
                        goal_state,
                        args.precision_candidates,
                        args.horizon,
                        rng,
                    )
                    chunks = np.concatenate([chunks, p_chunks], axis=0)
                    labels = labels + p_labels
                recovery = pad_chunk(ep_actions, t + args.chunk_exec, args.recovery_horizon)
                precision_recovery = np.zeros((args.precision_recovery_horizon, ACTION_DIM), dtype=np.float32)
                scored = []
                for ci, chunk in enumerate(chunks):
                    label = labels[ci]
                    if precision_mode and str(label).startswith("precision_"):
                        prefix = chunk[: args.precision_score_horizon]
                        row = score_candidate(
                            oracle_env,
                            saved_snapshot,
                            goal_state,
                            prefix,
                            precision_recovery,
                            angle_weight=args.term_raw_angle_weight,
                        )
                    else:
                        prefix = chunk[: args.chunk_exec]
                        row = score_candidate(
                            oracle_env,
                            saved_snapshot,
                            goal_state,
                            prefix,
                            recovery,
                            angle_weight=args.term_raw_angle_weight if precision_mode else 42.0,
                        )
                    row["idx"] = ci
                    row["label"] = label
                    if precision_mode:
                        row = attach_terminal_pose_score(row, goal_state, args, pos, ang, axis)
                    scored.append(row)
                scored.sort(key=lambda r: r["score"], reverse=True)
                best = choose_phase_plan(scored, args, pos, ang, axis)
                current_chunk = chunks[best["idx"]]
                chunk_i = 0

            feature_rows.append(
                build_input_feature(
                    state,
                    goal_state,
                    t=t,
                    total_steps=total_steps,
                    prev_action=prev_action,
                    prev_state=prev_state,
                    prev_contact=prev_contact,
                    hist_rows=hist_rows,
                )
            )
            teacher_action = current_chunk[chunk_i].astype(np.float32)
            next_obs, _, term, trunc, info = live_env.step(np.clip(teacher_action, -1.0, 1.0))
            next_state = current_state(next_obs).astype(np.float32)
            contact = float(info.get("n_contacts", 0) > 0)
            hist_rows.append(history_row(state, teacher_action, next_state, contact))
            action_rows.append(teacher_action)
            contact_rows.append(contact)
            states.append(next_state)
            prev_state = state
            prev_action = teacher_action
            prev_contact = contact
            obs = next_obs
            chunk_i += 1
            t += 1
            if trunc or (term and not args.continue_after_env_success):
                break
    finally:
        try:
            for _ in range(args.hold_frames):
                state = current_state(obs).astype(np.float32)
                feature_rows.append(
                    build_input_feature(
                        state,
                        goal_state,
                        t=t,
                        total_steps=total_steps,
                        prev_action=prev_action,
                        prev_state=prev_state,
                        prev_contact=prev_contact,
                        hist_rows=hist_rows,
                    )
                )
                settle_action = np.zeros(2, dtype=np.float32)
                next_obs, _, term, trunc, info = live_env.step(settle_action)
                next_state = current_state(next_obs).astype(np.float32)
                contact = float(info.get("n_contacts", 0) > 0)
                hist_rows.append(history_row(state, settle_action, next_state, contact))
                action_rows.append(settle_action)
                contact_rows.append(contact)
                states.append(next_state)
                prev_state = state
                prev_action = settle_action
                prev_contact = contact
                obs = next_obs
                t += 1
                if trunc or (term and not args.continue_after_env_success):
                    break
        finally:
            live_env.close()
            oracle_env.close()

    n_steps = len(action_rows)
    if n_steps == 0:
        return {
            "episode_idx": int(ep_idx),
            "success": False,
            "steps": 0,
            "goal_state": goal_state,
            "features": np.zeros((0, 1), dtype=np.float32),
            "action_chunk": np.zeros((0, args.pred_horizon, ACTION_DIM), dtype=np.float32),
            "future_pose": np.zeros((0, POSE_DIM), dtype=np.float32),
            "future_contact": np.zeros((0, args.pred_horizon), dtype=np.float32),
            "final_state": goal_state,
            "final_block_error": float("inf"),
            "final_theta_error": float("inf"),
            "relaxed_success": False,
        }

    chunk_targets = np.zeros((n_steps, args.pred_horizon, ACTION_DIM), dtype=np.float32)
    contact_targets = np.zeros((n_steps, args.pred_horizon), dtype=np.float32)
    pose_targets = np.zeros((n_steps, POSE_DIM), dtype=np.float32)
    states_arr = np.asarray(states, dtype=np.float32)
    for i in range(n_steps):
        tail = np.asarray(action_rows[i : i + args.pred_horizon], dtype=np.float32)
        if len(tail) > 0:
            chunk_targets[i, : len(tail)] = tail
        ctail = np.asarray(contact_rows[i : i + args.pred_horizon], dtype=np.float32)
        if len(ctail) > 0:
            contact_targets[i, : len(ctail)] = ctail
        pose_targets[i] = future_pose_target(states_arr, i, args.pred_horizon)

    final_state = states_arr[-1]
    final_pos, final_ang = pose_error(final_state, goal_state)
    return {
        "episode_idx": int(ep_idx),
        "success": bool(final_pos <= POS_MICRO and final_ang <= ANGLE_MICRO),
        "relaxed_success": relax_success(final_state, goal_state),
        "steps": int(n_steps),
        "goal_state": goal_state,
        "features": np.asarray(feature_rows, dtype=np.float32),
        "action_chunk": chunk_targets,
        "future_pose": pose_targets,
        "future_contact": contact_targets,
        "final_state": final_state,
        "final_block_error": float(final_pos),
        "final_theta_error": float(final_ang),
    }


def parse_episode_list(spec: str):
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def build_dataset(data, episode_ids, args):
    rows = [causal_rollout_episode(data, ep_idx, args) for ep_idx in episode_ids]
    valid = [r for r in rows if len(r["features"]) > 0]
    if not valid:
        raise RuntimeError("No valid teacher rollouts were collected for the requested episodes.")
    out = {
        "feature": np.concatenate([r["features"] for r in valid], axis=0).astype(np.float32),
        "action_chunk": np.concatenate([r["action_chunk"] for r in valid], axis=0).astype(np.float32),
        "future_pose": np.concatenate([r["future_pose"] for r in valid], axis=0).astype(np.float32),
        "future_contact": np.concatenate([r["future_contact"] for r in valid], axis=0).astype(np.float32),
        "episodes_requested": [int(x) for x in episode_ids],
        "episodes_used": [int(r["episode_idx"]) for r in valid],
        "teacher_rows": [
            {
                "episode_idx": int(r["episode_idx"]),
                "success": bool(r["success"]),
                "relaxed_success": bool(r["relaxed_success"]),
                "steps": int(r["steps"]),
                "final_block_error": float(r["final_block_error"]),
                "final_theta_error": float(r["final_theta_error"]),
            }
            for r in rows
        ],
    }
    out["teacher_success_rate"] = float(np.mean([r["success"] for r in rows]))
    out["teacher_relaxed_success_rate"] = float(np.mean([r["relaxed_success"] for r in rows]))
    return out


class ChunkPoseStudent(nn.Module):
    def __init__(self, input_dim: int, hidden: int, pred_horizon: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.pred_horizon = int(pred_horizon)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )
        self.chunk_head = nn.Linear(self.hidden, self.pred_horizon * ACTION_DIM)
        self.pose_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(),
            nn.Linear(self.hidden // 2, POSE_DIM),
        )
        self.contact_head = nn.Linear(self.hidden, self.pred_horizon)

    def __call__(self, x):
        z = self.backbone(x)
        chunk = mx.reshape(mx.tanh(self.chunk_head(z)), (-1, self.pred_horizon, ACTION_DIM))
        pose = self.pose_head(z)
        contact_logits = self.contact_head(z)
        return chunk, pose, contact_logits


@dataclass
class LossWeights:
    pose: float = 0.20
    contact: float = 0.05


def action_loss_fn(pred_chunk, target_chunk, action_weights):
    err = (pred_chunk - target_chunk) ** 2
    return mx.mean(err * action_weights[None, :, None])


def loss_fn(model, x, action_target, pose_target, contact_target, action_weights, weights: LossWeights):
    pred_chunk, pred_pose, contact_logits = model(x)
    act_loss = action_loss_fn(pred_chunk, action_target, action_weights)
    pose_loss = mx.mean((pred_pose - pose_target) ** 2)
    contact_loss = mx.mean((sigmoid(contact_logits) - contact_target) ** 2)
    total = act_loss + weights.pose * pose_loss + weights.contact * contact_loss
    return total


def eval_components(model, x, action_target, pose_target, contact_target, action_weights):
    pred_chunk, pred_pose, contact_logits = model(x)
    act_loss = action_loss_fn(pred_chunk, action_target, action_weights)
    pose_loss = mx.mean((pred_pose - pose_target) ** 2)
    contact_loss = mx.mean((sigmoid(contact_logits) - contact_target) ** 2)
    mx.eval(act_loss, pose_loss, contact_loss)
    return {
        "action_loss": float(act_loss.item()),
        "pose_loss": float(pose_loss.item()),
        "contact_loss": float(contact_loss.item()),
    }


def save_model(model, out_path: str) -> None:
    mx.savez(out_path, **dict(tree_flatten(model.parameters())))


def load_model(model, path: str) -> ChunkPoseStudent:
    model.update(tree_unflatten(list(mx.load(path).items())))
    mx.eval(model.parameters())
    return model


def summarize_eval_rows(rows):
    return {
        "micro_success": int(sum(r["micro_success"] for r in rows)),
        "n": int(len(rows)),
        "micro_success_rate": float(np.mean([r["micro_success"] for r in rows])),
        "env_success": int(sum(r["env_success"] for r in rows)),
        "env_success_rate": float(np.mean([r["env_success"] for r in rows])),
        "final_block_error_mean": float(np.mean([r["final_block_error"] for r in rows])),
        "final_theta_error_mean": float(np.mean([r["final_theta_error"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "episodes": rows,
    }


def rollout_summary(final_state, goal_state, relaxed_hit, steps):
    final_pos, final_ang = pose_error(final_state, goal_state)
    return {
        "micro_success": bool(final_pos <= POS_MICRO and final_ang <= ANGLE_MICRO),
        "env_success": bool(final_pos < CONTACT_GOAL_POS and final_ang < CONTACT_GOAL_ANGLE),
        "relaxed_hit_step": None if relaxed_hit is None else int(relaxed_hit),
        "final_block_error": float(final_pos),
        "final_theta_error": float(final_ang),
        "steps": int(steps),
    }


def run_student_episode(model, data, ep_idx, args):
    env = gym.make(ENV_ID, render_mode=None)
    raw = env.unwrapped
    ep_states, _ = episode_slice(data, ep_idx)
    goal_state = ep_states[-1].astype(np.float32)
    obs = set_episode(env, ep_states[0], goal_state)
    hist_rows: list[np.ndarray] = []
    prev_action = np.zeros(2, dtype=np.float32)
    prev_state = None
    prev_contact = 0.0
    t = 0
    relaxed_hit = None
    micro_streak = 0
    try:
        total_steps = args.max_steps + args.eval_hold_frames
        while t < args.max_steps:
            state = current_state(obs).astype(np.float32)
            pos, ang = pose_error(state, goal_state)
            if relaxed_hit is None and pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE:
                relaxed_hit = t
            if pos <= POS_MICRO and ang <= ANGLE_MICRO:
                micro_streak += 1
            else:
                micro_streak = 0
            if micro_streak >= args.micro_hold_steps:
                break

            feat = build_input_feature(
                state,
                goal_state,
                t=t,
                total_steps=total_steps,
                prev_action=prev_action,
                prev_state=prev_state,
                prev_contact=prev_contact,
                hist_rows=hist_rows,
            )
            pred_chunk, _, contact_logits = model(mx.array(feat[None]))
            mx.eval(pred_chunk, contact_logits)
            action = np.array(pred_chunk, dtype=np.float32)[0, 0]
            contact_prob = float(np.array(sigmoid(contact_logits), dtype=np.float32)[0, 0])
            if contact_prob < args.guide_blend_contact_thresh and pos > args.guide_blend_min_pos:
                action = (
                    (1.0 - args.guide_blend) * action
                    + args.guide_blend * guide_action(state, goal_state)
                )
            if pos < 24.0 and ang < 0.12:
                action *= 0.40
            elif pos < 42.0 and ang < 0.22:
                action *= 0.70
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            if args.student_terminal_safety and pos < args.student_safety_pos and ang < args.student_safety_angle:
                hold_action = student_terminal_hold_action(state, goal_state, raw, args)
                action = (
                    (1.0 - args.student_safety_blend) * action
                    + args.student_safety_blend * hold_action
                ).astype(np.float32)
            next_obs, _, term, trunc, info = env.step(action)
            next_state = current_state(next_obs).astype(np.float32)
            contact = float(info.get("n_contacts", 0) > 0)
            hist_rows.append(history_row(state, action, next_state, contact))
            prev_state = state
            prev_action = action
            prev_contact = contact
            obs = next_obs
            t += 1
            if trunc or (term and not args.continue_after_env_success):
                break
        for _ in range(args.eval_hold_frames):
            state = current_state(obs).astype(np.float32)
            pos, ang = pose_error(state, goal_state)
            if relaxed_hit is None and pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE:
                relaxed_hit = t
            hold_action = np.zeros(2, dtype=np.float32)
            if args.student_terminal_hold:
                hold_action = student_terminal_settle_action(state, goal_state, raw, args)
            next_obs, _, term, trunc, info = env.step(hold_action)
            next_state = current_state(next_obs).astype(np.float32)
            contact = float(info.get("n_contacts", 0) > 0)
            hist_rows.append(history_row(state, hold_action, next_state, contact))
            prev_state = state
            prev_action = hold_action
            prev_contact = contact
            obs = next_obs
            t += 1
            if trunc or (term and not args.continue_after_env_success):
                break
        final_state = current_state(obs).astype(np.float32)
        return rollout_summary(final_state, goal_state, relaxed_hit, t)
    finally:
        env.close()


def evaluate(model, data, episode_ids, args):
    rows = [run_student_episode(model, data, ep_idx, args) for ep_idx in episode_ids]
    return summarize_eval_rows(rows)


def build_knn_bank(train_data):
    feat = train_data["feature"].astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-4
    return {
        "feature": feat,
        "feature_norm": (feat - mean) / std,
        "action_chunk": train_data["action_chunk"].astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def save_knn_bank(bank, out_path: str) -> None:
    np.savez_compressed(
        out_path,
        feature=bank["feature"],
        action_chunk=bank["action_chunk"],
        mean=bank["mean"],
        std=bank["std"],
    )


def knn_predict_chunk(bank, feat, k):
    z = (feat.astype(np.float32) - bank["mean"][0]) / bank["std"][0]
    d = np.mean((bank["feature_norm"] - z) ** 2, axis=1)
    k = min(int(k), len(d))
    idx = np.argpartition(d, k - 1)[:k]
    w = 1.0 / (d[idx] + 1e-6)
    w = w / np.sum(w)
    return np.sum(bank["action_chunk"][idx] * w[:, None, None], axis=0).astype(np.float32)


def run_knn_episode(bank, data, ep_idx, args):
    env = gym.make(ENV_ID, render_mode=None)
    raw = env.unwrapped
    ep_states, _ = episode_slice(data, ep_idx)
    goal_state = ep_states[-1].astype(np.float32)
    obs = set_episode(env, ep_states[0], goal_state)
    hist_rows: list[np.ndarray] = []
    prev_action = np.zeros(2, dtype=np.float32)
    prev_state = None
    prev_contact = 0.0
    t = 0
    relaxed_hit = None
    micro_streak = 0
    try:
        total_steps = args.max_steps + args.eval_hold_frames
        while t < args.max_steps:
            state = current_state(obs).astype(np.float32)
            pos, ang = pose_error(state, goal_state)
            if relaxed_hit is None and pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE:
                relaxed_hit = t
            if pos <= POS_MICRO and ang <= ANGLE_MICRO:
                micro_streak += 1
            else:
                micro_streak = 0
            if micro_streak >= args.micro_hold_steps:
                break

            feat = build_input_feature(
                state,
                goal_state,
                t=t,
                total_steps=total_steps,
                prev_action=prev_action,
                prev_state=prev_state,
                prev_contact=prev_contact,
                hist_rows=hist_rows,
            )
            action = np.clip(knn_predict_chunk(bank, feat, args.knn_k)[0], -1.0, 1.0).astype(np.float32)
            if args.student_terminal_safety and pos < args.student_safety_pos and ang < args.student_safety_angle:
                hold_action = student_terminal_hold_action(state, goal_state, raw, args)
                action = (
                    (1.0 - args.student_safety_blend) * action
                    + args.student_safety_blend * hold_action
                ).astype(np.float32)
            next_obs, _, term, trunc, info = env.step(action)
            next_state = current_state(next_obs).astype(np.float32)
            contact = float(info.get("n_contacts", 0) > 0)
            hist_rows.append(history_row(state, action, next_state, contact))
            prev_state = state
            prev_action = action
            prev_contact = contact
            obs = next_obs
            t += 1
            if trunc or (term and not args.continue_after_env_success):
                break
        for _ in range(args.eval_hold_frames):
            state = current_state(obs).astype(np.float32)
            pos, ang = pose_error(state, goal_state)
            if relaxed_hit is None and pos < CONTACT_GOAL_POS and ang < CONTACT_GOAL_ANGLE:
                relaxed_hit = t
            hold_action = np.zeros(2, dtype=np.float32)
            if args.student_terminal_hold:
                hold_action = student_terminal_settle_action(state, goal_state, raw, args)
            next_obs, _, term, trunc, info = env.step(hold_action)
            next_state = current_state(next_obs).astype(np.float32)
            contact = float(info.get("n_contacts", 0) > 0)
            hist_rows.append(history_row(state, hold_action, next_state, contact))
            prev_state = state
            prev_action = hold_action
            prev_contact = contact
            obs = next_obs
            t += 1
            if trunc or (term and not args.continue_after_env_success):
                break
        final_state = current_state(obs).astype(np.float32)
        return rollout_summary(final_state, goal_state, relaxed_hit, t)
    finally:
        env.close()


def evaluate_knn(bank, data, episode_ids, args):
    rows = [run_knn_episode(bank, data, ep_idx, args) for ep_idx in episode_ids]
    return summarize_eval_rows(rows)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    config = vars(args) | {
        "dataset_version": DATASET_VERSION,
        "history_steps": HIST_STEPS,
        "history_row_dim": HIST_ROW_DIM,
        "teacher": "PushT true-env causal-tree teacher with guarded chunk selection",
        "student": "object-pose-aware chunk BC student with future-pose/contact auxiliaries",
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    t0 = time.time()
    data = load_compact_dataset(args.dataset)
    train_eps = parse_episode_list(args.train_episodes)
    val_eps = parse_episode_list(args.val_episodes)
    train_data = build_dataset(data, train_eps, args)
    val_data = build_dataset(data, val_eps, args)
    np.savez_compressed(
        os.path.join(args.out_dir, "train_samples.npz"),
        feature=train_data["feature"],
        action_chunk=train_data["action_chunk"],
        future_pose=train_data["future_pose"],
        future_contact=train_data["future_contact"],
    )
    np.savez_compressed(
        os.path.join(args.out_dir, "val_samples.npz"),
        feature=val_data["feature"],
        action_chunk=val_data["action_chunk"],
        future_pose=val_data["future_pose"],
        future_contact=val_data["future_contact"],
    )
    with open(os.path.join(args.out_dir, "train_cases.json"), "w") as f:
        json.dump(
            {
                "episodes_requested": train_data["episodes_requested"],
                "episodes_used": train_data["episodes_used"],
                "teacher_rows": train_data["teacher_rows"],
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.out_dir, "val_cases.json"), "w") as f:
        json.dump(
            {
                "episodes_requested": val_data["episodes_requested"],
                "episodes_used": val_data["episodes_used"],
                "teacher_rows": val_data["teacher_rows"],
            },
            f,
            indent=2,
        )
    print(
        f"[data] train={len(train_data['feature'])} val={len(val_data['feature'])} "
        f"teacher_train={train_data['teacher_success_rate']:.3f} "
        f"teacher_val={val_data['teacher_success_rate']:.3f} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    input_dim = int(train_data["feature"].shape[-1])
    model = ChunkPoseStudent(input_dim=input_dim, hidden=args.hidden, pred_horizon=args.pred_horizon)
    mx.eval(model.parameters())
    lr_sched = optim.cosine_decay(args.lr, args.iters, args.lr / 8.0)
    opt = optim.AdamW(learning_rate=lr_sched, weight_decay=args.wd)
    loss_weights = LossWeights(pose=args.pose_weight, contact=args.contact_weight)
    action_weights = mx.array(discount_weights(args.pred_horizon))
    grad_fn = nn.value_and_grad(
        model,
        lambda m, x, a, p, c: loss_fn(m, x, a, p, c, action_weights, loss_weights),
    )
    rng = np.random.default_rng(args.seed)
    n_train = len(train_data["feature"])
    n_val = len(val_data["feature"])
    log = []
    for it in range(args.iters):
        idx = rng.integers(0, n_train, size=args.batch)
        loss, grads = grad_fn(
            model,
            mx.array(train_data["feature"][idx]),
            mx.array(train_data["action_chunk"][idx]),
            mx.array(train_data["future_pose"][idx]),
            mx.array(train_data["future_contact"][idx]),
        )
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if it % args.log_every == 0 or it == args.iters - 1:
            vidx = rng.choice(n_val, size=min(args.eval_batch, n_val), replace=False)
            parts = eval_components(
                model,
                mx.array(val_data["feature"][vidx]),
                mx.array(val_data["action_chunk"][vidx]),
                mx.array(val_data["future_pose"][vidx]),
                mx.array(val_data["future_contact"][vidx]),
                action_weights,
            )
            row = {"iter": int(it), "train_loss": float(loss.item()), **parts}
            log.append(row)
            print(
                f"[{it:5d}] train={row['train_loss']:.6f} "
                f"chunk={row['action_loss']:.6f} pose={row['pose_loss']:.6f} "
                f"contact={row['contact_loss']:.6f}"
            )

    ckpt = os.path.join(args.out_dir, "pusht_causal_student_v2.npz")
    save_model(model, ckpt)
    knn_bank = build_knn_bank(train_data)
    knn_bank_path = os.path.join(args.out_dir, "pusht_causal_student_v2_knn_bank.npz")
    save_knn_bank(knn_bank, knn_bank_path)
    eval_row_model = evaluate(model, data, val_eps, args)
    eval_row_knn = evaluate_knn(knn_bank, data, val_eps, args)
    summary = {
        "ckpt": os.path.abspath(ckpt),
        "knn_bank": os.path.abspath(knn_bank_path),
        "dataset_version": DATASET_VERSION,
        "train_episodes_requested": train_eps,
        "val_episodes_requested": val_eps,
        "train_episodes_used": train_data["episodes_used"],
        "val_episodes_used": val_data["episodes_used"],
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "teacher_train_success_rate": train_data["teacher_success_rate"],
        "teacher_train_relaxed_success_rate": train_data["teacher_relaxed_success_rate"],
        "teacher_val_success_rate": val_data["teacher_success_rate"],
        "teacher_val_relaxed_success_rate": val_data["teacher_relaxed_success_rate"],
        "student_eval": eval_row_knn,
        "student_eval_knn": eval_row_knn,
        "student_eval_model": eval_row_model,
        "final": log[-1],
        "real_training": True,
        "contract": "chunked pose-aware BC from causal-tree rollouts with kNN exemplar inference",
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out_dir", default="meadow/pusht_causal_student_v2")
    ap.add_argument("--train_episodes", default="0,61,144,238,18,52,77,105")
    ap.add_argument("--val_episodes", default="4,17,83,126")
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--chunk_exec", type=int, default=3)
    ap.add_argument("--pred_horizon", type=int, default=4)
    ap.add_argument("--recovery_horizon", type=int, default=64)
    ap.add_argument("--hold_frames", type=int, default=50)
    ap.add_argument("--eval_hold_frames", type=int, default=50)
    ap.add_argument("--expert_guard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--expert_guard_margin", type=float, default=25.0)
    ap.add_argument("--expert_guard_pos_slack", type=float, default=5.0)
    ap.add_argument("--expert_guard_angle_slack", type=float, default=0.05)
    ap.add_argument("--accept_candidate_pos", type=float, default=0.0)
    ap.add_argument("--accept_candidate_angle", type=float, default=0.0)
    ap.add_argument("--accept_candidate_max_pos", type=float, default=0.0)
    ap.add_argument("--accept_candidate_max_angle", type=float, default=0.0)
    ap.add_argument("--precision_phase_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--precision_phase_pos", type=float, default=20.0)
    ap.add_argument("--precision_phase_angle", type=float, default=float(np.pi / 9))
    ap.add_argument("--terminal_precision_teacher", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--precision_candidates", type=int, default=24)
    ap.add_argument("--precision_score_horizon", type=int, default=14)
    ap.add_argument("--precision_recovery_horizon", type=int, default=10)
    ap.add_argument("--term_pos_weight", type=float, default=1.0)
    ap.add_argument("--term_axis_weight", type=float, default=0.45)
    ap.add_argument("--term_angle_weight", type=float, default=58.0)
    ap.add_argument("--term_score_weight", type=float, default=1.8)
    ap.add_argument("--term_raw_angle_weight", type=float, default=58.0)
    ap.add_argument("--term_micro_bonus", type=float, default=180.0)
    ap.add_argument("--term_precision_prior", type=float, default=8.0)
    ap.add_argument("--term_min_current_improve", type=float, default=0.8)
    ap.add_argument("--term_min_expert_improve", type=float, default=0.4)
    ap.add_argument("--extra_precision_steps", type=int, default=0)
    ap.add_argument("--continue_after_env_success", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--micro_hold_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--min_tail_steps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=2200)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--eval_batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--pose_weight", type=float, default=0.18)
    ap.add_argument("--contact_weight", type=float, default=0.06)
    ap.add_argument("--guide_blend", type=float, default=0.18)
    ap.add_argument("--guide_blend_contact_thresh", type=float, default=0.35)
    ap.add_argument("--guide_blend_min_pos", type=float, default=28.0)
    ap.add_argument("--student_terminal_safety", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--student_terminal_hold", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--student_hold_mode", choices=("zero", "active", "release"), default="zero")
    ap.add_argument("--student_safety_pos", type=float, default=18.0)
    ap.add_argument("--student_safety_angle", type=float, default=0.20)
    ap.add_argument("--student_safety_blend", type=float, default=0.55)
    ap.add_argument("--student_hold_pos", type=float, default=8.0)
    ap.add_argument("--student_hold_angle", type=float, default=0.08)
    ap.add_argument("--student_hold_pos_gain", type=float, default=0.34)
    ap.add_argument("--student_hold_vel_gain", type=float, default=0.025)
    ap.add_argument("--student_hold_contact_dist", type=float, default=30.0)
    ap.add_argument("--student_hold_contact_tol", type=float, default=13.0)
    ap.add_argument("--student_hold_min_action", type=float, default=0.012)
    ap.add_argument("--student_hold_max_action", type=float, default=0.055)
    ap.add_argument("--student_hold_norm_scale", type=float, default=90.0)
    ap.add_argument("--student_hold_clip", type=float, default=0.10)
    ap.add_argument("--student_release_pos", type=float, default=4.5)
    ap.add_argument("--student_release_angle", type=float, default=0.05)
    ap.add_argument("--student_release_action", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--knn_k", type=int, default=9)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        if ap.get_default("train_episodes") == args.train_episodes:
            args.train_episodes = "0,61,144,238,18,52"
        if ap.get_default("val_episodes") == args.val_episodes:
            args.val_episodes = "4,17,83"
        args.recovery_horizon = min(args.recovery_horizon, 96)
        args.iters = min(args.iters, 1800)
        args.batch = min(args.batch, 160)
        args.log_every = min(args.log_every, 150)
    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
