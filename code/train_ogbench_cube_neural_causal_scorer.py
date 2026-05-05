"""Train an OGBench Cube neural causal scorer.

This is the real OGBench robot-arm Cube task counterpart to the local
TwoRoom/PushT/Cube scorers:

OGBench cube-single state/proprio + candidate grasp-lift-place future
      -> neural score for whether the cube reaches the target position.

Python split:
  - Python 3.11: OGBench/MuJoCo case collection and optional render frames.
  - Python 3.14: MLX training and video assembly.

The scorer is a v1 bridge, not a full policy/world model. It uses real OGBench
initial/goal states and renders, while candidate manipulation futures are a
fast reach/grasp/lift/place program library in OGBench coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_NAME = "cube-single-play-v0"
DATASET_VERSION = "ogbench_cube_grasp_lift_place_v3_precision"
IMG_SIZE = 32
PATH_POINTS = 20
SUCCESS_POS = 0.015
WORLD_W = 860
WORLD_H = 560
FPS = 24


def _import_mlx():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    return mx, nn, optim, tree_flatten, tree_unflatten


def py311():
    return os.environ.get("OGBENCH_PY", "python3.11")


def as_f32(x):
    return np.asarray(x, dtype=np.float32)


def yaw_features(yaw):
    return np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)


def case_scene_feature(case):
    task = np.zeros(5, dtype=np.float32)
    task[int(case["task_id"]) - 1] = 1.0
    feat = np.concatenate(
        [
            as_f32(case["obs"]),
            as_f32(case["goal_obs"]),
            as_f32(case["effector_pos"]),
            yaw_features(case["effector_yaw"]),
            as_f32([case["gripper_opening"], case["gripper_contact"]]),
            as_f32(case["block_pos"]),
            yaw_features(case["block_yaw"]),
            as_f32(case["goal_pos"]),
            task,
        ]
    )
    return feat.astype(np.float32)


def candidate_future(case, lateral=0.0, lift=0.17, overshoot=0.0, mode="servo", steps=88):
    """Fast candidate reach/grasp/lift/place future in OGBench coords.

    Row layout:
      effector xyz, cube xyz, gripper_closed, contact_flag
    """
    eff0 = as_f32(case["effector_pos"])
    block0 = as_f32(case["block_pos"])
    goal = as_f32(case["goal_pos"])
    xy_dir = goal[:2] - block0[:2]
    norm = float(np.linalg.norm(xy_dir))
    if norm < 1e-6:
        xy_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        xy_dir = (xy_dir / norm).astype(np.float32)
    perp = np.array([-xy_dir[1], xy_dir[0]], dtype=np.float32)

    pick = block0.copy()
    pick[:2] += perp * lateral
    pick[2] = block0[2]
    pick_above = pick.copy()
    pick_above[2] = max(block0[2] + 0.12, lift)

    place = goal.copy()
    place[:2] += xy_dir * overshoot + perp * lateral * 0.15
    place[2] = goal[2]
    place_above = place.copy()
    place_above[2] = max(goal[2] + 0.12, lift)

    clearance = (pick_above + place_above) * 0.5
    clearance[:2] += perp * lateral * 0.4
    clearance[2] = max(pick_above[2], place_above[2]) + 0.02

    if mode == "drop":
        carry_alpha = 0.55
        final_alpha = 0.62
    elif mode == "miss":
        carry_alpha = 1.0
        final_alpha = 1.0
        place[:2] += perp * 0.08 + xy_dir * 0.06
        place_above[:2] = place[:2]
    elif mode == "hover":
        carry_alpha = 1.0
        final_alpha = 1.0
        place[2] = goal[2] + 0.11
    elif mode == "over":
        carry_alpha = 1.0
        final_alpha = 1.0
        place[:2] += xy_dir * 0.055
        place_above[:2] = place[:2]
    else:
        carry_alpha = 1.0
        final_alpha = 1.0

    pts = []
    for t in range(steps):
        u = (t + 1) / steps
        if u < 0.16:
            a = u / 0.16
            eff = eff0 * (1 - a) + pick_above * a
            cube = block0.copy()
            grip = 0.0
            contact = 0.0
        elif u < 0.28:
            a = (u - 0.16) / 0.12
            eff = pick_above * (1 - a) + pick * a
            cube = block0.copy()
            grip = 0.0
            contact = 0.25
        elif u < 0.38:
            a = (u - 0.28) / 0.10
            eff = pick.copy()
            cube = block0.copy()
            grip = a
            contact = a
        elif u < 0.52:
            a = (u - 0.38) / 0.14
            eff = pick * (1 - a) + pick_above * a
            cube = block0 * (1 - a) + pick_above * a
            cube[2] -= 0.015
            grip = 1.0
            contact = 1.0
        elif u < 0.70:
            a = (u - 0.52) / 0.18
            eff = pick_above * (1 - a) + clearance * a
            cube = pick_above * (1 - a) + clearance * a
            cube[2] -= 0.015
            grip = 1.0
            contact = 1.0
        elif u < 0.84:
            a = (u - 0.70) / 0.14
            a *= carry_alpha
            eff = clearance * (1 - a) + place_above * a
            cube = clearance * (1 - a) + place_above * a
            cube[2] -= 0.015
            grip = 1.0
            contact = 1.0
        elif u < 0.94:
            a = (u - 0.84) / 0.10
            a *= final_alpha
            eff = place_above * (1 - a) + place * a
            cube = place_above * (1 - a) + place * a
            cube[2] = max(float(goal[2]), float(cube[2] - 0.015 * (1 - a)))
            grip = 1.0
            contact = 1.0
        else:
            a = (u - 0.94) / 0.06
            eff = place.copy()
            cube = place.copy()
            grip = 1.0 - a
            contact = 1.0 - a
        pts.append(np.concatenate([eff, cube, [grip, contact]]).astype(np.float32))
    return np.asarray(pts, dtype=np.float32)


def candidate_paths(case, count=160):
    paths = []
    for j in range(count):
        if j < count // 3:
            lateral = np.linspace(-0.035, 0.035, max(1, count // 3))[j]
            lift = [0.14, 0.17, 0.20][j % 3]
            paths.append(candidate_future(case, lateral=float(lateral), lift=lift, mode="servo"))
        elif j < 2 * count // 3:
            k = j - count // 3
            lateral = math.sin(k * 1.7) * 0.045
            lift = 0.13 + 0.08 * ((k % 7) / 6.0)
            mode = ["servo", "over", "miss", "drop"][k % 4]
            overshoot = [-0.03, -0.012, 0.0, 0.018, 0.04][(k // 4) % 5]
            paths.append(candidate_future(case, lateral=float(lateral), lift=lift, overshoot=overshoot, mode=mode))
        else:
            k = j - 2 * count // 3
            lateral = [-0.075, -0.05, -0.025, 0.025, 0.05, 0.075][k % 6]
            mode = ["miss", "drop", "hover", "over"][k % 4]
            paths.append(candidate_future(case, lateral=float(lateral), lift=0.18, overshoot=0.055, mode=mode))
    return paths


def sampled_path_features(path, n_points=PATH_POINTS):
    idx = np.linspace(0, len(path) - 1, n_points)
    lo = np.floor(idx).astype(np.int32)
    hi = np.ceil(idx).astype(np.int32)
    frac = (idx - lo).astype(np.float32)[:, None]
    pts = path[lo] * (1.0 - frac) + path[hi] * frac
    out = pts.copy()
    center = np.array([0.425, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.25, 0.35, 0.35], dtype=np.float32)
    out[:, 0:3] = (out[:, 0:3] - center) / scale
    out[:, 3:6] = (out[:, 3:6] - center) / scale
    return out.astype(np.float32).reshape(-1)


def path_label(path, goal_pos):
    d = np.linalg.norm(path[:, 3:6] - as_f32(goal_pos)[None, :], axis=1)
    best = float(np.min(d))
    return float(best <= SUCCESS_POS), best


def truncate_path_at_goal(path, goal_pos):
    d = np.linalg.norm(path[:, 3:6] - as_f32(goal_pos)[None, :], axis=1)
    i = int(np.argmin(d))
    if float(d[i]) <= SUCCESS_POS:
        return path[: i + 1].copy()
    return path[: i + 1].copy()


def collect_cases(count, seed, dataset_name=DATASET_NAME):
    import ogbench

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    cases = []
    for i in range(count):
        task_id = i % 5 + 1
        obs, info = env.reset(seed=seed + i, options={"task_id": task_id, "render_goal": True})
        unwrapped = env.unwrapped
        goal_pos = np.asarray(unwrapped.cur_task_info["goal_xyzs"][0], dtype=np.float32)
        goal_obs = np.asarray(info.get("goal", np.zeros_like(obs)), dtype=np.float32)
        frame = env.render()
        cases.append(
            {
                "dataset_name": dataset_name,
                "reset_seed": int(seed + i),
                "task_id": int(task_id),
                "obs": np.asarray(obs, dtype=np.float32).tolist(),
                "goal_obs": goal_obs.tolist(),
                "effector_pos": np.asarray(info["proprio/effector_pos"], dtype=np.float32).tolist(),
                "effector_yaw": float(np.asarray(info["proprio/effector_yaw"]).reshape(-1)[0]),
                "gripper_opening": float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]),
                "gripper_contact": float(np.asarray(info["proprio/gripper_contact"]).reshape(-1)[0]),
                "block_pos": np.asarray(info["privileged/block_0_pos"], dtype=np.float32).tolist(),
                "block_yaw": float(np.asarray(info["privileged/block_0_yaw"]).reshape(-1)[0]),
                "goal_pos": goal_pos.tolist(),
                "frame": np.asarray(frame, dtype=np.uint8),
            }
        )
    env.close()
    return cases


def collect_real_rollout(case, reset_seed, frames=210, dataset_name=DATASET_NAME):
    """Roll out a real MuJoCo grasp-lift-place controller and return frames.

    This is for visualization only. The neural scorer still selects the causal
    candidate on the right panel; the left panel shows a real OGBench robot arm
    executing a compatible pick-and-place motion so the video is not just a
    reset image.
    """
    import ogbench
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
    ob, info = env.reset(seed=int(reset_seed), options={"task_id": int(case["task_id"]), "render_goal": True})
    goal = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][0], dtype=np.float32)
    oracle = CubeMarkovOracle(env=env)

    def oracle_info(info_now):
        out = dict(info_now)
        out["privileged/target_block"] = 0
        out["privileged/target_block_pos"] = goal.copy()
        out["privileged/target_block_yaw"] = np.array([0.0], dtype=np.float32)
        return out

    oracle.reset(ob, oracle_info(info))
    imgs, cube_pos, eff_pos, gripper_contact, gripper_opening = [], [], [], [], []

    def snapshot():
        info_now = env.unwrapped.compute_ob_info()
        imgs.append(np.asarray(env.render(), dtype=np.uint8))
        cube_pos.append(np.asarray(info_now["privileged/block_0_pos"], dtype=np.float32).copy())
        eff_pos.append(np.asarray(info_now["proprio/effector_pos"], dtype=np.float32).copy())
        gripper_contact.append(float(np.asarray(info_now["proprio/gripper_contact"]).reshape(-1)[0]))
        gripper_opening.append(float(np.asarray(info_now["proprio/gripper_opening"]).reshape(-1)[0]))
        return info_now

    for _ in range(frames):
        info_now = snapshot()
        if oracle.done:
            continue
        action = oracle.select_action(ob, oracle_info(info_now))
        ob, _, _, _, info = env.step(action)

    final_info = env.unwrapped.compute_ob_info()
    final_block = np.asarray(final_info["privileged/block_0_pos"], dtype=np.float32)
    cube_arr = np.asarray(cube_pos, dtype=np.float32)
    env.close()
    max_cube_z = float(np.max(cube_arr[:, 2])) if len(cube_arr) else float(final_block[2])
    rollout = {
        "frames": np.asarray(imgs, dtype=np.uint8),
        "cube_pos": cube_arr,
        "eff_pos": np.asarray(eff_pos, dtype=np.float32),
        "goal_pos": goal.astype(np.float32),
        "final_dist": float(np.linalg.norm(final_block - goal)),
        "max_cube_z": max_cube_z,
        "lifted": bool(max_cube_z >= 0.08),
        "gripper_contact": np.asarray(gripper_contact, dtype=np.float32),
        "gripper_opening": np.asarray(gripper_opening, dtype=np.float32),
        "controller_search_best_dist": float(np.linalg.norm(final_block - goal)),
        "controller_config": {
            "type": "ogbench_cube_markov_oracle",
            "mode": "grasp_lift_place",
            "success_pos": SUCCESS_POS,
        },
    }
    return rollout


def build_dataset_from_cases(cases, candidates):
    scenes, path_feats, labels, min_dist, case_ids = [], [], [], [], []
    meta_cases = []
    frames = []
    for cid, case in enumerate(cases):
        scene = case_scene_feature(case)
        paths = candidate_paths(case, count=candidates)
        frames.append(np.asarray(case["frame"], dtype=np.uint8))
        meta_cases.append({k: v for k, v in case.items() if k != "frame"})
        for p in paths:
            y, d = path_label(p, case["goal_pos"])
            scenes.append(scene)
            path_feats.append(sampled_path_features(p))
            labels.append(y)
            min_dist.append(d)
            case_ids.append(cid)
    return {
        "scene": np.asarray(scenes, dtype=np.float32),
        "path": np.asarray(path_feats, dtype=np.float32),
        "label": np.asarray(labels, dtype=np.float32),
        "min_dist": np.asarray(min_dist, dtype=np.float32),
        "case_id": np.asarray(case_ids, dtype=np.int32),
        "frames": np.asarray(frames, dtype=np.uint8),
        "cases": meta_cases,
        "n_cases": len(cases),
        "candidates": candidates,
    }


def _collect_cmd(args):
    cases = collect_cases(args.cases, args.seed, args.dataset_name)
    data = build_dataset_from_cases(cases, args.candidates)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        scene=data["scene"],
        path=data["path"],
        label=data["label"],
        min_dist=data["min_dist"],
        case_id=data["case_id"],
        frames=data["frames"],
    )
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(data["cases"], f, indent=2)
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump(
            {
                "dataset_version": DATASET_VERSION,
                "dataset_name": args.dataset_name,
                "success_pos": SUCCESS_POS,
                "candidates": args.candidates,
                "cases": args.cases,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(
        json.dumps(
            {
                "out": str(out),
                "dataset_version": DATASET_VERSION,
                "cases": data["n_cases"],
                "samples": int(len(data["label"])),
                "positive_rate": float(data["label"].mean()),
                "oracle_success": int(sum(np.max(data["label"][data["case_id"] == i]) >= 0.5 for i in range(data["n_cases"]))),
            },
            indent=2,
        )
    )


def _rollout_cmd(args):
    with open(args.case_json) as f:
        case = json.load(f)
    rollout = collect_real_rollout(case, args.reset_seed, frames=args.frames, dataset_name=args.dataset_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frames=rollout["frames"],
        cube_pos=rollout["cube_pos"],
        eff_pos=rollout["eff_pos"],
        goal_pos=rollout["goal_pos"],
        final_dist=np.asarray([rollout["final_dist"]], dtype=np.float32),
        max_cube_z=np.asarray([rollout["max_cube_z"]], dtype=np.float32),
        lifted=np.asarray([rollout["lifted"]], dtype=np.bool_),
        gripper_contact=rollout["gripper_contact"],
        gripper_opening=rollout["gripper_opening"],
        controller_search_best_dist=np.asarray([rollout["controller_search_best_dist"]], dtype=np.float32),
        controller_config_json=np.asarray(json.dumps(rollout["controller_config"]), dtype=np.str_),
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "frames": int(len(rollout["frames"])),
                "final_dist": rollout["final_dist"],
                "max_cube_z": rollout["max_cube_z"],
                "lifted": rollout["lifted"],
                "controller_search_best_dist": rollout["controller_search_best_dist"],
                "controller_config": rollout["controller_config"],
                "real_ogbench_rollout": True,
            },
            indent=2,
        )
    )


def ensure_collected(split, out_dir, cases, candidates, seed, dataset_name=DATASET_NAME):
    npz = Path(out_dir) / f"{split}_samples.npz"
    js = npz.with_suffix(".json")
    meta = npz.with_suffix(".meta.json")
    if npz.exists() and js.exists() and meta.exists():
        try:
            with open(meta) as f:
                m = json.load(f)
            if (
                m.get("dataset_version") == DATASET_VERSION
                and m.get("dataset_name") == dataset_name
                and int(m.get("candidates", -1)) == int(candidates)
                and int(m.get("cases", -1)) == int(cases)
                and int(m.get("seed", -1)) == int(seed)
                and abs(float(m.get("success_pos", -1.0)) - SUCCESS_POS) < 1e-9
            ):
                return npz
        except Exception:
            pass
    elif npz.exists() or js.exists():
        print(f"[data] refreshing stale {split} data for {DATASET_VERSION}")
    for stale in [npz, js, meta, Path(out_dir) / f"{split}_cases.json"]:
        stale.unlink(missing_ok=True)
    cmd = [
        py311(),
        os.path.abspath(__file__),
        "_collect",
        "--out",
        str(npz),
        "--cases",
        str(cases),
        "--candidates",
        str(candidates),
        "--seed",
        str(seed),
        "--dataset_name",
        dataset_name,
    ]
    subprocess.run(cmd, check=True)
    return npz


def load_collected(npz_path):
    arr = np.load(npz_path)
    with open(Path(npz_path).with_suffix(".json")) as f:
        cases = json.load(f)
    return {
        "scene": arr["scene"].astype(np.float32),
        "path": arr["path"].astype(np.float32),
        "label": arr["label"].astype(np.float32),
        "min_dist": arr["min_dist"].astype(np.float32),
        "case_id": arr["case_id"].astype(np.int32),
        "frames": arr["frames"],
        "cases": cases,
        "n_cases": len(cases),
    }


class CausalPathScorer:
    pass


def make_model(input_dim, hidden=256):
    mx, nn, *_ = _import_mlx()

    class _CausalPathScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.l3 = nn.Linear(hidden, 128)
            self.out = nn.Linear(128, 1)

        def __call__(self, scene, path):
            x = mx.concatenate([scene, path], axis=-1)
            x = nn.relu(self.l1(x))
            x = nn.relu(self.l2(x))
            x = nn.relu(self.l3(x))
            return self.out(x).squeeze(-1)

    return _CausalPathScorer()


def bce_logits(logits, labels, pos_weight=1.0):
    mx, *_ = _import_mlx()
    loss = mx.maximum(logits, 0) - logits * labels + mx.log1p(mx.exp(-mx.abs(logits)))
    weights = 1.0 + labels * (pos_weight - 1.0)
    return mx.mean(loss * weights)


def sigmoid_np(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def eval_batch(model, data, idx):
    mx, *_ = _import_mlx()
    logits = model(mx.array(data["scene"][idx]), mx.array(data["path"][idx]))
    mx.eval(logits)
    probs = sigmoid_np(np.array(logits, dtype=np.float32))
    y = data["label"][idx]
    return float(np.mean((probs >= 0.5) == (y >= 0.5))), float(np.mean(np.abs(probs - y)))


def eval_selection(model, data):
    mx, *_ = _import_mlx()
    selected_success = 0
    oracle_success = 0
    top5_success = 0
    selected_dists = []
    for cid in range(data["n_cases"]):
        idx = np.where(data["case_id"] == cid)[0]
        logits = model(mx.array(data["scene"][idx]), mx.array(data["path"][idx]))
        mx.eval(logits)
        scores = np.array(logits, dtype=np.float32)
        order = np.argsort(-scores)
        labels = data["label"][idx]
        dists = data["min_dist"][idx]
        selected_success += int(labels[order[0]] >= 0.5)
        oracle_success += int(np.max(labels) >= 0.5)
        top5_success += int(np.max(labels[order[:5]]) >= 0.5)
        selected_dists.append(float(dists[order[0]]))
    n = max(1, data["n_cases"])
    return {
        "selected_success": selected_success,
        "selected_success_rate": selected_success / n,
        "oracle_success": oracle_success,
        "oracle_success_rate": oracle_success / n,
        "top5_success": top5_success,
        "top5_success_rate": top5_success / n,
        "selected_dist_mean": float(np.mean(selected_dists)),
        "selected_dist_median": float(np.median(selected_dists)),
    }


def save_model(model, out_path):
    mx, nn, optim, tree_flatten, _ = _import_mlx()
    mx.savez(out_path, **dict(tree_flatten(model.parameters())))


def load_model(ckpt, scene_dim, path_dim, hidden=256):
    mx, nn, optim, tree_flatten, tree_unflatten = _import_mlx()
    model = make_model(scene_dim + path_dim, hidden)
    mx.eval(model.parameters())
    model.update(tree_unflatten(list(mx.load(ckpt).items())))
    mx.eval(model.parameters())
    return model


def train(args):
    mx, nn, optim, *_ = _import_mlx()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = vars(args) | {
        "dataset_version": DATASET_VERSION,
        "dataset_name": args.dataset_name,
        "success_pos": SUCCESS_POS,
        "path_points": PATH_POINTS,
        "real_ogbench_case_source": True,
        "candidate_source": "fast reach/grasp/lift/place futures in OGBench coordinates",
        "task": "OGBench Cube pick-and-place hybrid neural causal scoring",
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("[device]", mx.default_device())
    print("[data] collecting/loading OGBench Cube cases via Python 3.11...")
    t0 = time.time()
    train_npz = ensure_collected("train", args.out_dir, args.train_cases, args.candidates, args.seed, args.dataset_name)
    val_npz = ensure_collected("val", args.out_dir, args.val_cases, args.candidates, args.seed + 1000, args.dataset_name)
    train_data = load_collected(train_npz)
    val_data = load_collected(val_npz)
    # Store plan-compatible names.
    shutil.copyfile(Path(train_npz).with_suffix(".json"), os.path.join(args.out_dir, "train_cases.json"))
    shutil.copyfile(Path(val_npz).with_suffix(".json"), os.path.join(args.out_dir, "val_cases.json"))
    print(
        f"[data] train={len(train_data['label'])} pos={train_data['label'].mean():.3f} "
        f"val={len(val_data['label'])} pos={val_data['label'].mean():.3f} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    input_dim = train_data["scene"].shape[1] + train_data["path"].shape[1]
    model = make_model(input_dim, args.hidden)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    pos_rate = float(train_data["label"].mean())
    pos_weight = min(8.0, max(1.0, (1.0 - pos_rate) / max(pos_rate, 1e-4)))

    def loss_fn(m, scene, path, label):
        return bce_logits(m(scene, path), label, pos_weight=pos_weight)

    grad_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)
    n_train = len(train_data["label"])
    n_val = len(val_data["label"])
    log = []
    t_train = time.time()
    for it in range(args.iters):
        idx = rng.integers(0, n_train, size=args.batch)
        loss, grads = grad_fn(
            model,
            mx.array(train_data["scene"][idx]),
            mx.array(train_data["path"][idx]),
            mx.array(train_data["label"][idx]),
        )
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if it % args.log_every == 0 or it == args.iters - 1:
            vidx = rng.choice(n_val, size=min(args.eval_batch, n_val), replace=False)
            acc, mae = eval_batch(model, val_data, vidx)
            sel = eval_selection(model, val_data)
            row = {"iter": it, "train_loss": float(loss.item()), "val_acc": acc, "val_prob_mae": mae, **sel}
            log.append(row)
            print(
                f"[{it:5d}] loss={row['train_loss']:.4f} acc={acc:.3f} "
                f"select={sel['selected_success']}/{val_data['n_cases']} "
                f"top5={sel['top5_success']}/{val_data['n_cases']} "
                f"oracle={sel['oracle_success']}/{val_data['n_cases']} "
                f"med_dist={sel['selected_dist_median']:.4f} "
                f"elapsed={time.time()-t_train:.1f}s"
            )

    ckpt = os.path.join(args.out_dir, "neural_causal_scorer.npz")
    save_model(model, ckpt)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    summary = {
        "ckpt": ckpt,
        "dataset_name": args.dataset_name,
        "dataset_version": DATASET_VERSION,
        "success_pos": SUCCESS_POS,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "train_positive_rate": float(train_data["label"].mean()),
        "val_positive_rate": float(val_data["label"].mean()),
        "final": log[-1],
        "real_ogbench_env": True,
        "real_training": True,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))
    return ckpt


def font(size=18):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_caption(draw, text):
    draw.rectangle([18, 14, WORLD_W - 18, 52], fill=(255, 255, 255, 238), outline=(224, 228, 236))
    draw.text((30, 24), text, fill=(38, 43, 52), font=font(18))


def world_to_panel(xy, panel):
    x0, y0, x1, y1 = panel
    x = x0 + (float(xy[0]) - 0.25) / 0.35 * (x1 - x0)
    y = y1 - (float(xy[1]) + 0.35) / 0.70 * (y1 - y0)
    return x, y


def draw_future(draw, path, panel, color, width=2, alpha=70, upto=None):
    if upto is None:
        upto = len(path)
    block_pts = [world_to_panel(row[3:5], panel) for row in path[: max(2, upto)]]
    eff_pts = [world_to_panel(row[0:2], panel) for row in path[: max(2, upto)]]
    if len(block_pts) > 1:
        draw.line(block_pts, fill=(*color, alpha), width=width)
    if len(eff_pts) > 1:
        draw.line(eff_pts, fill=(37, 99, 235, max(20, alpha // 2)), width=max(1, width - 1))


def load_real_rollout(case, out_path, reset_seed, frames, dataset_name):
    rollout_npz = Path(out_path).with_suffix(".real_rollout.npz")
    case_json = Path(out_path).with_suffix(".case.json")
    with open(case_json, "w") as f:
        json.dump(case, f)
    cmd = [
        py311(),
        os.path.abspath(__file__),
        "_rollout",
        "--case_json",
        str(case_json),
        "--out",
        str(rollout_npz),
        "--reset_seed",
        str(reset_seed),
        "--frames",
        str(frames),
        "--dataset_name",
        dataset_name,
    ]
    subprocess.run(cmd, check=True)
    case_json.unlink(missing_ok=True)
    arr = np.load(rollout_npz)
    return {
        "frames": arr["frames"],
        "cube_pos": arr["cube_pos"],
        "eff_pos": arr["eff_pos"],
        "goal_pos": arr["goal_pos"],
        "final_dist": float(arr["final_dist"][0]),
        "max_cube_z": float(arr["max_cube_z"][0]) if "max_cube_z" in arr else 0.0,
        "lifted": bool(arr["lifted"][0]) if "lifted" in arr else False,
        "gripper_contact": arr["gripper_contact"] if "gripper_contact" in arr else np.zeros(len(arr["cube_pos"]), dtype=np.float32),
        "gripper_opening": arr["gripper_opening"] if "gripper_opening" in arr else np.zeros(len(arr["cube_pos"]), dtype=np.float32),
        "controller_search_best_dist": float(arr["controller_search_best_dist"][0])
        if "controller_search_best_dist" in arr
        else float(arr["final_dist"][0]),
        "controller_config": json.loads(str(arr["controller_config_json"].item()))
        if "controller_config_json" in arr
        else {},
    }


def record_case(model, out_path, case, rollout, candidates=180, frames=210):
    mx, *_ = _import_mlx()
    paths = candidate_paths(case, candidates)
    scene = case_scene_feature(case)
    path_feat = np.asarray([sampled_path_features(p) for p in paths], dtype=np.float32)
    scene_batch = np.repeat(scene[None], len(paths), axis=0)
    logits = model(mx.array(scene_batch), mx.array(path_feat))
    mx.eval(logits)
    scores = np.array(logits, dtype=np.float32)
    order = np.argsort(-scores)
    selected = truncate_path_at_goal(paths[int(order[0])], case["goal_pos"])
    top_paths = [paths[int(i)] for i in order[:12]]
    frame_dir = out_path + "_frames"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    rollout_frames = rollout["frames"]
    panel = (450, 96, 820, 466)
    exec_i = 0
    exec_row = np.concatenate([case["effector_pos"], case["block_pos"], [case["gripper_opening"], 0.0]]).astype(np.float32)
    for fr in range(frames):
        if fr >= 84 and exec_i < len(selected) - 1:
            dist = float(np.linalg.norm(exec_row[3:6] - as_f32(case["goal_pos"])))
            if dist > SUCCESS_POS:
                exec_i += 1
                exec_row = selected[exec_i].copy()

        im = Image.new("RGB", (WORLD_W, WORLD_H), (249, 250, 252))
        draw = ImageDraw.Draw(im, "RGBA")
        draw.rectangle([32, 78, 408, 482], fill=(255, 255, 255), outline=(219, 225, 235), width=2)
        rf = Image.fromarray(rollout_frames[min(fr, len(rollout_frames) - 1)]).resize((360, 360), Image.Resampling.LANCZOS)
        im.paste(rf, (40, 92))
        draw.text((48, 458), "real OGBench/MuJoCo grasp-lift-place rollout", fill=(71, 85, 105), font=font(15))

        draw.rectangle([430, 78, 840, 482], fill=(255, 255, 255), outline=(219, 225, 235), width=2)
        draw.text((454, 82), "candidate grasp-lift-place futures in table coordinates", fill=(71, 85, 105), font=font(15))
        draw.rectangle(panel, outline=(203, 213, 225), width=1)
        gx, gy = world_to_panel(case["goal_pos"][:2], panel)
        bx, by = world_to_panel(case["block_pos"][:2], panel)
        ex, ey = world_to_panel(case["effector_pos"][:2], panel)
        draw.ellipse([gx - 8, gy - 8, gx + 8, gy + 8], outline=(22, 101, 52, 230), width=3)
        draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(220, 38, 38, 190))
        draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(37, 99, 235, 190))

        if fr < 42:
            caption = "OGBench Cube neural scorer: candidate robot-arm pick-and-place futures"
        elif fr < 84:
            caption = "neural scorer reinforces a grasp-lift-place causal chain"
        else:
            caption = "left: real robot arm grasps and places; right: neural-selected causal chain"

        progress = 1.0 if fr >= 42 else fr / 41.0
        for p in paths:
            upto = int(2 + progress * (len(p) - 2))
            draw_future(draw, p, panel, (8, 145, 178), width=1, alpha=34, upto=upto)
        if fr >= 42:
            conf = min(1.0, (fr - 42) / 42.0)
            for p in top_paths[1:]:
                draw_future(draw, p, panel, (37, 99, 235), width=1, alpha=28)
            draw_future(draw, selected, panel, (37, 99, 235), width=max(2, int(2 + 3 * conf)), alpha=int(80 + 110 * conf))
            draw_future(draw, selected, panel, (220, 38, 38), width=max(2, int(2 + 2 * conf)), alpha=int(90 + 120 * conf))

        eb = world_to_panel(exec_row[3:5], panel)
        ee = world_to_panel(exec_row[0:2], panel)
        draw.ellipse([eb[0] - 8, eb[1] - 8, eb[0] + 8, eb[1] + 8], fill=(220, 38, 38, 230), outline=(127, 29, 29), width=2)
        draw.ellipse([ee[0] - 6, ee[1] - 6, ee[0] + 6, ee[1] + 6], fill=(37, 99, 235, 230))
        real_cube = rollout["cube_pos"][min(fr, len(rollout["cube_pos"]) - 1)]
        dist = float(np.linalg.norm(real_cube - as_f32(case["goal_pos"])))
        draw.rectangle([640, 390, 820, 452], fill=(255, 255, 255, 235), outline=(226, 232, 240))
        draw.text((654, 403), f"real cube-goal: {dist:.3f} m", fill=(38, 43, 52), font=font(16))
        draw.text((654, 425), f"max lift: {rollout['max_cube_z']:.3f} m", fill=(38, 43, 52), font=font(15))
        if dist <= SUCCESS_POS:
            draw.ellipse([790, 424, 808, 442], fill=(22, 163, 74, 230))

        draw_caption(draw, caption)
        im.save(os.path.join(frame_dir, f"frame_{fr:05d}.png"))

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            os.path.join(frame_dir, "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            out_path,
        ],
        check=True,
    )
    shutil.rmtree(frame_dir)


def make_gif(mp4_path):
    gif_path = os.path.splitext(mp4_path)[0] + ".gif"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            mp4_path,
            "-vf",
            "fps=12,scale=620:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            gif_path,
        ],
        check=True,
    )
    return gif_path


def record(args):
    val_npz = Path(args.out_dir) / "val_samples.npz"
    if not val_npz.exists():
        ensure_collected("val", args.out_dir, 8, args.candidates, args.seed + 1000, args.dataset_name)
    data = load_collected(val_npz)
    model = load_model(args.ckpt, data["scene"].shape[1], data["path"].shape[1], hidden=args.hidden)
    os.makedirs(args.video_dir, exist_ok=True)
    names = ["ogcube_base", "ogcube_high_goal", "ogcube_side_goal", "ogcube_hard_start"]
    outputs = []
    rollout_summary = []
    for i, name in enumerate(names):
        case = data["cases"][i % data["n_cases"]]
        mp4 = os.path.join(args.video_dir, f"{name}.mp4")
        reset_seed = int(case.get("reset_seed", args.seed + 1000 + i))
        rollout = load_real_rollout(case, mp4, reset_seed, args.frames, args.dataset_name)
        record_case(model, mp4, case, rollout, candidates=args.candidates, frames=args.frames)
        outputs.append(make_gif(mp4))
        rollout_summary.append(
            {
                "name": name,
                "task_id": int(case["task_id"]),
                "reset_seed": reset_seed,
                "final_dist": rollout["final_dist"],
                "success": bool(rollout["final_dist"] <= SUCCESS_POS),
                "max_cube_z": rollout["max_cube_z"],
                "lifted": rollout["lifted"],
                "controller_search_best_dist": rollout["controller_search_best_dist"],
                "controller_config": rollout["controller_config"],
            }
        )
        print(outputs[-1])
    with open(os.path.join(args.video_dir, "rollout_summary.json"), "w") as f:
        json.dump(rollout_summary, f, indent=2)
    return outputs


def smoke(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = ensure_collected("smoke", out_dir, args.cases, args.candidates, args.seed, args.dataset_name)
    data = load_collected(npz)
    oracle = sum(np.max(data["label"][data["case_id"] == i]) >= 0.5 for i in range(data["n_cases"]))
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "samples": int(len(data["label"])),
                "cases": data["n_cases"],
                "positive_rate": float(data["label"].mean()),
                "oracle_success": int(oracle),
                "oracle_success_rate": float(oracle / max(1, data["n_cases"])),
                "render_frames": list(data["frames"].shape),
                "real_ogbench_env": True,
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    col = sub.add_parser("_collect")
    col.add_argument("--out", required=True)
    col.add_argument("--cases", type=int, default=8)
    col.add_argument("--candidates", type=int, default=64)
    col.add_argument("--seed", type=int, default=19)
    col.add_argument("--dataset_name", default=DATASET_NAME)

    roll = sub.add_parser("_rollout")
    roll.add_argument("--case_json", required=True)
    roll.add_argument("--out", required=True)
    roll.add_argument("--reset_seed", type=int, required=True)
    roll.add_argument("--frames", type=int, default=210)
    roll.add_argument("--dataset_name", default=DATASET_NAME)

    sm = sub.add_parser("smoke")
    sm.add_argument("--out_dir", default="meadow/ogbench_cube_grasp_neural_causal_scorer")
    sm.add_argument("--cases", type=int, default=8)
    sm.add_argument("--candidates", type=int, default=64)
    sm.add_argument("--seed", type=int, default=19)
    sm.add_argument("--dataset_name", default=DATASET_NAME)

    tr = sub.add_parser("train")
    tr.add_argument("--train_cases", type=int, default=220)
    tr.add_argument("--val_cases", type=int, default=60)
    tr.add_argument("--candidates", type=int, default=160)
    tr.add_argument("--iters", type=int, default=1400)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--eval_batch", type=int, default=4096)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--wd", type=float, default=1e-5)
    tr.add_argument("--seed", type=int, default=19)
    tr.add_argument("--log_every", type=int, default=200)
    tr.add_argument("--dataset_name", default=DATASET_NAME)
    tr.add_argument("--out_dir", default="meadow/ogbench_cube_grasp_neural_causal_scorer")

    rec = sub.add_parser("record")
    rec.add_argument("--ckpt", default="meadow/ogbench_cube_grasp_neural_causal_scorer/neural_causal_scorer.npz")
    rec.add_argument("--out_dir", default="meadow/ogbench_cube_grasp_neural_causal_scorer")
    rec.add_argument("--video_dir", default="meadow/ogbench_cube_grasp_neural_causal_scorer/videos")
    rec.add_argument("--dataset_name", default=DATASET_NAME)
    rec.add_argument("--hidden", type=int, default=256)
    rec.add_argument("--candidates", type=int, default=180)
    rec.add_argument("--frames", type=int, default=210)
    rec.add_argument("--seed", type=int, default=19)

    args = ap.parse_args()
    if args.cmd == "_collect":
        _collect_cmd(args)
    elif args.cmd == "_rollout":
        _rollout_cmd(args)
    elif args.cmd == "smoke":
        smoke(args)
    elif args.cmd == "train":
        train(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
