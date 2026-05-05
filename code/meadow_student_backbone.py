"""Shared student backbone for Meadow causal-teacher distillation.

This module keeps the contract intentionally small:

  scene features + short history + task id
      -> shared latent
      -> action head
      -> physical-parameter head
      -> Unreal/engine-style parameter head

The action head is the solving contract. The physics/engine heads are metadata
probes for later generalization: by default their losses read a detached latent
and cannot pull the solving encoder away from the action objective. This keeps
"how to solve the task" separate from "which physical parameters describe the
world" until we have enough data to intentionally couple them.

The current envs are not Unreal Engine envs. The "engine" head is therefore an
engine-analog target: a shared UE/Chaos-style parameter vector distilled from
the task's real simulator parameters (Pymunk or MuJoCo) or from the hidden
teacher physics used to generate rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


TASK_NAMES = ("tworoom", "pusht", "reacher", "ogbench_cube")
PHYS_TARGET_NAMES = (
    "mass",
    "moment_or_inertia",
    "linear_damping",
    "angular_damping",
    "friction",
    "restitution",
    "actuator_gain",
    "goal_gain",
    "wind_x",
    "wind_y",
    "contact_stiffness",
    "contact_damping",
)
ENGINE_TARGET_NAMES = (
    "ue_mass_override",
    "ue_mass_scale",
    "ue_linear_damping",
    "ue_angular_damping",
    "ue_friction",
    "ue_restitution",
    "ue_gravity_x",
    "ue_gravity_y",
    "ue_inertia_scale",
    "ue_sleep_linear_threshold",
    "ue_max_angular_velocity",
    "ue_constraint_stiffness",
)


def task_vector(task: str) -> np.ndarray:
    vec = np.zeros(len(TASK_NAMES), dtype=np.float32)
    vec[TASK_NAMES.index(task)] = 1.0
    return vec


def batch_task_vector(task: str, n: int) -> np.ndarray:
    return np.repeat(task_vector(task)[None, :], n, axis=0).astype(np.float32)


def pad_history(rows: list[np.ndarray], hist_steps: int, row_dim: int) -> np.ndarray:
    if not rows:
        return np.zeros(hist_steps * row_dim, dtype=np.float32)
    clipped = rows[-hist_steps:]
    if len(clipped) < hist_steps:
        pad = [np.zeros(row_dim, dtype=np.float32) for _ in range(hist_steps - len(clipped))]
        clipped = pad + clipped
    return np.concatenate(clipped, axis=0).astype(np.float32)


def mse(a, b):
    return mx.mean((a - b) ** 2)


@dataclass
class LossWeights:
    action: float = 1.0
    phys: float = 0.15
    engine: float = 0.10


class SharedPhysicsStudent(nn.Module):
    def __init__(
        self,
        scene_dim: int,
        history_dim: int,
        action_dim: int,
        hidden: int = 256,
        latent: int = 160,
        phys_dim: int = len(PHYS_TARGET_NAMES),
        engine_dim: int = len(ENGINE_TARGET_NAMES),
        task_dim: int = len(TASK_NAMES),
        couple_physics_grad: bool = False,
    ):
        super().__init__()
        self.scene_dim = int(scene_dim)
        self.history_dim = int(history_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        self.phys_dim = int(phys_dim)
        self.engine_dim = int(engine_dim)
        self.couple_physics_grad = bool(couple_physics_grad)

        self.scene_l1 = nn.Linear(self.scene_dim, hidden)
        self.scene_l2 = nn.Linear(hidden, hidden)
        self.hist_l1 = nn.Linear(self.history_dim, hidden)
        self.hist_l2 = nn.Linear(hidden, hidden)
        self.task_l1 = nn.Linear(self.task_dim, hidden // 2)

        mix_dim = hidden + hidden + hidden // 2
        self.mix_l1 = nn.Linear(mix_dim, hidden)
        self.mix_l2 = nn.Linear(hidden, hidden)
        self.mix_l3 = nn.Linear(hidden, latent)

        self.action_l1 = nn.Linear(latent, hidden)
        self.action_out = nn.Linear(hidden, self.action_dim)

        self.phys_l1 = nn.Linear(latent, hidden // 2)
        self.phys_out = nn.Linear(hidden // 2, self.phys_dim)

        self.engine_l1 = nn.Linear(latent, hidden // 2)
        self.engine_out = nn.Linear(hidden // 2, self.engine_dim)

    def encode(self, scene, history, task):
        s = nn.gelu(self.scene_l1(scene))
        s = nn.gelu(self.scene_l2(s))
        h = nn.gelu(self.hist_l1(history))
        h = nn.gelu(self.hist_l2(h))
        t = nn.gelu(self.task_l1(task))
        x = mx.concatenate([s, h, t], axis=-1)
        x = nn.gelu(self.mix_l1(x))
        x = nn.gelu(self.mix_l2(x))
        return nn.gelu(self.mix_l3(x))

    def __call__(self, scene, history, task):
        z = self.encode(scene, history, task)
        a = nn.gelu(self.action_l1(z))
        a = mx.tanh(self.action_out(a))

        # Physics labels are future generalization metadata, not the current
        # control objective. Keep their gradients out of the solving encoder by
        # default so noisy or approximate physical labels cannot change the
        # action trajectory. Set couple_physics_grad=True only for explicit
        # abstraction-layer experiments.
        z_phys = z if self.couple_physics_grad else mx.stop_gradient(z)
        p = self.phys_out(nn.gelu(self.phys_l1(z_phys)))
        e = self.engine_out(nn.gelu(self.engine_l1(z_phys)))
        return a, p, e


def loss_fn(
    model: SharedPhysicsStudent,
    scene,
    history,
    task,
    action_target,
    phys_target,
    engine_target,
    weights: LossWeights,
):
    action_pred, phys_pred, engine_pred = model(scene, history, task)
    action_loss = mse(action_pred, action_target)
    phys_loss = mse(phys_pred, phys_target)
    engine_loss = mse(engine_pred, engine_target)
    total = (
        weights.action * action_loss
        + weights.phys * phys_loss
        + weights.engine * engine_loss
    )
    return total


def eval_components(model, scene, history, task, action_target, phys_target, engine_target):
    action_pred, phys_pred, engine_pred = model(scene, history, task)
    action_loss = mse(action_pred, action_target)
    phys_loss = mse(phys_pred, phys_target)
    engine_loss = mse(engine_pred, engine_target)
    mx.eval(action_loss, phys_loss, engine_loss)
    return {
        "action_loss": float(action_loss.item()),
        "phys_loss": float(phys_loss.item()),
        "engine_loss": float(engine_loss.item()),
    }


def save_model(model: SharedPhysicsStudent, out_path: str) -> None:
    mx.savez(out_path, **dict(tree_flatten(model.parameters())))


def load_model(model: SharedPhysicsStudent, path: str) -> SharedPhysicsStudent:
    model.update(tree_unflatten(list(mx.load(path).items())))
    mx.eval(model.parameters())
    return model
