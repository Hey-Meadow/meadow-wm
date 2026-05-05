# Meadow WM — Code

Reproducer scripts for the four showcase tasks in [Meadow v0.0.2](../web/meadow_ami.pdf).

## Setup

```bash
# Apple Silicon (M1/M2/M3/M4) recommended; MLX-first codebase
pip install -r requirements.txt
```

External dependency `stable-worldmodel` ([galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), Maes / Le Lidec / Balestriero) provides shared environment registry / wrappers. Same package used by [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm).

## File map

| File | Role |
|---|---|
| `meadow_student_backbone.py` | Shared MLP backbone for all reaction-policy students (Layer 3) |
| `rtg_translator.py` | Return-to-go state augmentation (used by PushT student) |
| `ik_reacher.py` | Closed-form IK + branch-picking helper (used by Reacher) |
| `train_tworoom_neural_causal_scorer.py` | TwoRoom **Layer 2** (neural scorer) training |
| `train_tworoom_causal_student.py` | TwoRoom **Layer 3** (BC student v1) — referenced by v2 |
| `train_tworoom_causal_student_v2.py` | TwoRoom **Layer 3** (BC student v2, used in report) |
| `train_tworoom_reflex_brain.py` | TwoRoom physics + sampling helpers |
| `train_pusht_causal_student_v2.py` | PushT **Layer 3** (kNN-exemplar reaction, achieves 0.81 px sub-pixel) |
| `train_reacher_causal_student_v2.py` | Reacher **Layer 3** (Reaction v2) |
| `train_cube_neural_causal_scorer.py` | Cube intermediate scorer (Layer 2) |
| `train_ogbench_cube_neural_causal_scorer.py` | OGBench Cube **Layer 2** (full scorer training) |
| `train_ogbench_cube_causal_student.py` | OGBench Cube **Layer 3** (Reaction v2) |

## Quick start

Each script is self-documenting via `--help`. Examples:

```bash
# 1. TwoRoom — train scorer (Layer 2) then student (Layer 3)
python train_tworoom_neural_causal_scorer.py --output runs/tworoom_scorer
python train_tworoom_causal_student_v2.py --scorer runs/tworoom_scorer

# 2. PushT — distill kNN exemplar reaction (achieves 0.81 px sub-pixel)
python train_pusht_causal_student_v2.py --output runs/pusht_v2

# 3. Reacher — train reaction student
python train_reacher_causal_student_v2.py --output runs/reacher_v2

# 4. OGBench Cube — full pipeline
python train_ogbench_cube_neural_causal_scorer.py --output runs/cube_scorer
python train_ogbench_cube_causal_student.py --scorer runs/cube_scorer --output runs/cube_v2
```

Wall-clock on Apple M1 Max: ~25–112 s per training, depending on task. Outputs include `metrics.json` (training curves), `summary.json`, `*.npz` (checkpoints), and `videos/` (rollouts).

See [`../web/meadow_ami.pdf`](../web/meadow_ami.pdf) for full reproduction protocol, sample counts, and evaluation metrics.

## Layer 1 (causal tree expansion) note

The non-learning physics expansion layer is implemented per task — see the report's §2.1 for the formal description. Per-task expansion uses MuJoCo / OGBench / 2D simulation as the backend; spec-driven branching logic is short and inline within the corresponding train script.

## Status

This is the v0.0.2 snapshot — code is research-grade, not packaged. For a packaged release with unified entry point, please contact `akai@fawstudio.com`.

## License

[MIT](../LICENSE) (same as repo root).

## Acknowledgments

`stable-worldmodel` infrastructure by Lucas Maes, Quentin Le Lidec, Randall Balestriero (used as shared environment layer).
