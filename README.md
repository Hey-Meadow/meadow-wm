# Meadow WM

Physics-grounded three-layer learning stack for robot control.

🌐 **Live demo (Cloudflare Pages)**: https://meadow-wm.pages.dev

## Report

- English: [`web/index.html`](web/index.html)
- 繁體中文: [`web/index_zh.html`](web/index_zh.html)

## Key results (Meadow v0.0.2)

| Task | Causal / Scorer | Success | Precision | Wall-clock |
|---|---|---|---|---|
| TwoRoom (2D nav) | val_acc 0.9990 | 80/80 (100%) | binary terminal | ~48 s |
| PushT (contact) | 4/4 pose-aware guard | 2/2 (100%) | **0.81 px** sub-pixel | ~56 s + ~112 s distill |
| Reacher (arm) | 4/4; median 0.00369 | 2/2 (100%) | mean 0.00494 | ~minutes |
| OGBench Cube (pick-and-place) | val_acc 0.9963 | 4/4 (100%) | phase-conditioned | ~25 s |

Single device: **Apple M1 Max + MLX** (no cloud, no NVIDIA). Reaction inference: 0.095–0.122 ms (Core ML).

## Architecture

Three explicit layers:

1. **Physics expansion** — causal tree of candidate trajectories under simulator dynamics (no learning)
2. **Neural scorer** — success likelihood over physics-validated branches
3. **Reaction policy** — distilled into a low-latency deployable controller

A **named-variable calibration loop** closes the system: deployment drift triggers local re-distillation, no full retraining.

## OGBench v1 single-pipeline pass

Same expansion–scorer–reaction code, only spec changes:

| Family | Causal | Reaction | Samples |
|---|---|---|---|
| Cube | 8/8 | 8/8 | 894 |
| PointMaze | 8/8 | 8/8 | 2,724 |
| Scene | 8/8 | 8/8 | 903 |
| Puzzle | 8/8 | 8/8 | 443 |

## Repository structure

```
/web/                    Cloudflare Pages deployment source
  index.html             English report
  index_zh.html          繁體中文 report
  tworoom_*/             TwoRoom 2D navigation videos
  pusht_*/               PushT contact manipulation videos
  reacher_*/             Reacher arm control videos
  ogbench_cube_*/        OGBench Cube pick-and-place videos
```

## License

To be determined.
