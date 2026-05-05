# Meadow WM

> Physics-grounded three-layer learning stack for robot control.

🌐 **Live demo (Cloudflare Pages)**: https://meadow-wm.pages.dev
📄 **Full report (English)**: [`web/meadow_ami.pdf`](web/meadow_ami.pdf) · [`web/index.html`](web/index.html)
📄 **完整報告 (繁體中文)**: [`web/meadow_ami_zh.pdf`](web/meadow_ami_zh.pdf) · [`web/index_zh.html`](web/index_zh.html)

---

## Key results (Meadow v0.0.2)

| Task | Causal / Scorer | Success | Precision | Wall-clock |
|---|---|---|---|---|
| **TwoRoom** (2D nav) | val_acc 0.9990 | 80/80 (100%) | binary terminal | ~48 s |
| **PushT** (contact) | 4/4 pose-aware guard | 2/2 (100%) | **0.81 px** sub-pixel | ~56 s + ~112 s distill |
| **Reacher** (arm) | 4/4; median 0.00369 | 2/2 (100%) | mean 0.00494 | ~minutes |
| **OGBench Cube** (pick-and-place) | val_acc 0.9963 | 4/4 (100%) | phase-conditioned | ~25 s |

**OGBench v1 single-pipeline pass** — same expansion–scorer–reaction code, only spec changes:

| Family | Causal | Reaction | Samples |
|---|---|---|---|
| Cube | 8/8 | 8/8 | 894 |
| PointMaze | 8/8 | 8/8 | 2,724 |
| Scene | 8/8 | 8/8 | 903 |
| Puzzle | 8/8 | 8/8 | 443 |

Single device: **Apple M1 Max + MLX** (no cloud, no NVIDIA). Reaction inference: **0.095–0.122 ms** (Core ML).

---

## Architecture

Meadow factors robot control into **three explicit layers**:

1. **Physics expansion (Layer 1)** — causal tree of candidate trajectories under simulator dynamics. *Non-learning.* Branches are filtered through physical constraint before any neural component sees them.
2. **Neural scorer (Layer 2)** — compact network that learns success likelihood over physics-validated branches. The training distribution has already passed physical filtering, so the abstractions formed respect conservation and feasibility without auxiliary losses.
3. **Reaction policy (Layer 3)** — distillation of the selected chain into a deployable low-latency controller (sub-millisecond inference on Apple Silicon via Core ML).

A **named-variable calibration loop** closes the system: when deployed behavior diverges from the predicted chain, specific physical parameters in the spec are re-estimated and the affected layer is re-distilled locally — no full retraining, no demonstration re-collection.

---

## Reproducibility

All numbers reported in the [full report](web/meadow_ami.pdf) come from local artifacts on a single Apple M1 Max workstation. Per-checkpoint scorer convergence is captured in local `metrics.json` files; videos in [`web/`](web/) subdirectories are direct rollouts from the trained checkpoints (not curated highlights).

**Code release**: source code is available on request — please email `akai@fawstudio.com` with a brief description of intended use. The 32 rollout videos in [`web/`](web/) provide visual reproduction of the four showcase tasks; the report PDF contains exact wall-clock, sample counts, and evaluation protocol.

---

## Repository structure

```
README.md                 # This file
web/                      # Cloudflare Pages deployment source
  index.html              # Full English report
  index_zh.html           # 完整繁體中文報告
  meadow_ami.pdf          # Downloadable English PDF (1.5 MB)
  meadow_ami_zh.pdf       # 中文 PDF (2.1 MB)
  tworoom_*/              # TwoRoom 2D navigation rollout videos
  pusht_*/                # PushT contact manipulation rollout videos
  reacher_*/              # Reacher arm control rollout videos
  ogbench_cube_*/         # OGBench Cube pick-and-place rollout videos
```

---

## Citation

```bibtex
@techreport{huang2026meadow,
  title  = {Meadow: A Physics-Grounded Three-Layer Learning Stack for Robot Control},
  author = {Huang, Sheng-Kai},
  year   = {2026},
  type   = {Technical report},
  institution = {Independent research, Hey-Meadow Lab},
  url    = {https://github.com/Hey-Meadow/meadow-wm}
}
```

---

## License

Report and rollout videos: **CC BY-NC 4.0** (attribution + non-commercial).
Source code license: to be determined upon release — contact `akai@fawstudio.com`.

---

## Contact

Sheng-Kai Huang (黃聖凱) · Independent researcher, Taipei
- Email: akai@fawstudio.com
- GitHub: [@akaiHuang](https://github.com/akaiHuang)
- Lab: [@Hey-Meadow](https://github.com/Hey-Meadow) — *Independent research lab. World models grounded in physical state, inspired by* A Path Towards Autonomous Machine Intelligence *(LeCun, 2022).*
