# A Benchmark for Catastrophic Forgetting in World Models

Code and results for the WMF (World Model Forgetting) benchmark: a protocol for
measuring how much a learned world model forgets a previous environment after
being trained on a new one, across three environment families and three levels
of dynamic distance.

The benchmark evaluates five continual-learning methods:

| Method | Description |
| --- | --- |
| `finetuning` | Sequential training, no protection (lower bound) |
| `replay_infinite` | Retains all data from all tasks (upper bound) |
| `ewc` | Elastic Weight Consolidation ([Kirkpatrick et al., 2017](https://arxiv.org/abs/1612.00796)) |
| `progressive_nets` | Progressive Networks ([Rusu et al., 2016](https://arxiv.org/abs/1606.04671)) |
| `ug_mtm` | UG-MTM: uncertainty-gated mixture of transition models (ours) |

across three families — MiniGrid (discrete), Gymnasium/MuJoCo (continuous,
variable physics), and DMControl (visual) — at three dynamic distances each,
with 5 seeds per cell (225 runs total).

## Repository layout

```
cf_worldmodels/
├── configs/
│   ├── benchmark/          # One YAML per environment family, 3 distance levels each
│   └── models/             # RSSM baseline and UG-MTM hyperparameters
├── src/
│   ├── envs/               # Wrappers normalizing every family to (64,64,3) float32 [0,1]
│   ├── models/             # ConvVAE, RSSM baseline, UG-MTM
│   ├── baselines/          # Fine-tuning, infinite replay, EWC, Progressive Nets
│   ├── benchmark/          # PF / RD / WMF / FT metrics, d_param / d_trans distances
│   └── utils/              # Replay buffer, checkpointing, logging
├── experiments/            # Benchmark runners and plotting scripts
├── tests/                  # Test suite (221 tests)
└── results/                # One directory per run (metrics.json + checkpoint)
```

## Installation

```bash
conda env create -f cf_worldmodels/environment.yml
```

```bash
conda activate cf_worldmodels
```

Or with pip into an existing Python 3.11 environment:

```bash
pip install -r cf_worldmodels/requirements.txt
```

MuJoCo and dm_control require a working OpenGL/EGL setup for offscreen
rendering. All commands below are run from the `cf_worldmodels/` directory.

## Reproducing the results

`run_full_benchmark.py` reproduces every number in the paper. It skips any run
whose `metrics.json` already exists, so it is safe to interrupt and resume:

```bash
python experiments/run_full_benchmark.py
```

Regenerate the paper figure and the results table from the stored metrics
(no training required):

```bash
python experiments/plot_final.py
```

Run a single method/family/distance combination:

```bash
python experiments/run_benchmark.py --method ug_mtm --config configs/benchmark/minigrid.yaml --distance distance_min --seeds 0 1 2 3 4 --no_wandb
```

## Results

> **Not yet published.** An earlier run of this benchmark was invalidated by a
> collapsed VAE posterior: 0 of 32 latent dimensions were active, so the
> transition model received a constant latent regardless of the observation and
> every metric was noise around a degenerate model. That has been fixed, and
> results are being regenerated. This section will be filled from the
> regenerated `metrics.json` files.

Running `experiments/run_full_benchmark.py` writes one directory per run to
`results/<method>/<family>_<distance>_<seed>/`, containing `metrics.json` and
`checkpoint_final.pt`. Checkpoints (~32 MB each) are not tracked in git — see
`.gitignore` — but every table and figure is derived from the `metrics.json`
files, which are.

## Metrics

- **PF** (Prediction Fidelity) — `NLL(M_k, D_i) - NLL(M_i, D_i)`. Positive means
  the model got worse at task *i* after training on later tasks.
- **RD** (Rollout Divergence) — mean KL between imagined rollouts of the model
  before and after the task switch.
- **PIS** (Policy Impact Score) — reserved in the metrics schema; reported as
  `0.0` in this release (see Known limitations).
- **WMF** = `alpha*PF + beta*RD + gamma*PIS`, with `alpha=beta=0.4, gamma=0.2`.
- **FT** (Forward Transfer) — `NLL(random init) - NLL(pretrained)`. Positive
  means prior knowledge helped.

All of these are evaluated on `D_i`: held-out task-A rollouts, collected
separately from the training buffer and encoded once by the post-task-A model
(`protocol.build_latent_eval_dataset`), so that both models being compared are
scored on identical inputs and identical targets.

Dynamic distance between two tasks is measured by `d_param` (normalized L2
distance between physics parameter vectors, Gymnasium family only) and
`d_trans` (expected KL between the two trained transition models, all families).

## Tests

```bash
python -m pytest
```

221 tests covering the models, baselines, metrics, distances, buffer,
checkpoint format, and config consistency. The 20 tests marked `integration`
build real MiniGrid / MuJoCo / dm_control environments; skip them with:

```bash
python -m pytest -m "not integration"
```

## Known limitations

These are properties of the released code that reviewers and reusers should be
aware of before building on it.

1. **Uncertainty routing works at large dynamic distance and inverts at
   moderate distance.** Measured as `AUC = P(u_B > u_A)` on held-out
   transitions after training on task A (0.5 = no discrimination): the
   MC-dropout signal reaches `0.864` on FourRooms→KeyCorridor but falls to
   `0.294` on Empty-8x8→FourRooms, i.e. task B looks *less* uncertain than
   task A and the gate routes the wrong way. UG-MTM's premise holds only in
   the first regime.

2. **Training scale does not match the configs.** The benchmark configs declare
   `n_collect: 1000`, `n_train: 50000`, `batch_size: 32`, `seq_len: 50`, but
   `run_full_benchmark.py` hardcodes `N_COLLECT=20`, `STEPS=1000`,
   `BATCH_SIZE=8`, `SEQ_LEN=5`, and overrides `mc_dropout_T` from 10 to 3. The
   published numbers come from the hardcoded values.

3. **The DMControl `distance_min` pair is not actually two different tasks.**
   Both `task_A` and `task_B` are `cheetah/run`; the `lateral_wind: true`
   parameter that was meant to differentiate them is never read by
   `DMControlEnv`, which only takes `domain_name` and `task_name`.

4. **No ablation study ships with this release.** The previous
   `run_ablations.py` built an overridden config and then never passed it to
   the training routine, which reloaded the unmodified YAML from disk — so all
   five ablations silently ran plain UG-MTM. It has been removed rather than
   left in place producing misleading output.

5. **Gate scaling uses only the final timestep's gates.** `UG_MTM.transition`
   clears and re-registers its backward hooks on every call, so after unrolling
   a sequence the gradients for the whole sequence are scaled by the gates
   computed at the last step.

## Citation

```bibtex
@misc{perezbazarot2026wmf,
  title  = {A Benchmark for Catastrophic Forgetting in World Models},
  author = {P{\'e}rez Bazarot, Jes{\'u}s},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
