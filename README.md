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
├── tests/                  # Test suite (291 tests)
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

The training protocol is not defined in the runner. Every value comes from the
`protocol:` block of `configs/benchmark/<family>.yaml`, is printed before
training starts, and is recorded in each `metrics.json` — so the protocol table
in the paper is generated from the results rather than written by hand. Runs
cached under a different protocol are refused rather than silently averaged into
the same cell.

Before the five methods of a cell, the runner trains the pair of reference models
that forward transfer and `d_trans` are defined against: one plain RSSM per
environment, from scratch, per (family, distance, seed). They are cached in
`results/_reference/` and shared by every method, since neither quantity depends
on the continual-learning method. To skip them — `ft` and `d_trans` are then
stored as `null`, never as 0:

```bash
python experiments/run_full_benchmark.py --skip-reference
```

Print the effective protocol and the run plan without training anything:

```bash
python experiments/run_full_benchmark.py --dry-run
```

Restrict the grid, or override a protocol field explicitly:

```bash
python experiments/run_full_benchmark.py --families minigrid --methods ewc --seeds 0 1 --steps 2000
```

Measure how far task A is actually learned, and how that changes with the
training budget (one run at the largest budget, evaluated at each multiple of
`n_train` along the way):

```bash
python experiments/convergence_A.py --family minigrid --multipliers 1 2 5 10
```

Regenerate the main figure from the stored metrics (no training required): one
row per reported metric, one column per family, with the measured `d_trans` on
the X axis wherever the runs carry it:

```bash
python experiments/plot_final.py
```

Aggregate every cell, with the task-A quality columns alongside the forgetting
metrics, and optionally a seed-paired comparison of two methods:

```bash
python experiments/summarize_results.py --compare replay_infinite finetuning
```

It refuses to average runs that were produced under different protocols, and it
reports an exact paired permutation p rather than a t-test: with 5 seeds the
smallest two-sided p an exact test can return is 2/2^5 = 0.0625, so a parametric
p-value in that regime describes the normality assumption more than the data.

Run a single method/family/distance combination:

```bash
python experiments/run_benchmark.py --method ug_mtm --config configs/benchmark/minigrid.yaml --distance distance_min --seeds 0 1 2 3 4 --no_wandb
```

### Determinism

Re-running a cell with the same seed reproduces its metrics bit-for-bit. Getting
there needs more than seeding `torch` and `numpy`, because two independent
sources of nondeterminism sit outside them:

- **The environments own RNGs no global seed reaches.** Gymnasium environments
  and action spaces each carry their own generator, seeded from OS entropy, and
  dm_control randomizes the initial state through the task's own `random`
  argument. Left unseeded, two runs of the same seed collected different rollouts
  and therefore trained on different data. `BaseEnv.seed()` seeds both the
  episode RNG and the action sampler; every runner calls it.
- **cuDNN picks kernels by heuristic** and its GRU backward is nondeterministic
  by default. `src/utils/seeding.py::set_seed()` sets
  `cudnn.deterministic = True` and `cudnn.benchmark = False`, which costs some
  throughput and was verified sufficient to make training bit-identical here.

Both are covered by `tests/test_seeding.py`, including an end-to-end check that
training twice from the same seed yields identical weights.

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

The suite is **PF, RD and FT**. An earlier description of this benchmark
announced a fourth, PIS (Policy Impact Score); it is not part of the suite and
never was implemented — see Known limitations.

- **PF** (Prediction Fidelity) — `NLL(M_k, D_i) - NLL(M_i, D_i)`. Positive means
  the model got worse at task *i* after training on later tasks.
- **RD** (Rollout Divergence) — mean KL between imagined rollouts of the model
  before and after the task switch.
- **WMF** = `alpha*PF + beta*RD + gamma*PIS`, with `alpha=beta=0.4, gamma=0.2`.
  Computed and stored, but **not the headline number**: RD supplies 78-97% of
  it, so `summarize_results.py` reports PF and RD side by side and prints WMF
  under a heading that says what it is, next to the share of it that comes from
  RD. It is there to reproduce the previous paper's number, and its `gamma`
  term is evaluated at zero — which is what that number was computed with.
- **FT** (Forward Transfer) — `recon_B(trained from scratch) -
  recon_B(pretrained on A)`, on held-out task-B frames in pixel space. Positive
  means knowledge of task A helped learn task B under the same budget and the
  same data. The from-scratch arm comes from a reference model trained on task B
  alone, one per (family, distance, seed), shared by all five methods.
- **`task_A_fit_gain`** — `NLL(random init, D_i) - NLL(post-task-i model, D_i)`.
  This is what earlier releases called FT; it measures how well task *i* was
  learned, and no task-B data enters it.

All of these are evaluated on `D_i`: held-out task-A rollouts, collected
separately from the training buffer and encoded once by the post-task-A model
(`protocol.build_latent_eval_dataset`), so that both models being compared are
scored on identical inputs and identical targets. That also means they are blind
to drift in the encoder itself — see Known limitations.

Alongside them, each run records how well task A was learned in the first place,
because a forgetting benchmark has to show there was something to forget:

- `heldout_reconstruction_A_after_task_A` / `..._after_task_B` — squared
  reconstruction error per frame, in **pixel** space, on held-out task-A frames.
  This is the only quality signal that is comparable across training budgets: the
  latent NLL is scored against latents the model itself produces, and that target
  moves as the encoder trains.
- `heldout_reconstruction_B_after_task_B` and
  `heldout_reconstruction_B_from_scratch` — the two arms of FT.
- `nll_A_after_task_A`, `nll_A_after_task_B`, `nll_A_random_init` — the three
  NLLs that PF and `task_A_fit_gain` are built from, so both stay decomposable.
- `initial/final_reconstruction_loss_A` and `_B`, plus a 20-point curve for each
  task, and `n_nan_steps_A/B` for steps dropped as non-finite.

Dynamic distance between two tasks is measured by `d_param` (normalized L2
distance between physics parameter vectors, for the variable-physics pairs) and
`d_trans` (Eq. 9: expected KL between one transition model per environment, each
trained on its own task from scratch, all families). `d_trans` is a property of
the task pair and the seed rather than of the method, so it is computed once per
(family, distance, seed) alongside the forward-transfer reference and stored in
`results/_reference/`.

## Tests

```bash
python -m pytest
```

345 tests covering the models, baselines, metrics, distances, buffer,
checkpoint format, seeding, protocol resolution and config consistency. The 25 tests marked
`integration` build real MiniGrid / MuJoCo / dm_control environments; skip them
with:

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

2. **PF and RD are blind to forgetting in the encoder.** They are evaluated on
   latents that were encoded once, by the post-task-A model, and `compute_nll`
   never calls `encode` — so they measure drift in the GRU and the stochastic
   head within a *frozen* latent basis. Measured on MiniGrid `distance_med`
   (seed 999): fine-tuning's held-out task-A reconstruction degrades by a factor
   of **112** (6.49 → 725.27 squared error per frame) while its PF comes out
   **negative** (−1.78). This is deliberate — the benchmark's scope is the
   transition component — but it means WMF is not a measure of how much the
   world model as a whole forgot. Every run records
   `heldout_reconstruction_A_after_task_{A,B}` so both can be read side by side.

3. **The training scale is small.** 20 episodes of a random policy per task and
   5000 gradient updates at batch 8, sequence length 5; UG-MTM's training-time
   MC-dropout budget is 3 passes. Every one of those values is declared in the
   `protocol:` block of the family config, recorded in each `metrics.json`, and
   printed before training. Task A does get learned at this scale — held-out
   reconstruction reaches 5.3e-04 per pixel, RMSE ≈ 0.023 on `[0,1]` — and
   `experiments/convergence_A.py` measures how that changes with the budget.

4. **Forward transfer is measured in pixels, and replay's number needs a
   caveat.** The two arms of FT are models with unrelated latent bases, so a
   latent NLL would score one of them in the other's coordinates; held-out pixel
   reconstruction is the one scale they share. Separately, `replay_infinite`
   trains on A+B during the task-B phase, so half of its gradient steps go to
   task-A data: its FT mixes transfer with a halved effective budget on B.

   Earlier releases reported a quantity named FT that was computed from the
   post-task-i model on task-i data alone. No task-B data entered it, so methods
   sharing an architecture got **identical** values by construction — fine-tuning
   and infinite replay differed by exactly 0.000 across 5 seeds and two distance
   levels. That quantity is still stored, under the name `task_A_fit_gain`.

5. **EWC protects the transition component and nothing else.** Its Fisher is
   defined over `log P(z'|z, a)`, so it is exactly zero on every encoder
   parameter: the penalty cannot constrain the VAE, and EWC's pixel-space
   reconstruction of task A degrades like fine-tuning's. It is also zero on
   `gru.weight_hh`, because the Fisher set is single transitions started from
   `h = 0` — the recurrent pathway is unprotected, and `compute_nll` scores from
   `h = 0` too, so PF does not see it either. RD, which rolls out 15 steps, does.

6. **No ablation study ships with this release.** The previous
   `run_ablations.py` built an overridden config and then never passed it to
   the training routine, which reloaded the unmodified YAML from disk — so all
   five ablations silently ran plain UG-MTM. It has been removed rather than
   left in place producing misleading output.

7. **Gate scaling uses only the final timestep's gates.** `UG_MTM.transition`
   clears and re-registers its backward hooks on every call, so after unrolling
   a sequence the gradients for the whole sequence are scaled by the gates
   computed at the last step.

8. **PIS was announced and is not part of the suite.** An earlier description of
   this benchmark listed a fourth metric, PIS (Policy Impact Score), meant to
   score how much a task switch costs a policy. It was never implemented:
   measuring it means training a controller inside the model's imagination and
   evaluating it in the real environment, and no controller ships here. It is
   withdrawn rather than reported — the suite is PF, RD and FT — and `pis` is
   stored as `null`, the same way `ft` and `d_trans` are when their reference
   model was skipped. Runs produced before this change stored `0.0`; the
   aggregation treats null and 0.0 alike here, because the `gamma` term of WMF
   was evaluated at zero either way, which is also what the previous paper's
   WMF numbers were computed with.

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
