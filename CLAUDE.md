# Working in this repository

Context for an AI agent arriving cold. Humans should read `README.md` instead;
this file is the map, the house rules, and the traps.

## What this is

A benchmark measuring catastrophic forgetting in the **transition component**
`M` of a world model — not in the policy, not in the system as a whole — and a
paper reporting two negative results it produced. The benchmark is the
instrument; the findings are the contribution.

The work is **finished**. 375 runs are committed, the paper is written, verified
number by number, and compiled. What remains is submitting it to a workshop
(deadline 29 Aug 2026). Do not start new experiments unless asked.

## Where things are

| Path | What it is |
| --- | --- |
| `cf_worldmodels/src/` | Models, baselines, metrics, environment wrappers |
| `cf_worldmodels/experiments/` | Runners, and the two scripts that produce every published number |
| `cf_worldmodels/configs/benchmark/` | One YAML per family; the `protocol:` block is the single source of truth |
| `cf_worldmodels/results/` | 375 `metrics.json` + 75 reference pairs. Committed |
| `cf_worldmodels/results-2x/`, `results-seq/` | Doubled-budget probe, four-task sequence |
| `paper/main.tex` | The full paper, 26 pages, compiled to `paper/WMF.pdf` |
| `paper/main_workshop.tex` | The 8-page version. **Do not move it into a subdirectory** — see traps |
| `paper/tables/` | **Generated.** Never edit by hand |
| `_devlog/` | Development log: findings, decisions, run history. Tracked, but not part of the published artifact |

**`_devlog/HANDOFF.md` is the entry point for picking the project up.** It is
self-contained and opens with a state block. `decisions.md` (D1–D23) says why
things are the way they are, `findings.md` (F0–F29) is every defect found with
its evidence, `runs.md` (R0–R20) is what was executed.

## House rules

Not style preferences. Each one exists because breaking it cost something.

1. **No number is ever transcribed by hand.** Tables come from
   `experiments/export_tables.py`, console summaries from
   `summarize_results.py`. The second imports the first, so the paper and the
   console can only disagree if the code disagrees with itself. The previous
   version of this paper drifted from its own runs, which is why
   `_devlog/paper-vs-code.md` exists.

2. **Results are never quietly fixed** (D4). When a defect invalidates numbers,
   the code is corrected *and* the impact is documented. The paper has a section
   saying none of the earlier version's five findings survived, and why.

3. **The protocol lives in the config, not in the code.** `resolve_protocol()`
   reads, casts and validates it; a missing field is an error, not a default.
   Every `metrics.json` stores the block it ran under, and the runner refuses to
   combine results produced under different ones.

4. **Measuring must not change the result.** All instrumentation goes inside
   `seeding.preserve_rng_state()`. Not paranoia: UG-MTM's gate keeps MC-dropout
   active at evaluation *by design*, so every `compute_nll` consumes the random
   stream.

5. **Cells are summarised by median and range, not mean ± sd** (D15). RD is an
   unbounded KL, and a method that ends up overconfident gets a heavy right tail
   by construction — one cell runs 17.7 to 8135 across ten seeds.

6. **`scipy` is installed but not declared.** Do not import it in repository
   code without adding it to `requirements.txt`. Spearman is written out by hand
   in `summarize_results.py` for exactly this reason.

## Commands

Everything runs from `cf_worldmodels/`. `wandb` is broken here; runners take
`--no_wandb`.

```bash
python -m pytest -m "not integration and not slow"
```

```bash
python experiments/summarize_results.py
```

```bash
python experiments/export_tables.py
```

Structural check of the paper, since there is no LaTeX toolchain on this
machine. It catches unbalanced braces, mismatched environments, an `\input`
without a file, a `\ref` without a `\label`, a `\cite` without a bib entry:

```bash
python _devlog/check-paper.py
```

Classify every number in the paper's prose by whether a generated table backs
it. It **classifies, it does not verify**: it says where to look, not whether
the figure is right. Four wrong numbers were found by hand inside its own
"backed" list.

```bash
python _devlog/check-numbers.py
```

## Traps, each one paid for once

- **Every environment family needs its own process.** dm_control cannot get an
  OpenGL context in a process where MuJoCo already has one. The runner forks per
  family; a 30-hour run died at that boundary before it did.
- **Do not run the benchmark and the integration tests at once.** Render
  contexts serialise and both crawl.
- **Any number in `runs.md` before R13 is from a different budget** (1000
  gradient steps, not 5000), including the task-A convergence figures that
  circulated as "5.3e-04 per pixel".
- **Overleaf compiles from the project root**, not from the directory of the
  main document. `paper/main_workshop.tex` sits beside `refs.bib`, `tables/` and
  `figures/` for exactly this reason, after three build failures caused by
  putting it in a subdirectory. Do not tidy it away again.
- **`bibtex` does not expand LaTeX macros** and refuses paths that climb out of
  the compile directory, so a `\bibliography` name cannot go through a prefix
  macro the way `\input` can.
- **Page-count estimates from word density run low** by roughly a page on a
  nine-section document: they miss the vertical space taken by section headings,
  `\paragraph`, paragraph breaks and float placement. Compile, do not estimate.

## What the results actually say

Worth having straight, because several readings are easy to overstate.

- **The labelled distance axis carries no rank information** (Spearman 0.00
  against forgetting), which peaks at the *medium* level in all three families.
  This is the robust finding: it survives changes of aggregation, metric,
  threshold, training budget and seed count.
- **The two measured predictors are not ranked against each other.** They swap
  places when seeds are added, which is itself the evidence that nine cells do
  not resolve them. Do not turn this back into a ranking.
- **`d_trans` does not order levels within a family** and moves with the
  training budget. The paper says so about its own instrument.
- **Three of the nine cells are declared controls** — nothing forgets there — so
  the effective grid is six cells.
- **Replay's advantage is not universal**: lower RD in five of the six
  forgetting cells, worse in one, indistinguishable in another.
- **That forgetting tracks task-B demand rather than distance is the best
  available account, not a demonstrated mechanism.** DMControl breaks it. Keep
  the hedge.
- **UG-MTM is the author's method and it does not win.** It is characterised:
  it does not forget because it does not learn, and freezing its encoder makes
  its transition models overconfident. Do not write it up as a success.

## Conventions

- The development log is written in Spanish; code, code comments, commit
  messages and the paper are in English.
- Comments explain a non-obvious *why*, not what the line does.
- Commit messages are prose describing what changed and what it cost, without
  bullet lists of files.
