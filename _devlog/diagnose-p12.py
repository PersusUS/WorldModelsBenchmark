"""
Diagnóstico de P12 / F23: por qué RD de UG-MTM estalla en minigrid/distance_max.

Por semilla, RD sale 574 / 520 / 4364 / 17.7 / 40.0 — tres órdenes de magnitud
dentro de la misma casilla, y sólo a UG-MTM (los otros cuatro métodos van de 18 a
59 en las cinco semillas).

RD es la KL media entre los rollouts imaginados de 15 pasos del modelo post-A y
el post-B. Para una gaussiana diagonal,

    KL = 0.5 * sum( log_var_k - log_var_i
                    + (var_i + (mu_i - mu_k)^2) / var_k
                    - 1 )

así que hay dos formas de que explote y se distinguen mirando los términos por
separado:

  * **divergencia de medias** — el rollout se realimenta con su propia
    predicción y |mu| crece paso a paso;
  * **colapso de varianza** — var_k se va a cero y el término cuadrático se
    dispara aunque las medias apenas difieran.

El script reproduce la celda entera (mismo protocolo, misma semilla, mismos
datos), comprueba que PF y RD coinciden con el metrics.json guardado —si no
coinciden, el diagnóstico no es sobre el objeto real— e instrumenta el rollout
paso a paso dentro de `preserve_rng_state`.

Uso:
    python _devlog/diagnose-p12.py --seeds 2 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cf_worldmodels"))
# El runner resuelve las rutas de config desde el cwd, igual que cuando lo
# lanza uno a mano desde cf_worldmodels/.
os.chdir(ROOT / "cf_worldmodels")

from omegaconf import OmegaConf  # noqa: E402

import experiments.run_full_benchmark as runner  # noqa: E402
from src.utils.seeding import preserve_rng_state  # noqa: E402

FAMILY, DISTANCE, METHOD = "minigrid", "distance_max", "ug_mtm"
RESULTS = ROOT / "cf_worldmodels" / "results"


@torch.no_grad()
def instrumented_rollout(model_i, model_k, dataset, horizon, n_samples):
    """La misma marcha que compute_rd, guardando los términos por paso."""
    device = next(model_i.parameters()).device
    model_i.eval()
    model_k.eval()
    obs = dataset["obs"].to(device)
    actions = dataset["actions"].to(device)
    n = min(n_samples, obs.shape[0])

    z_i = z_k = obs[torch.randperm(obs.shape[0])[:n]]
    h_i = torch.zeros(n, model_i.hidden_dim, device=device)
    h_k = torch.zeros(n, model_k.hidden_dim, device=device)

    rows = []
    for step in range(horizon):
        a = actions[torch.randint(0, actions.shape[0], (n,), device=device)]
        h_i = model_i.transition(h_i, z_i, a)
        h_k = model_k.transition(h_k, z_k, a)
        mu_i, log_var_i = model_i.predict_stoch(h_i)
        mu_k, log_var_k = model_k.predict_stoch(h_k)

        var_i, var_k = log_var_i.exp(), log_var_k.exp()
        term_log = 0.5 * (log_var_k - log_var_i).sum(-1)
        term_trace = 0.5 * (var_i / var_k).sum(-1)
        term_mean = 0.5 * ((mu_i - mu_k).pow(2) / var_k).sum(-1)
        kl = term_log + term_trace + term_mean - 0.5 * mu_i.shape[-1]

        rows.append({
            "step": step,
            "kl": float(kl.mean()),
            "term_log": float(term_log.mean()),
            "term_trace": float(term_trace.mean()),
            "term_mean": float(term_mean.mean()),
            "abs_mu_i": float(mu_i.abs().mean()),
            "abs_mu_k": float(mu_k.abs().mean()),
            "log_var_i": float(log_var_i.mean()),
            "log_var_k": float(log_var_k.mean()),
            "min_log_var_k": float(log_var_k.min()),
        })
        z_i, z_k = mu_i, mu_k
    return rows


def run(seed, protocol, cfg, device):
    seq_cfg = cfg.benchmark.sequences[DISTANCE]
    env_A, env_B = runner.create_env_pair(FAMILY, seq_cfg)
    try:
        buffers = runner.collect_cell_buffers(env_A, env_B, seed, protocol)
        captured = {}
        original = runner.compute_rd

        def capturing(model_i, model_k, dataset, horizon, n_samples):
            with preserve_rng_state():
                captured["rows"] = instrumented_rollout(
                    model_i, model_k, dataset, horizon, n_samples)
            return original(model_i, model_k, dataset, horizon, n_samples)

        runner.compute_rd = capturing
        try:
            metrics = runner.run_cell(
                METHOD, buffers, env_A.action_dim, device, seed, FAMILY,
                DISTANCE, runner.task_spec(seq_cfg), protocol,
                Path(args.out), reference=None)
        finally:
            runner.compute_rd = original
    finally:
        env_A.close()
        env_B.close()
    return metrics, captured["rows"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--out", default=str(ROOT / "_devlog" / "p12-scratch"))
    args = parser.parse_args()

    cfg = OmegaConf.load(runner.FAMILY_CONFIGS[FAMILY])
    protocol = runner.resolve_protocol(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        stored = json.loads(
            (RESULTS / METHOD / f"{FAMILY}_{DISTANCE}_{seed}" /
             "metrics.json").read_text())
        metrics, rows = run(seed, protocol, cfg, device)

        print(f"\n{'=' * 78}\nsemilla {seed}\n{'=' * 78}")
        print(f"  RD  guardado {stored['rd']:12.4f}   reproducido "
              f"{metrics['rd']:12.4f}")
        print(f"  PF  guardado {stored['pf']:12.4f}   reproducido "
              f"{metrics['pf']:12.4f}")
        print(f"\n  {'paso':>4} {'KL':>12} {'termino log':>12} "
              f"{'termino tr':>12} {'termino med':>12} {'|mu_i|':>9} "
              f"{'|mu_k|':>9} {'logvar_i':>9} {'logvar_k':>9} {'min lv_k':>9}")
        for r in rows:
            print(f"  {r['step']:>4} {r['kl']:>12.3f} {r['term_log']:>12.3f} "
                  f"{r['term_trace']:>12.3f} {r['term_mean']:>12.3f} "
                  f"{r['abs_mu_i']:>9.3f} {r['abs_mu_k']:>9.3f} "
                  f"{r['log_var_i']:>9.3f} {r['log_var_k']:>9.3f} "
                  f"{r['min_log_var_k']:>9.3f}")
