"""
Logging utilities for experiment tracking.
"""
from typing import Optional
import json
from pathlib import Path


def init_wandb(cfg, method: str, family: str, distance: str, seed: int,
               no_wandb: bool = False):
    """Initialize wandb run or return None if disabled."""
    if no_wandb:
        return None
    import wandb
    from omegaconf import OmegaConf
    run = wandb.init(
        project="cf_worldmodels",
        group=method,
        name=f"{method}_{family}_{distance}_{seed}",
        tags=[method, family, distance],
        config=OmegaConf.to_container(cfg),
    )
    return run


def log_metrics(step: int, metrics: dict, wandb_run=None, prefix: str = ""):
    """Log metrics to wandb and/or console."""
    tagged = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
    if wandb_run is not None:
        wandb_run.log(tagged, step=step)


def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
