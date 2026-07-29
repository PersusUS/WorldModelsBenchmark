"""
Checkpoint save/load utilities.
All checkpoints follow the format specified in RULES.md.
"""
from pathlib import Path
from typing import Optional
import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    task_id: int,
    config: dict,
    metrics: dict,
    path: Path,
) -> None:
    """Save checkpoint in the required format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "task_id": task_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": {
                "nll": float(metrics.get("nll", 0.0)),
                "kl": float(metrics.get("kl", 0.0)),
                "reconstruction": float(metrics.get("reconstruction", 0.0)),
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load checkpoint and restore model (and optionally optimizer) state."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
