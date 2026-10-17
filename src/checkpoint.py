from __future__ import annotations

from pathlib import Path

import torch

from src.config import ModelConfig
from src.model.network import IRTPathModel


def load_checkpoint(path: str | Path, device: torch.device) -> IRTPathModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = IRTPathModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


__all__ = ["load_checkpoint"]
