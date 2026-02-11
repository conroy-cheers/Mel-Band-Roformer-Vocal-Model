from __future__ import annotations

from typing import Any

import torch


def get_model_from_config(model_type: str, config) -> torch.nn.Module:
    if model_type == "mel_band_roformer":
        from .models.mel_band_roformer import MelBandRoformer

        return MelBandRoformer(**dict(config.model))
    raise ValueError(f"Unknown model_type: {model_type!r}")


def load_state_dict(model: torch.nn.Module, checkpoint_path: str) -> None:
    obj: Any = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
        # lightning often prefixes with "model."
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return
    if isinstance(obj, dict):
        model.load_state_dict(obj)
        return
    raise ValueError("Unsupported checkpoint format; expected a dict or dict with 'state_dict'")

