from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        # torch.cuda.amp.autocast is deprecated in newer torch; prefer torch.amp.autocast.
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        except Exception:
            return torch.cuda.amp.autocast()
    if device.type == "mps":
        try:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        except Exception:
            return nullcontext()
    # CPU autocast is available in newer torch versions; fall back gracefully.
    try:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        return nullcontext()


def get_windowing_array(window_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def demix_track(
    *,
    config,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    progress: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    """
    mix: torch.Tensor shaped (channels, samples)
    returns: dict stem -> np.ndarray shaped (channels, samples)
    """
    C = int(config.inference.chunk_size)
    N = int(config.inference.num_overlap)
    step = C // N
    fade_size = C // 10
    border = C - step

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    windowing_array = get_windowing_array(C, fade_size, device)

    with _autocast(device, enabled=use_amp):
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1,) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = max(1, (total_length + step - 1) // step)

            while i < total_length:
                part = mix[:, i : i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(part, (0, C - length), mode="reflect")
                    else:
                        part = nn.functional.pad(part, (0, C - length, 0, 0), mode="constant", value=0)

                x = model(part.unsqueeze(0))[0]

                window = windowing_array
                if i == 0:
                    window = window.clone()
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window = window.clone()
                    window[-fade_size:] = 1

                result[..., i : i + length] += x[..., :length] * window[..., :length]
                counter[..., i : i + length] += window[..., :length]
                i += step

                if progress is not None:
                    progress(min(1.0, (i / step) / num_chunks))

            estimated_sources = (result / counter).cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if mix.shape[1] > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        instruments = list(config.training.instruments)
    else:
        instruments = [str(config.training.target_instrument)]

    return {k: v for k, v in zip(instruments, estimated_sources)}
