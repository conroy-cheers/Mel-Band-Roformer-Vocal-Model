from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from .config import load_config
from .demix import demix_track
from .modeling import get_model_from_config, load_state_dict


@dataclass(frozen=True)
class SeparationResult:
    vocals: np.ndarray | None
    instrumental: np.ndarray | None
    stems: dict[str, np.ndarray]
    sample_rate: int


def _to_samples_channels(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Returns (samples, channels) and whether original was mono.
    """
    if audio.ndim == 1:
        return audio[:, None], True
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")

    # Heuristic: if first dim is small, assume (channels, samples)
    if audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
        return audio.T, audio.shape[0] == 1
    return audio, audio.shape[1] == 1


class Separator:
    """
    Simple downstream-friendly API for vocal separation.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None,
        config_path: str | Path | None = None,
        model_type: str = "mel_band_roformer",
        device: str | torch.device | None = None,
        device_ids: list[int] | None = None,
        use_amp: bool = True,
    ) -> None:
        self.config = load_config(config_path)
        self.model_type = model_type
        self.use_amp = use_amp

        if model_path is None:
            env_model_path = os.environ.get("MEL_BAND_ROFORMER_CKPT")
            if not env_model_path:
                raise ValueError("model_path is required (or set $MEL_BAND_ROFORMER_CKPT)")
            model_path = env_model_path

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device: torch.device = device

        model = get_model_from_config(self.model_type, self.config)

        if model_path is not None:
            load_state_dict(model, str(model_path))

        if self.device.type == "cuda" and device_ids and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids).to(self.device)
        else:
            model = model.to(self.device)

        model.eval()
        self.model = model

    def separate_audio(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        return_instrumental: bool = True,
    ) -> SeparationResult:
        expected_sr = int(getattr(self.config.model, "sample_rate", sample_rate))
        if sample_rate != expected_sr:
            warnings.warn(
                f"Input sample_rate={sample_rate} differs from config sample_rate={expected_sr}. "
                "Results may be degraded; resample to match for best quality.",
                RuntimeWarning,
                stacklevel=2,
            )

        audio_sc, input_was_mono = _to_samples_channels(np.asarray(audio))

        expects_stereo = bool(getattr(self.config.model, "stereo", False))
        expected_channels = 2 if expects_stereo else 1

        if audio_sc.shape[1] == expected_channels:
            audio_for_model = audio_sc
        elif audio_sc.shape[1] == 1 and expected_channels == 2:
            audio_for_model = np.repeat(audio_sc, 2, axis=1)
        elif audio_sc.shape[1] == 2 and expected_channels == 1:
            audio_for_model = audio_sc.mean(axis=1, keepdims=True)
        else:
            raise ValueError(
                f"Unsupported channel count: got {audio_sc.shape[1]} channels, expected {expected_channels}"
            )

        want_mono_output = input_was_mono or expected_channels == 1
        orig_out = (
            (audio_sc[:, 0] if audio_sc.shape[1] == 1 else audio_sc.mean(axis=1))
            if want_mono_output
            else audio_sc
        )

        mixture = torch.tensor(audio_for_model.T, dtype=torch.float32)
        stems = demix_track(
            config=self.config,
            model=self.model,
            mix=mixture,
            device=self.device,
            use_amp=self.use_amp,
        )

        stems_out: dict[str, np.ndarray] = {}
        for name, stem in stems.items():
            wav_sc = stem.T  # (samples, channels)
            if want_mono_output:
                stems_out[name] = wav_sc[:, 0]
            else:
                stems_out[name] = wav_sc

        vocals = stems_out.get(str(self.config.training.target_instrument or "vocals"))
        instrumental = None
        if return_instrumental and vocals is not None:
            instrumental = orig_out - vocals

        return SeparationResult(
            vocals=vocals,
            instrumental=instrumental,
            stems=stems_out,
            sample_rate=sample_rate,
        )

    def separate_file(self, input_path: str | Path, output_dir: str | Path) -> SeparationResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        audio, sr = sf.read(str(input_path), always_2d=True)

        result = self.separate_audio(audio, sample_rate=int(sr), return_instrumental=True)

        stem_prefix = input_path.stem
        for name, wav in result.stems.items():
            out = output_dir / f"{stem_prefix}_{name}.wav"
            sf.write(str(out), wav, sr, subtype="FLOAT")

        if result.instrumental is not None:
            out = output_dir / f"{stem_prefix}_instrumental.wav"
            sf.write(str(out), result.instrumental, sr, subtype="FLOAT")

        return result

    def separate_folder(self, input_dir: str | Path, output_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(input_dir.glob("*.wav"))
        for wav in wavs:
            self.separate_file(wav, output_dir)
