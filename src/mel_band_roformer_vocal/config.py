from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml
from ml_collections import ConfigDict


@dataclass(frozen=True)
class PackageConfig:
    default_yaml: str = "config_vocals_mel_band_roformer.yaml"


def default_config_resource():
    return resources.files("mel_band_roformer_vocal").joinpath("configs", PackageConfig().default_yaml)


def load_config(path: str | Path | None = None) -> ConfigDict:
    if path is None:
        with default_config_resource().open("r") as f:
            data: Any = yaml.load(f, Loader=yaml.FullLoader)
    else:
        from pathlib import Path as _Path

        with _Path(path).open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    return ConfigDict(data)
