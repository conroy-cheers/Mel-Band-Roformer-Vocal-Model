from __future__ import annotations

# Backwards-compat exports (old scripts used `utils.get_model_from_config` / `utils.demix_track`)

from pathlib import Path
import sys

try:
    from mel_band_roformer_vocal.demix import demix_track  # noqa: F401
    from mel_band_roformer_vocal.modeling import get_model_from_config  # noqa: F401
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))
    from mel_band_roformer_vocal.demix import demix_track  # type: ignore[no-redef]  # noqa: F401
    from mel_band_roformer_vocal.modeling import get_model_from_config  # type: ignore[no-redef]  # noqa: F401

__all__ = ["demix_track", "get_model_from_config"]
