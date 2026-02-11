from __future__ import annotations

from ._version import __version__

__all__ = ["__version__", "Separator"]


def __getattr__(name: str):
    if name == "Separator":
        from .separator import Separator  # heavy deps (torch, librosa)

        return Separator
    raise AttributeError(name)

