from __future__ import annotations

# Backwards-compatible wrapper. Prefer the installed CLI:
#   mel-band-roformer-separate --model_path ... --input_folder ... --store_dir ...

from pathlib import Path
import sys


def _import_main():
    try:
        from mel_band_roformer_vocal.cli import main as _main

        return _main
    except ModuleNotFoundError:
        # Allows running from a fresh clone without `pip install -e .`
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root / "src"))
        from mel_band_roformer_vocal.cli import main as _main  # type: ignore[no-redef]

        return _main


main = _import_main()


if __name__ == "__main__":
    raise SystemExit(main())
