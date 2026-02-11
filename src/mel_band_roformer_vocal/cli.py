from __future__ import annotations

import argparse
import os
from pathlib import Path

from .separator import Separator


def _expand_path(p: Path | None) -> Path | None:
    if p is None:
        return None
    return Path(os.path.expandvars(str(p))).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mel-band-roformer-separate")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        required=False,
        help="Path to .ckpt or state_dict .pt/.pth. If omitted, uses $MEL_BAND_ROFORMER_CKPT.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Path to config yaml. Defaults to packaged config.",
    )
    parser.add_argument("--input_file", type=Path, default=None, help="Single .wav to process")
    parser.add_argument("--input_folder", type=Path, default=None, help="Folder containing .wav files to process")
    parser.add_argument("--store_dir", type=Path, required=True, help="Directory to write outputs")
    parser.add_argument("--device", type=str, default=None, help="torch device string, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional GPU ids for DataParallel (e.g. --device_ids 0 1)",
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable autocast / AMP")

    args = parser.parse_args(argv)

    args.model_path = _expand_path(args.model_path)
    args.config_path = _expand_path(args.config_path)
    args.input_file = _expand_path(args.input_file)
    args.input_folder = _expand_path(args.input_folder)
    args.store_dir = _expand_path(args.store_dir)

    if args.model_path is None and os.environ.get("MEL_BAND_ROFORMER_CKPT") is None:
        parser.error("Missing --model_path (or set $MEL_BAND_ROFORMER_CKPT)")

    if (args.input_file is None) == (args.input_folder is None):
        parser.error("Specify exactly one of --input_file or --input_folder")

    if args.input_file is not None:
        if not args.input_file.exists():
            parser.error(f"--input_file does not exist: {args.input_file}")
        if args.input_file.suffix.lower() != ".wav":
            parser.error(f"--input_file must be a .wav file: {args.input_file}")

    if args.input_folder is not None:
        if not args.input_folder.exists():
            parser.error(f"--input_folder does not exist: {args.input_folder}")
        if not args.input_folder.is_dir():
            parser.error(f"--input_folder must be a directory: {args.input_folder}")

    separator = Separator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        device_ids=args.device_ids,
        use_amp=not args.no_amp,
    )
    print(f"Using device: {separator.device}")

    args.store_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file is not None:
        print(f"Processing file: {args.input_file}")
        separator.separate_file(args.input_file, args.store_dir)
        return 0

    wavs = sorted(args.input_folder.glob("*.wav"))
    if not wavs:
        parser.error(f"No .wav files found in --input_folder: {args.input_folder}")

    print(f"Found {len(wavs)} .wav files in {args.input_folder}")
    for i, wav in enumerate(wavs, 1):
        print(f"[{i}/{len(wavs)}] {wav.name}")
        separator.separate_file(wav, args.store_dir)
    return 0
