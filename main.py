"""Drum extraction and transcription pipeline.

Usage:
    python main.py <input_file_or_dir> [options]

Examples:
    python main.py input/song.wav
    python main.py input/
    python main.py input/song.wav --model htdemucs --keep-tmp --overwrite
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from pipeline.stem_separation.separator import separate
from pipeline.transcription.transcriber import transcribe

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TMP_DIR = BASE_DIR / "tmp"

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_file(
    input_path: Path,
    output_dir: Path,
    tmp_dir: Path,
    model: str,
    keep_tmp: bool,
    overwrite: bool,
) -> Path:
    """Run the full pipeline on a single audio file."""
    midi_out = output_dir / (input_path.stem + ".mid")
    if midi_out.exists() and not overwrite:
        logger.warning("Skipping (already exists, use --overwrite): %s", midi_out)
        return midi_out

    logger.info("=== Step 1: Stem separation — %s ===", input_path.name)
    drums_wav = separate(input_path, tmp_dir, model=model)

    logger.info("=== Step 2: Drum transcription — %s ===", drums_wav.name)
    midi_path = transcribe(drums_wav, output_dir)

    # Rename to match the original audio file stem if omnizart used a different name.
    if midi_path.stem != input_path.stem:
        renamed = midi_path.parent / (input_path.stem + ".mid")
        midi_path.rename(renamed)
        midi_path = renamed

    if not keep_tmp:
        stem_tmp = tmp_dir / input_path.stem
        if stem_tmp.exists():
            shutil.rmtree(stem_tmp)
            logger.debug("Removed tmp directory: %s", stem_tmp)

    logger.info("Done: %s", midi_path)
    return midi_path


def collect_inputs(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file format: %s", target.suffix)
            sys.exit(1)
        return [target]
    if target.is_dir():
        files = [
            p for p in sorted(target.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            logger.error("No supported audio files found in: %s", target)
            sys.exit(1)
        return files
    logger.error("Path does not exist: %s", target)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drum extraction & transcription pipeline")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(INPUT_DIR),
        help="Input audio file or directory (default: input/)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory for MIDI files (default: output/)",
    )
    parser.add_argument(
        "--tmp",
        default=str(TMP_DIR),
        help="Temporary directory for drum stem wav (default: tmp/)",
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Demucs model name (default: htdemucs)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep intermediate drum stem wav files after processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MIDI output files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    tmp_dir = Path(args.tmp)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    inputs = collect_inputs(Path(args.input))
    logger.info("Found %d file(s) to process.", len(inputs))

    results: list[tuple[str, str]] = []
    for audio_path in inputs:
        try:
            midi = process_file(
                audio_path,
                output_dir,
                tmp_dir,
                model=args.model,
                keep_tmp=args.keep_tmp,
                overwrite=args.overwrite,
            )
            results.append((audio_path.name, str(midi)))
        except Exception as exc:
            logger.error("Failed to process %s: %s", audio_path.name, exc)
            results.append((audio_path.name, f"ERROR: {exc}"))

    print("\n--- Results ---")
    for src, dest in results:
        print(f"  {src}  ->  {dest}")


if __name__ == "__main__":
    main()
