import logging
import pkgutil
import shutil
from pathlib import Path

# Python 3.12 removed pkgutil.ImpImporter, but older deps (pkg_resources, music21) still reference it.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = type(None)  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def transcribe(drums_wav_path: Path, output_dir: Path) -> Path:
    """Transcribe a drum stem wav to MIDI using omnizart.

    The raw omnizart output is saved without any post-processing
    (no quantization, no note filtering, no velocity adjustment).

    Args:
        drums_wav_path: Path to the drum stem wav produced by stem_separation.
        output_dir:     Directory where the final MIDI file will be saved.

    Returns:
        Path to the saved MIDI file.

    Raises:
        FileNotFoundError: If drums_wav_path does not exist.
        RuntimeError:      If omnizart fails to produce a MIDI file.
    """
    drums_wav_path = Path(drums_wav_path)
    output_dir = Path(output_dir)

    if not drums_wav_path.exists():
        raise FileNotFoundError(f"Drum wav not found: {drums_wav_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from omnizart.drum import app as drum_app
    except ImportError as exc:
        raise ImportError(
            "omnizart is not installed. Run: pip install omnizart"
        ) from exc

    logger.info("Running omnizart drum transcription on: %s", drums_wav_path)

    # omnizart writes the MIDI next to the input file by default.
    # We pass output_dir so it lands in the right place directly.
    result = drum_app.transcribe(str(drums_wav_path), output=str(output_dir))

    # omnizart returns the output path as a string.
    if result:
        midi_path = Path(result)
    else:
        # Fallback: search for any .mid written into output_dir.
        candidates = sorted(output_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(
                f"omnizart did not produce a MIDI file in {output_dir}"
            )
        midi_path = candidates[-1]

    logger.info("MIDI saved: %s", midi_path)
    return midi_path
