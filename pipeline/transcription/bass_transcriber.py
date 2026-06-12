"""Bass transcription using Basic Pitch (Spotify).

Requires:
    pip install 'basic-pitch[onnx]'

Basic Pitch v0.3+ uses ONNX Runtime on most platforms — no TensorFlow required.
Compatible with the same PyTorch venv as Demucs and YourMT3+.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Frequency bounds for bass guitar.
# Lower bound: just below B0 (30.87 Hz) to absorb standard 4-string open E (41.2 Hz).
# Upper bound: None — the Demucs bass stem already limits high-freq content.
_MIN_FREQ_HZ: float = 30.0


def transcribe_bass(wav_path: Path, output_dir: Path, stem_name: str = "bass") -> Path:
    """Transcribe a bass stem wav to MIDI using Basic Pitch.

    Args:
        wav_path:   Path to the bass stem wav produced by stem_separation.
        output_dir: Directory where the MIDI file will be saved.
        stem_name:  Label appended to the output filename (default: "bass").

    Returns:
        Path to the saved MIDI file.

    Raises:
        FileNotFoundError: If wav_path does not exist.
        ImportError:       If basic-pitch is not installed.
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError as exc:
        raise ImportError(
            "basic-pitch not installed: pip install basic-pitch"
        ) from exc

    wav_path = Path(wav_path)
    output_dir = Path(output_dir)

    if not wav_path.exists():
        raise FileNotFoundError(f"Wav not found: {wav_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running Basic Pitch transcription: %s", wav_path.name)
    _, midi_data, _ = predict(
        str(wav_path),
        ICASSP_2022_MODEL_PATH,
        minimum_frequency=_MIN_FREQ_HZ,
    )

    out_stem = wav_path.stem + (f"_{stem_name}" if stem_name else "")
    midi_path = output_dir / (out_stem + ".mid")
    midi_data.write(str(midi_path))

    logger.info("MIDI saved: %s", midi_path)
    return midi_path
