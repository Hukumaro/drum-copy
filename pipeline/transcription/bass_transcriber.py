"""Bass transcription using YourMT3+ (shared model with drum transcription)."""
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_non_drum_notes(src_path: Path, dst_path: Path) -> None:
    """Write a new MIDI keeping all notes except channel-9 (GM percussion).

    The bass stem is already isolated by Demucs, so the full YourMT3+ output
    is kept. Channel 9 is excluded to remove any percussion false-positives.
    """
    import mido

    mid = mido.MidiFile(str(src_path))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        filtered = [
            msg for msg in track
            if msg.type not in ("note_on", "note_off") or msg.channel != 9
        ]
        has_notes = any(m.type in ("note_on", "note_off") for m in filtered)
        if has_notes:
            new_track = mido.MidiTrack()
            new_track.extend(filtered)
            out.tracks.append(new_track)

    if not out.tracks:
        logger.warning(
            "No non-drum notes found in YourMT3+ output (%s); saving full MIDI.",
            src_path.name,
        )
        shutil.copy2(str(src_path), str(dst_path))
    else:
        out.save(str(dst_path))


def transcribe_bass(wav_path: Path, output_dir: Path, stem_name: str = "bass") -> Path:
    """Transcribe a bass stem wav to MIDI using YourMT3+.

    Reuses the same model instance as drum transcription (loaded once, cached).

    Args:
        wav_path:   Path to the bass stem wav produced by stem_separation.
        output_dir: Directory where the MIDI file will be saved.
        stem_name:  Label appended to the output filename (default: "bass").

    Returns:
        Path to the saved MIDI file.

    Raises:
        FileNotFoundError: If wav_path does not exist.
    """
    from pipeline.transcription.transcriber import _load_model, _DEFAULT_YMT3_DIR

    wav_path = Path(wav_path)
    output_dir = Path(output_dir)

    if not wav_path.exists():
        raise FileNotFoundError(f"Wav not found: {wav_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ymt3_dir = Path(os.environ.get("YMT3_DIR", str(_DEFAULT_YMT3_DIR)))
    model = _load_model(ymt3_dir)

    src = ymt3_dir / "amt" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torchaudio
    from model_helper import transcribe as _ymt3_transcribe

    info = torchaudio.info(str(wav_path))
    audio_info = {
        "filepath": str(wav_path),
        "track_name": wav_path.stem,
        "sample_rate": info.sample_rate,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration": info.num_frames / info.sample_rate,
        "encoding": info.encoding,
    }

    logger.info("Running YourMT3+ bass transcription: %s", wav_path.name)
    raw_midi = Path(_ymt3_transcribe(model, audio_info))

    out_stem = wav_path.stem + (f"_{stem_name}" if stem_name else "")
    midi_path = output_dir / (out_stem + ".mid")

    _extract_non_drum_notes(raw_midi, midi_path)
    logger.info("MIDI saved: %s", midi_path)
    return midi_path
