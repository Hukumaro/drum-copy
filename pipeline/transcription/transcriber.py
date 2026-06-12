"""Drum transcription using YourMT3+ (mimbres/YourMT3).

Setup (handled automatically on first call, or pre-run in Colab setup cell):
    git clone --depth=1 https://huggingface.co/spaces/mimbres/YourMT3 /tmp/ymt3
    pip install -r /tmp/ymt3/requirements.txt

Model weights (~600 MB) download automatically from HuggingFace on first call.
Override the clone location with: export YMT3_DIR=/path/to/ymt3
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_YMT3_DIR = Path(os.environ.get("YMT3_DIR", "/tmp/ymt3"))

# YPTF.MoE+ Multi (noPS) — best general accuracy, no pitch-shifting augmentation
_CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
_CHECKPOINT_ARGS = [
    _CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
    "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
    "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
    "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "16",
]

_cached_model = None


def _ensure_ymt3(ymt3_dir: Path) -> Path:
    """Return the YourMT3+ src directory, cloning the repo if not already present."""
    src = ymt3_dir / "amt" / "src"
    if src.is_dir():
        return src

    logger.info("Cloning YourMT3+ to %s ...", ymt3_dir)
    ymt3_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://huggingface.co/spaces/mimbres/YourMT3",
         str(ymt3_dir)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         str(ymt3_dir / "requirements.txt")],
        check=True,
    )
    return src


def _load_model(ymt3_dir: Path):
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    src = _ensure_ymt3(ymt3_dir)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from model_helper import load_model_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading YourMT3+ checkpoint on %s ...", device)
    _cached_model = load_model_checkpoint(args=_CHECKPOINT_ARGS, device=device)
    return _cached_model


def _extract_drum_channel(src_path: Path, dst_path: Path) -> None:
    """Write a new MIDI containing only channel-9 (GM percussion) note events.

    Meta messages (tempo, time signature, etc.) are preserved in each drum track.
    If no channel-9 notes are found, the full source MIDI is copied unchanged —
    this avoids an empty output when the model assigns drums to a non-standard channel.
    """
    import mido
    import shutil

    mid = mido.MidiFile(str(src_path))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        # Keep meta messages + channel-9 note events; drop all other note events.
        filtered = [
            msg for msg in track
            if msg.type not in ("note_on", "note_off") or msg.channel == 9
        ]
        has_notes = any(m.type in ("note_on", "note_off") for m in filtered)
        if has_notes:
            new_track = mido.MidiTrack()
            new_track.extend(filtered)
            out.tracks.append(new_track)

    if not out.tracks:
        logger.warning(
            "No channel-9 notes found in YourMT3+ output (%s); saving full MIDI.",
            src_path.name,
        )
        shutil.copy2(str(src_path), str(dst_path))
    else:
        out.save(str(dst_path))


def transcribe(wav_path: Path, output_dir: Path, stem_name: str = "") -> Path:
    """Transcribe a drum stem wav to MIDI using YourMT3+.

    Args:
        wav_path:   Path to the drum stem wav produced by stem_separation.
        output_dir: Directory where the MIDI file will be saved.
        stem_name:  Optional label appended to the output filename
                    (e.g. "drums" → "{wav_stem}_drums.mid").

    Returns:
        Path to the saved MIDI file.

    Raises:
        FileNotFoundError: If wav_path does not exist.
    """
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

    logger.info("Running YourMT3+ transcription: %s", wav_path.name)
    raw_midi = Path(_ymt3_transcribe(model, audio_info))

    out_stem = wav_path.stem + (f"_{stem_name}" if stem_name else "")
    midi_path = output_dir / (out_stem + ".mid")

    _extract_drum_channel(raw_midi, midi_path)
    logger.info("MIDI saved: %s", midi_path)
    return midi_path
