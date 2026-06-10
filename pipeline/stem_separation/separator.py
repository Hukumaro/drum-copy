import logging
from pathlib import Path

import soundfile as sf
import torch

from demucs.apply import apply_model
from demucs.audio import AudioFile
from demucs.pretrained import get_model

logger = logging.getLogger(__name__)


def separate(input_path: Path, tmp_dir: Path, model: str = "htdemucs") -> Path:
    """Separate drum stem from an audio file using Demucs internal API.

    Args:
        input_path: Path to the input audio file (wav or mp3).
        tmp_dir:    Directory where the drum stem wav will be saved.
        model:      Demucs model name (default: htdemucs).

    Returns:
        Path to the separated drums.wav file.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError:      If the drum stem cannot be extracted.
    """
    input_path = Path(input_path)
    tmp_dir = Path(tmp_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if device == "cpu":
        logger.warning("CUDA not available — running on CPU. Processing will be slow.")

    logger.info("Loading Demucs model: %s", model)
    demucs_model = get_model(name=model)
    demucs_model.to(device)
    demucs_model.eval()

    logger.info("Loading audio: %s", input_path)
    wav = AudioFile(input_path).read(
        streams=0,
        samplerate=demucs_model.samplerate,
        channels=demucs_model.audio_channels,
    )

    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()

    logger.info("Separating stems...")
    sources = apply_model(
        demucs_model,
        wav[None],
        device=device,
        shifts=1,
        split=True,
        overlap=0.25,
        progress=True,
    )[0]

    sources *= ref.std()
    sources += ref.mean()

    if "drums" not in demucs_model.sources:
        raise RuntimeError(
            f"Model '{model}' has no 'drums' stem. "
            f"Available: {demucs_model.sources}"
        )

    drum_idx = demucs_model.sources.index("drums")
    drums_tensor = sources[drum_idx]

    out_dir = tmp_dir / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    drums_path = out_dir / "drums.wav"

    # soundfile expects (samples, channels); drums_tensor is (channels, samples)
    drums_np = drums_tensor.cpu().numpy().T
    sf.write(str(drums_path), drums_np, samplerate=demucs_model.samplerate)

    logger.info("Drum stem saved: %s", drums_path)
    return drums_path
