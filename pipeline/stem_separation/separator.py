import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from demucs.apply import apply_model
from demucs.pretrained import get_model

logger = logging.getLogger(__name__)


def separate(
    input_path: Path,
    tmp_dir: Path,
    model: str = "htdemucs",
    stems: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Separate audio stems from an audio file using Demucs internal API.

    Args:
        input_path: Path to the input audio file (wav or mp3).
        tmp_dir:    Directory where stem wav files will be saved.
        model:      Demucs model name (default: htdemucs).
        stems:      Stem names to extract (e.g. ["drums", "bass"]).
                    Defaults to ["drums"]. htdemucs provides:
                    drums, bass, other, vocals.

    Returns:
        Dict mapping stem name to the saved wav Path.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError:      If a requested stem is not available in the model.
    """
    input_path = Path(input_path)
    tmp_dir = Path(tmp_dir)
    if stems is None:
        stems = ["drums"]

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

    available = list(demucs_model.sources)
    missing = [s for s in stems if s not in available]
    if missing:
        raise RuntimeError(
            f"Stem(s) {missing} not available in model '{model}'. "
            f"Available: {available}"
        )

    logger.info("Loading audio: %s", input_path)
    # torchaudio avoids soundfile entirely (uses sox_io backend on Linux/Colab),
    # sidestepping the numpy.dtypes.Float64DType incompatibility in soundfile.
    wav, sr = torchaudio.load(str(input_path))
    if sr != demucs_model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, demucs_model.samplerate)
    if wav.shape[0] == 1 and demucs_model.audio_channels == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 1 and demucs_model.audio_channels == 1:
        wav = wav.mean(0, keepdim=True)

    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()

    logger.info("Separating stems: %s", stems)
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

    out_dir = tmp_dir / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Path] = {}
    for stem in stems:
        idx = available.index(stem)
        wav_path = out_dir / f"{stem}.wav"
        torchaudio.save(str(wav_path), sources[idx].cpu(), demucs_model.samplerate)
        logger.info("%s stem saved: %s", stem, wav_path)
        result[stem] = wav_path

    return result
