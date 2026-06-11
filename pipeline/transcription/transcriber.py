"""Drum stem transcription using Google MT3 (Multi-Task Music Transcription).

Required packages:
    pip install git+https://github.com/magenta/mt3
    pip install t5x seqio note-seq 'jax[cuda12_pip]'

Checkpoint (download once):
    gsutil -q -m cp -r gs://mt3/checkpoints/mt3/ /tmp/mt3/
    or export MT3_CHECKPOINT=/path/to/checkpoint/
"""
import functools
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_DEFAULT_CHECKPOINT = os.environ.get("MT3_CHECKPOINT", "/tmp/mt3/mt3/")

_cached_model = None
_cached_ckpt = None


class _InferenceModel:
    """T5X-based MT3 transcription model.

    Adapted from the MT3 Colab notebook:
    https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/
    music_transcription_with_transformers.ipynb
    """

    def __init__(self, checkpoint_path: str, model_type: str = "mt3") -> None:
        import gin
        import seqio
        import t5x
        import t5x.adafactor
        import t5x.partitioning
        import t5x.utils
        import tensorflow as tf
        import mt3 as _mt3_root
        from mt3 import models as mt3_models, network, spectrograms, vocabularies

        mt3_dir = Path(_mt3_root.__file__).parent
        gin.clear_config()
        gin.parse_config_files_and_bindings(
            config_files=[
                str(mt3_dir / "gin" / "model.gin"),
                str(mt3_dir / "gin" / f"{model_type}.gin"),
            ],
            bindings=[],
        )

        self._batch_size = 8
        self._inputs_length = 512    # spectrogram frames per chunk (~4 s at 16 kHz)
        self._outputs_length = 1024  # max output token length

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabConfig())
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        input_depth = spectrograms.input_depth(self.spectrogram_config)

        output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }
        partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        t5x_model = mt3_models.ContinuousInputsEncoderDecoderModel(
            module=network.Transformer(config=gin.query_parameter("%MODEL")),
            input_vocabulary=output_features["inputs"].vocabulary,
            output_vocabulary=output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=input_depth,
        )
        restore_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )
        initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=t5x_model.optimizer_def,
            init_fn=t5x_model.get_initial_variables,
            input_shapes={
                "encoder_input_tokens": (self._batch_size, self._inputs_length, input_depth),
                "decoder_input_tokens": (self._batch_size, self._outputs_length),
            },
            partitioner=partitioner,
        )
        self._train_state = list(initializer.from_checkpoints([restore_cfg]))[0]
        train_state_axes = initializer.train_state_axes

        self._predict_fn = partitioner.partition(
            functools.partial(
                t5x_model.predict_batch_with_aux,
                decoder_params={"max_decode_steps": self._outputs_length},
                return_all_decodes=False,
                num_decodes=1,
            ),
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec("data"),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def __call__(self, audio: np.ndarray):
        """Transcribe 1-D 16 kHz audio to a note_seq.NoteSequence."""
        from mt3 import metrics_utils, note_sequences

        frames, frame_times = self._audio_to_frames(audio)
        all_predictions = []
        for i in range(0, len(frames), self._batch_size):
            batch = frames[i : i + self._batch_size]
            n = len(batch)
            if n < self._batch_size:
                pad = np.zeros(
                    (self._batch_size - n, *batch.shape[1:]), dtype=batch.dtype
                )
                batch = np.concatenate([batch, pad])
            tokens, _ = self._predict_fn(
                self._train_state.params,
                {
                    "encoder_input_tokens": batch,
                    "decoder_input_tokens": np.zeros(
                        (self._batch_size, self._outputs_length), dtype=np.int32
                    ),
                },
                None,
            )
            all_predictions.extend(tokens[:n].tolist())

        result = metrics_utils.event_predictions_to_ns(
            [
                {"est_tokens": p, "start_time": t, "raw_inputs": []}
                for p, t in zip(all_predictions, frame_times)
            ],
            codec=self.codec,
            encoding_spec=note_sequences.NoteEncodingWithTiesSpec,
        )
        return result["est_ns"]

    def _audio_to_frames(self, audio: np.ndarray):
        """Compute mel spectrogram and split into fixed-length chunks."""
        from mt3 import spectrograms as spec_lib

        spectrogram = spec_lib.compute_spectrogram(audio, self.spectrogram_config)
        frames, times = [], []
        sr = self.spectrogram_config.sample_rate
        hw = self.spectrogram_config.hop_width

        for start in range(0, spectrogram.shape[0], self._inputs_length):
            chunk = spectrogram[start : start + self._inputs_length]
            if chunk.shape[0] < self._inputs_length:
                chunk = np.pad(chunk, [(0, self._inputs_length - chunk.shape[0]), (0, 0)])
            frames.append(chunk)
            times.append(start * hw / sr)

        return np.array(frames, dtype=np.float32), times


def transcribe(drums_wav_path: Path, output_dir: Path) -> Path:
    """Transcribe a drum stem wav to MIDI using Google MT3.

    Args:
        drums_wav_path: Path to the drum stem wav produced by stem_separation.
        output_dir:     Directory where the final MIDI file will be saved.

    Returns:
        Path to the saved MIDI file.

    Raises:
        FileNotFoundError: If drums_wav_path does not exist.
        ImportError:       If MT3 dependencies are not installed.
    """
    global _cached_model, _cached_ckpt

    try:
        import librosa
        import note_seq
    except ImportError as exc:
        raise ImportError(
            "Required packages not installed: pip install librosa note-seq"
        ) from exc

    drums_wav_path = Path(drums_wav_path)
    output_dir = Path(output_dir)

    if not drums_wav_path.exists():
        raise FileNotFoundError(f"Drum wav not found: {drums_wav_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading audio at %d Hz: %s", _SAMPLE_RATE, drums_wav_path)
    audio, _ = librosa.load(str(drums_wav_path), sr=_SAMPLE_RATE, mono=True)

    if _cached_model is None or _cached_ckpt != _DEFAULT_CHECKPOINT:
        logger.info("Loading MT3 model from: %s", _DEFAULT_CHECKPOINT)
        _cached_model = _InferenceModel(_DEFAULT_CHECKPOINT)
        _cached_ckpt = _DEFAULT_CHECKPOINT

    logger.info("Running MT3 transcription...")
    note_sequence = _cached_model(audio)

    midi_path = output_dir / (drums_wav_path.stem + ".mid")
    note_seq.sequence_proto_to_midi_file(note_sequence, str(midi_path))

    logger.info("MIDI saved: %s", midi_path)
    return midi_path
