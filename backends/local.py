import logging
from pathlib import Path

from .base import PipelineBackend, PipelinePaths

logger = logging.getLogger(__name__)


class LocalBackend(PipelineBackend):
    """Runs the pipeline on the local machine."""

    def __init__(self, input_dir: Path, output_dir: Path, tmp_dir: Path) -> None:
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._tmp_dir = Path(tmp_dir)

    def setup(self) -> None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("CUDA not available — running on CPU. Processing will be slow.")
            else:
                logger.info("GPU: %s", torch.cuda.get_device_name(0))
        except ImportError:
            pass

        for d in [self._output_dir, self._tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_paths(self) -> PipelinePaths:
        return PipelinePaths(
            input_dir=self._input_dir,
            output_dir=self._output_dir,
            tmp_dir=self._tmp_dir,
        )
