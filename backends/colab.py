import logging
from pathlib import Path

from .base import PipelineBackend, PipelinePaths

logger = logging.getLogger(__name__)

GDRIVE_MOUNT = "/content/drive"
GDRIVE_MYDRIVE = f"{GDRIVE_MOUNT}/MyDrive"
DEFAULT_DATA_ROOT = f"{GDRIVE_MYDRIVE}/drum-copy_data"


class ColabBackend(PipelineBackend):
    """Runs the pipeline on Google Colab.

    Code lives at /content/drum-copy (git clone, Colab-local).
    Input / output files live on Google Drive (persisted between sessions).
    Intermediate files use /content/tmp/ (fast local SSD, ephemeral).

    Args:
        data_root:  Google Drive directory that holds input/ and output/.
        use_gdrive: Whether to mount Google Drive automatically.
        tmp_dir:    Temporary directory inside Colab's local storage.
    """

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        use_gdrive: bool = True,
        tmp_dir: str = "/content/tmp/drum-copy",
    ) -> None:
        self._data_root = Path(data_root)
        self._use_gdrive = use_gdrive
        self._tmp_dir = Path(tmp_dir)

    def setup(self) -> None:
        self._check_gpu()
        if self._use_gdrive:
            self._mount_gdrive()
        for d in [self._data_root / "input", self._data_root / "output", self._tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _check_gpu(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPU: %s  (%.1f GB)", torch.cuda.get_device_name(0),
                            torch.cuda.get_device_properties(0).total_memory / 1e9)
            else:
                logger.warning(
                    "No GPU detected. "
                    "Enable via: Runtime > Change runtime type > Hardware accelerator: GPU"
                )
        except ImportError:
            pass

    def _mount_gdrive(self) -> None:
        try:
            from google.colab import drive  # type: ignore[import]
            drive.mount(GDRIVE_MOUNT)
            logger.info("Google Drive mounted at %s", GDRIVE_MOUNT)
        except ImportError as exc:
            raise RuntimeError(
                "google.colab is not available. This backend requires Google Colab."
            ) from exc

    def get_paths(self) -> PipelinePaths:
        return PipelinePaths(
            input_dir=self._data_root / "input",
            output_dir=self._data_root / "output",
            tmp_dir=self._tmp_dir,
        )
