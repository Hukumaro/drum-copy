from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelinePaths:
    input_dir: Path
    output_dir: Path
    tmp_dir: Path


class PipelineBackend(ABC):
    """Abstract base for pipeline execution backends.

    Concrete subclasses handle environment-specific concerns:
    - File storage location (local disk, Google Drive, object storage)
    - Compute setup (GPU check, package installation)

    To add a serverless GPU backend (Modal, RunPod, Replicate, etc.):
    1. Subclass PipelineBackend
    2. Override setup() to authenticate and configure the remote provider
    3. Override get_paths() to return paths accessible from the remote environment
       (or override the pipeline execution itself to dispatch jobs remotely)
    """

    def setup(self) -> None:
        """One-time environment setup (mount drives, check GPU, etc.)."""
        pass

    @abstractmethod
    def get_paths(self) -> PipelinePaths:
        """Return resolved input / output / tmp directories."""
        ...

    def teardown(self) -> None:
        """Post-run cleanup. Override as needed."""
        pass
