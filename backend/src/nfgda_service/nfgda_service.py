from src.nfgda_service.asset_store import AssetStore
from src.nfgda_service.job_store import JobStore
from src.nfgda_service.models import FramesIndex, Job, RunRequest
from src.nfgda_service.nfgda_runner import NfgdaRunner


class NfgdaService:
    """High-level service that orchestrates NFGDA runs, job tracking, and asset management."""

    def __init__(
        self,
        runner: NfgdaRunner,
        job_queue: (Job),
        job_store: JobStore,
        asset_store: AssetStore,
        default_out_prefix: str,
    ) -> None:
        self._runner = runner
        self._job_store = job_store
        self._asset_store = asset_store
        self._default_out_prefix = default_out_prefix

    def start_run(self, req: RunRequest) -> Job:
        """Accept a RunRequest, create a Job, and kick off the NFGDA runner."""
        pass

    def get_status(self, job_id: str) -> Job:
        """Return the current state of a Job."""
        pass

    def get_frames(self, job_id: str) -> FramesIndex:
        """Return the FramesIndex for a completed Job."""
        pass
