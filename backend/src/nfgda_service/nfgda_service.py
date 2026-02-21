from src.nfgda_service.models import JobStatus, Job
from src.nfgda_service.nfgda_runner import NfgdaRunner

class NfgdaService:
    """High-level service that orchestrates NFGDA runs, job tracking, and asset management."""

    def __init__(self) -> None:
        self.runner = NfgdaRunner()
        self.default_out_prefix = "" #TODO: figure out where we storing things


    def run_job(self, job: Job) -> JobStatus:
        """Run the NFGDA algorithm."""
        pass
