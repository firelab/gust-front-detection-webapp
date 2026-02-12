from src.nfgda_service.models import RunRequest


class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, req: RunRequest, out_dir: str) -> None:
        """Run the NFGDA process, writing output to *out_dir*."""
        pass
