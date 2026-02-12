from src.nfgda_service.models import FramesIndex


class AssetStore:
    """Persists and retrieves rendered frame assets on disk."""

    def __init__(self, base_dir: str, public_url_prefix: str) -> None:
        self._base_dir = base_dir
        self._public_url_prefix = public_url_prefix

    def write_frames(self, job_id: str, idx: FramesIndex) -> None:
        """Write a FramesIndex to persistent storage."""
        pass

    def read_frames(self, job_id: str) -> FramesIndex:
        """Read the FramesIndex for the given job from storage."""
        pass
