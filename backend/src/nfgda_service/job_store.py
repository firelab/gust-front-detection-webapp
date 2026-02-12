from typing import Dict

from src.nfgda_service.models import Job, RunRequest


class JobStore:
    """In-memory store that manages the lifecycle of Job objects."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    def create(self, req: RunRequest) -> Job:
        """Create a new Job from a RunRequest and persist it."""
        pass

    def update(self, job: Job) -> None:
        """Update an existing Job in the store."""
        pass

    def get(self, job_id: str) -> Job:
        """Retrieve a Job by its id."""
        pass
