from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from typing import Dict, List


# ── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(Enum):
    """Lifecycle status of an NFGDA processing job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ── Value Objects ─────────────────────────────────────────────────────────────


@dataclass
class GeoBounds:
    """Axis-aligned geographic bounding box."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


@dataclass
class RunRequest:
    """Parameters submitted by a client to start a new NFGDA run."""

    station_id: str
    start_utc: datetime
    end_utc: datetime
    options: Dict = field(default_factory=dict)


@dataclass
class Job:
    """Tracks the state of a single NFGDA processing job."""

    id: uuid.UUID
    status: JobStatus
    error: str = ""
    output_path: str = ""
    run_request: RunRequest = None


@dataclass
class FramesIndex:
    """Index of rendered frames produced by a completed job."""

    job_id: str
    frame_urls: List[str] = field(default_factory=list)
    timestamps_utc: List[datetime] = field(default_factory=list)
    bounds: GeoBounds = None
