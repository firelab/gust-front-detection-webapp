"""
Frame Data API – returns the list of rendered frames for a completed job.
"""

from flask import jsonify
from src.nfgda_service.nfgda_service import NfgdaService


def get_frames(job_id: str, nfgda_service: NfgdaService):
    """
    Response shape:
    {
        "jobId": "<jobId>",
        "frameUrls": ["https://...frame0.png", ...],
        "timestampsUtc": ["2024-07-07T01:22:24Z", ...],
        "bounds": {
            "minLat": 34.0, "minLon": -107.0,
            "maxLat": 36.0, "maxLon": -105.0
        }
    }
    """
    pass
