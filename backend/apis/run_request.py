"""
Algorithm Runner API – accepts a station ID and time frame, kicks off an
NFGDA processing job, and returns the new job ID.
"""

from flask import jsonify, request
from src.nfgda_service.nfgda_service import NfgdaService


def start_run(nfgda_service: NfgdaService):
    """
    Expected JSON body:
    {
        "stationId": "KABX",
        "startUtc": "2024-07-07T01:22:24Z",
        "endUtc":   "2024-07-07T03:48:02Z",
        "options":  {}
    }

    Response shape:
    {
        "id": "<jobId>",
        "status": "PENDING"
    }
    """
    pass
