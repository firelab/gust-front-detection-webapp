"""
Algorithm / Job Status API – returns the current status of a processing job.
"""

from flask import jsonify
from src.nfgda_service.nfgda_service import NfgdaService


def get_status(job_id: str, nfgda_service: NfgdaService):
    """
    Response shape:
    {
        "id": "<jobId>",
        "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
        "error": ""
    }
    """
    pass
