"""
Algorithm / Job Status API – returns the current status of a processing job.
"""

from flask import jsonify
from src.nfgda_service.nfgda_service import NfgdaService


# Add redis client to the arguments so that it can see the container
def get_status(job_id: str, redis_client, nfgda_service: NfgdaService):
    """
    Response shape:
    {
        "id": "<jobId>",
        "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
        "error": ""
    }
    """

    job = redis_client.hgetall(job_id)  # reads the hash from Redis

    if not job:
        # job doesn't exist — return a 404
        return jsonify({"error": "Job not found"}), 404

    # job exists — return id and status as JSON
    return jsonify(
        {"id": job_id, "status": job.get("status", ""), "error": job.get("error", "")}
    ), 200
