
from flask import jsonify

# Add redis client to the arguments so that it can see the container
def get_status(job_id: str, redis_client):
    """
    Response shape:
    {
        "id": "<jobId>",
        "status": "PENDING" | "PROCESSING" | "COMPLETE" | "FAILED"
    }
    """

    job = redis_client.hgetall(job_id)  # reads the hash from Redis

    if not job:
        # job doesn't exist
        return jsonify({"error": "Job not found", "status": 400})

    # job exists
    return jsonify({"id": job_id, "jobStatus": job.get("status", ""), "status": 200})
