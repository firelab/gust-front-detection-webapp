from flask import jsonify


def get_job_status(redis_client, job_id: str):
    """Look up a job's fields in Redis and return them as JSON.

    Response shape:
        { "job_id": "<id>", "stationId": "...", "status": "...", ... }
        OR
        { "error": "Job not found", "status": 400 }
    """
    job_key = f"job:{job_id}"
    job_fields = redis_client.hgetall(job_key)
    if not job_fields:
        return jsonify({"error": "Job not found", "status": 400})

    # Check the job's position in the queue
    queue = redis_client.lrange("job_queue", 0, -1)
    try:
        queue_position = queue.index(job_id) + 1
    except ValueError:
        queue_position = None  # not in queue (already processing or done)

    return jsonify({"job_id": job_id, "queue_position": queue_position, **job_fields})
