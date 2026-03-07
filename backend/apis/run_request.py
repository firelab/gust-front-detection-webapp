
import uuid
from datetime import datetime, timedelta, timezone
from flask import jsonify, request

def send_job_to_redis_queue(redis_client, request_fields: dict):
    """
    Expected JSON body via request_fields:
    {
        "stationId": "KABX",
        "startUtc": "2024-07-07T01:22:24Z",  (optional; defaults to now)
        "endUtc":   "2024-07-07T03:48:02Z"   (optional; defaults to startUtc + 30s)
    }

    Response shape:
    {
        "job_id": "<jobId>",
        "status": 200
        OR
        "error": "<error message>",
        "status": 400
    }
    """

    # Validate stationId
    if not request_fields.get("stationId"):
        return jsonify({"error": "Missing stationId request field", "status": 400})

    # Validate and/or set default timebox parameters
    validation_error = validate_time_parameters(request_fields)
    if validation_error:
        return validation_error

    # generate job id via uuidv5
    job_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 
        request_fields["stationId"] + 
        request_fields["startUtc"] + 
        request_fields["endUtc"])
    )

    # add job to redis
    job_key = f"job:{job_id}"
    redis_client.hset(job_key, mapping={
        "stationId": request_fields["stationId"],
        "startUtc": request_fields["startUtc"],
        "endUtc": request_fields["endUtc"],
        "status": "PENDING"
    })

    # push job id to job queue
    redis_client.lpush("job_queue", job_id)

    # the cat's meow
    return jsonify({"job_id": job_id, "status": 200})


def validate_time_parameters(request_fields: dict):
    """Validate the time parameters recieved via the request."""
    
    # Default timebox when not provided: look back 15 minutes from now
    # so the algorithm captures 2-3 recent NEXRAD scans for detection + forecast
    # (2 scan minimum needed for forcasting)
    now = datetime.now(timezone.utc)
    if not request_fields.get("startUtc") and not request_fields.get("endUtc"):
        request_fields["startUtc"] = (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        request_fields["endUtc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif not request_fields.get("startUtc") or not request_fields.get("endUtc"):
        return jsonify({"error": "Must provide both startUtc and endUtc, or neither", "status": 400})

    # Parse and validate timebox
    try:
        start_utc = datetime.strptime(request_fields["startUtc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        end_utc = datetime.strptime(request_fields["endUtc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Expected ISO 8601: YYYY-MM-DDTHH:MM:SSZ", "status": 400})

    # startUtc must be within the last 2 hours
    if start_utc < now - timedelta(minutes=120):
        return jsonify({"error": "startUtc must be within the last 2 hours", "status": 400})

    # endUtc must be after startUtc
    if end_utc <= start_utc:
        return jsonify({"error": "endUtc must be after startUtc", "status": 400})

    # Duration must be between 5 minutes and 6 hours
    duration = end_utc - start_utc
    if duration < timedelta(minutes=5):
        return jsonify({"error": "Timebox duration must be at least 5 minutes", "status": 400})
    if duration > timedelta(hours=6):
        return jsonify({"error": "Timebox duration must not exceed 6 hours", "status": 400})

    # endUtc must not be in the future
    if end_utc > now:
        return jsonify({"error": "endUtc must not be later than the current time", "status": 400})

    return None
