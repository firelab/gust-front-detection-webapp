
import os
import uuid
from datetime import datetime, timedelta, timezone
from flask import jsonify

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
        "status": 202
        OR
        "error": "<error message>",
        "status": 400
    }
    """

    # validate stationId
    station_id = request_fields.get("stationId")
    if not station_id:
        return jsonify({"error": "Missing stationId request field"}), 400

    from src.station_service.station_service import StationService
    try:
        StationService(redis_client).get_station(station_id)
    except ValueError:
        return jsonify({"error": f"Invalid station ID: {station_id}"}), 400
    
    # validate and/or set default timebox parameters
    validation_error = validate_time_parameters(request_fields)
    if validation_error:
        return validation_error, 400

    # generate job id via uuidv5
    job_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 
        request_fields["stationId"] + 
        request_fields["startUtc"] + 
        request_fields["endUtc"])
    )

    # add job to redis
    job_key = f"job:{job_id}"
    expiry_minutes = int(os.getenv("FILE_EXPIRATION_TIME", "1440"))
    expiry_timestamp = (datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    redis_client.hset(job_key, mapping={
        "stationId": request_fields["stationId"],
        "startUtc": request_fields["startUtc"],
        "endUtc": request_fields["endUtc"],
        "status": "PENDING",
        "asset_expiry_timestamp": expiry_timestamp
    })

    # push job id to job queue
    redis_client.lpush("job_queue", job_id)

    # the cat's meow
    return jsonify({"job_id": job_id}), 202


def validate_time_parameters(request_fields: dict):
    """Validate the time parameters recieved via the request."""
    
    # Default timebox when not provided: look back over the last ~25 minutes, ending
    # 10 minutes ago. The 10-minute buffer ensures the algorithm's end time is always
    # fully in the past — if endUtc is too close to "now" the algorithm enters live
    # polling mode and runs indefinitely.
    now = datetime.now(timezone.utc)
    if not request_fields.get("startUtc") and not request_fields.get("endUtc"):
        request_fields["startUtc"] = (now - timedelta(minutes=35)).strftime("%Y-%m-%dT%H:%M:%SZ")
        request_fields["endUtc"] = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif not request_fields.get("startUtc") or not request_fields.get("endUtc"):
        return jsonify({"error": "Must provide both startUtc and endUtc, or neither"})

    # Parse and validate timebox
    try:
        start_utc = datetime.strptime(request_fields["startUtc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        end_utc = datetime.strptime(request_fields["endUtc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Expected ISO 8601: YYYY-MM-DDTHH:MM:SSZ"})

    # endUtc must be after startUtc
    if end_utc <= start_utc:
        return jsonify({"error": "endUtc must be after startUtc"})

    # duration must be between 15 minutes and MAX_JOB_DURATION (default is 180 minutes / 3 hours)
    max_duration = timedelta(minutes=int(os.getenv("MAX_JOB_DURATION", "180")))
    max_hours = max_duration.total_seconds() / 3600
    duration = end_utc - start_utc
    if duration < timedelta(minutes=15):
        return jsonify({"error": "Timebox duration must be at least 15 minutes"})
    if duration > max_duration:
        return jsonify({"error": f"Timebox duration must not exceed {max_hours:.0f} hours"})

    # endUtc must be at least 5 minutes in the past — the algorithm enters a live
    # polling loop if endUtc is too close to the current time, causing jobs to run
    # indefinitely instead of processing a closed historical window.
    if end_utc > now - timedelta(minutes=5):
        return jsonify({"error": "endUtc must be at least 5 minutes in the past"})

    return None
