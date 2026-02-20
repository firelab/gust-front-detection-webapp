"""
Algorithm Runner API – accepts a station ID and time frame, kicks off an
NFGDA processing job, and returns the new job ID.
"""
import uuid
from flask import jsonify, request
from src.nfgda_service.nfgda_service import NfgdaService


def start_algorithm_run(service: NfgdaService, request_fields: dict):
    """
    Expected JSON body via request_fields:
    {
        "stationId": "KABX",
        "startUtc": "2024-07-07T01:22:24Z",
        "endUtc":   "2024-07-07T03:48:02Z",
        "options":  {}
    }

    Response shape:
    {
        "job_id": "<jobId>",
        "job_status": "PENDING",
        "status": 200
    }
    """
    if not request_fields:
        return jsonify({"error": "Missing request fields"}), 500

    #TODO: validate timebox and station_id via redis cache

    job_id = uuid.uuidv5(uuid.NAMESPACE_DNS, request_fields["stationId"] + request_fields["startUtc"] + request_fields["endUtc"])
    new_job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        run_request=RunRequest(
            station_id=request_fields["stationId"],
            start_utc=request_fields["startUtc"],
            end_utc=request_fields["endUtc"],
        ),
    )

    status = service.add_job_to_queue(new_job) # 'PENDING' if job in queue, 'RUNNING' if queue is empty, 'FAILED' if queue is full
    new_job.status = status
    returncode = 500 if status == JobStatus.FAILED else 200
    return job_id, new_job.status, returncode
    

    
