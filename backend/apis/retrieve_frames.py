"""
Frame Data API: returns a single rendered GeoTIFF frame for a completed job.
"""

import os
import json
from flask import send_file, abort, make_response


def get_frame(job_id: str, index: int):
    """Return a single GeoTIFF frame file for the given job and frame index.
        Includes an X-Frame-Timestamp header of the radar observation time.
    """
    job_dir = "/processed_data/" + job_id
    if not os.path.exists(job_dir):
        abort(404, description="Job not found")

    frame_path = job_dir + f"/frame_{index}.tif"
    if not os.path.exists(frame_path):
        abort(404, description="Frame not found")

    response = make_response(
        send_file(
            frame_path,
            mimetype="image/tiff",
            as_attachment=False,
        )
    )

    # attach the radar observation timestamp if available
    timestamp = get_frame_timestamp(job_dir, index)
    if timestamp is not None:
        response.headers["X-Frame-Timestamp"] = timestamp

    return response


def get_frame_timestamp(job_dir: str, index: int) -> str | None:
    """Read the per-frame observation timestamp from the job's manifest file.
    Returns an ISO 8601 UTC string (e.g. "2024-07-07T01:22:24Z") or
    None if the manifest is missing or the frame index has no entry.
    """
    manifest_path = os.path.join(job_dir, "timestamps.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest.get(str(index))
    except Exception:
        return None
