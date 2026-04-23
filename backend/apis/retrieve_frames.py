
import os
import json
from flask import send_file, abort, make_response


def get_frame(job_id: str, index: int):
    """Return a single GeoTIFF frame file for the given job and frame index.
        Includes an X-Frame-Timestamp header of the radar observation time 
        and a boolean "X-Frame-Is-Forecast" header.
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

    # attach per-frame metadata from manifest
    entry = get_frame_manifest_entry(job_dir, index)
    if entry is not None:
        timestamp = entry.get("timestamp")
        if timestamp is not None:
            response.headers["X-Frame-Timestamp"] = timestamp
        is_forecast = entry.get("is_forecast", False)
        response.headers["X-Frame-Is-Forecast"] = "true" if is_forecast else "false"

    return response


def get_frame_manifest_entry(job_dir: str, index: int) -> dict | None:
    """Read the per-frame metadata entry from the job's manifest.json.
    Returns a dict with { "timestamp": "...", "is_forecast": bool },
    or None if the manifest is missing or the frame index has no entry.
    """
    manifest_path = os.path.join(job_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest.get(str(index))
        except Exception:
            return None

    return None
