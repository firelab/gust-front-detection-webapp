"""
Frame Data API: returns a single rendered GeoTIFF frame for a completed job.
"""

import os
from flask import send_file, abort


def get_frame(job_id: str, index: int):
    """Return a single GeoTIFF frame file for the given job and frame index."""
    job_dir = "/processed_data/" + job_id
    if not os.path.exists(job_dir):
        abort(404, description="Job not found")

    frame_path = job_dir + f"/frame_{index}.tif"
    if not os.path.exists(frame_path):
        abort(404, description="Frame not found")

    return send_file(
        frame_path,
        mimetype="image/tiff",
        as_attachment=False
    )
