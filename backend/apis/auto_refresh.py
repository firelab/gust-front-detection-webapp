
from flask import jsonify
from src.station_service.station_service import StationService


def _autorefresh_key(station_id: str) -> str:
    return f"autorefresh:{station_id}"


def enable_auto_refresh(redis_client, station_id: str):
    """
    Enable auto-refresh for a station.

    POST /apis/auto-refresh/<station_id>

    Validates the station ID, then creates or updates the autorefresh:<station_id>
    hash in Redis with refresh_enabled=true. The nfgda-service polling loop picks
    this up on its next cycle.

    Response shape:
        { "station_id": "<ID>", "refresh_enabled": true }
        OR
        { "error": "<message>" }, 400/404
    """
    station_id = station_id.upper().strip()

    try:
        StationService(redis_client).get_station(station_id)
    except ValueError:
        return jsonify({"error": f"Invalid station ID: {station_id}"}), 400

    key = _autorefresh_key(station_id)
    existing = redis_client.hgetall(key)

    # Preserve last_scan_time and current_job_id if the record already exists,
    # so we don't reset progress when re-enabling after a brief disable.
    redis_client.hset(key, mapping={
        "refresh_enabled": "true",
        "status": existing.get("status", "IDLE"),
        "last_scan_time": existing.get("last_scan_time", ""),
        "current_job_id": existing.get("current_job_id", ""),
    })

    return jsonify({"station_id": station_id, "refresh_enabled": True}), 200


def disable_auto_refresh(redis_client, station_id: str):
    """
    Disable auto-refresh for a station.

    DELETE /apis/auto-refresh/<station_id>

    Sets refresh_enabled=false. The polling loop will skip this station on its
    next cycle. The station:latest record and any completed job assets are left
    intact so previously-fetched frames remain accessible.

    Response shape:
        { "station_id": "<ID>", "refresh_enabled": false }
        OR
        { "error": "<message>" }, 400
    """
    station_id = station_id.upper().strip()

    try:
        StationService(redis_client).get_station(station_id)
    except ValueError:
        return jsonify({"error": f"Invalid station ID: {station_id}"}), 400

    key = _autorefresh_key(station_id)
    if not redis_client.exists(key):
        # Nothing to disable so return success anyway
        return jsonify({"station_id": station_id, "refresh_enabled": False}), 200

    redis_client.hset(key, "refresh_enabled", "false")
    return jsonify({"station_id": station_id, "refresh_enabled": False}), 200


def get_auto_refresh_status(redis_client, station_id: str):
    """
    Return the current auto-refresh state for a station.

    GET /apis/auto-refresh/<station_id>

    Response shape (when record exists):
        {
            "station_id": "<ID>",
            "refresh_enabled": true/false,
            "status": "IDLE" | "PROCESSING",
            "current_job_id": "<UUID>" | "",
            "last_scan_time": "<ISO 8601>" | ""
        }
        OR
        { "station_id": "<ID>", "refresh_enabled": false }  (no record yet)
        OR
        { "error": "<message>" }, 400
    """
    station_id = station_id.upper().strip()

    try:
        StationService(redis_client).get_station(station_id)
    except ValueError:
        return jsonify({"error": f"Invalid station ID: {station_id}"}), 400

    key = _autorefresh_key(station_id)
    fields = redis_client.hgetall(key)

    if not fields:
        return jsonify({"station_id": station_id, "refresh_enabled": False}), 200

    return jsonify({
        "station_id": station_id,
        "refresh_enabled": fields.get("refresh_enabled") == "true",
        "status": fields.get("status", "IDLE"),
        "current_job_id": fields.get("current_job_id", ""),
        "last_scan_time": fields.get("last_scan_time", ""),
    }), 200
