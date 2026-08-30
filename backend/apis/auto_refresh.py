from datetime import datetime, timedelta, timezone

from flask import jsonify
from src.station_service.station_service import StationService


def _autorefresh_key(station_id: str) -> str:
    return f"autorefresh:{station_id}"


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_auto_refresh_status(redis_client, key: str, fields: dict):
    station_id = key.split(":", 1)[1]
    current_job_id = fields.get("current_job_id", "")
    current_job_status = ""

    if current_job_id:
        current_job_status = redis_client.hget(
            f"job:{current_job_id}", "status"
        ) or ""

    return {
        "station_id": station_id,
        "refresh_enabled": fields.get("refresh_enabled") == "true",
        "status": fields.get("status", "IDLE"),
        "current_job_id": current_job_id,
        "current_job_status": current_job_status,
        "has_active_job": current_job_status in {"PENDING", "PROCESSING"},
        "last_scan_time": fields.get("last_scan_time", ""),
        "last_updated_at": fields.get("last_updated_at", ""),
        "auto_refresh_expiry": fields.get("auto_refresh_expiry", ""),
    }


def list_auto_refresh_statuses(redis_client):
    """Return all stations that currently have auto-refresh enabled."""
    stations = []

    for key in redis_client.scan_iter(match="autorefresh:*"):
        fields = redis_client.hgetall(key)
        if fields.get("refresh_enabled") != "true":
            continue

        stations.append(_format_auto_refresh_status(redis_client, key, fields))

    stations.sort(key=lambda station: station["station_id"])
    return jsonify({"stations": stations}), 200


def enable_auto_refresh(
    redis_client,
    station_id: str,
    duration_minutes: int,
    current_job_id: str = "",
):
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
    last_updated_at = _utc_now_str()
    autorefresh_expiry = (
        datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        StationService(redis_client).get_station(station_id)
    except ValueError:
        return jsonify({"error": f"Invalid station ID: {station_id}"}), 400

    key = _autorefresh_key(station_id)
    existing = redis_client.hgetall(key)
    current_job_status = ""

    if current_job_id:
        job_key = f"job:{current_job_id}"
        job_station_id = redis_client.hget(job_key, "stationId")
        if not job_station_id:
            return jsonify({"error": f"Invalid job ID: {current_job_id}"}), 400
        if job_station_id != station_id:
            return jsonify({
                "error": f"Job {current_job_id} does not belong to {station_id}"
            }), 400
        current_job_status = redis_client.hget(job_key, "status") or ""

    stored_job_id = current_job_id or existing.get("current_job_id", "")
    stored_status = existing.get("status", "IDLE")
    if stored_job_id and not current_job_status:
        current_job_status = redis_client.hget(f"job:{stored_job_id}", "status") or ""
    if current_job_status in {"PENDING", "PROCESSING"}:
        stored_status = current_job_status

    # Preserve scan progress when re-enabling after a brief disable.
    redis_client.hset(key, mapping={
        "refresh_enabled": "true",
        "status": stored_status,
        "last_scan_time": existing.get("last_scan_time", ""),
        "current_job_id": stored_job_id,
        "last_updated_at": last_updated_at,
        "auto_refresh_expiry": autorefresh_expiry,
    })

    return jsonify({
        "station_id": station_id,
        "refresh_enabled": True,
        "current_job_id": stored_job_id,
        "current_job_status": current_job_status,
        "has_active_job": current_job_status in {"PENDING", "PROCESSING"},
        "last_updated_at": last_updated_at,
        "auto_refresh_expiry": autorefresh_expiry,
    }), 200


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

    return jsonify(_format_auto_refresh_status(redis_client, key, fields)), 200
