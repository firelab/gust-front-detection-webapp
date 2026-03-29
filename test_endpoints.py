"""
End-to-end integration tests for the backend API endpoints.

Requires the full Docker Compose stack to be running:
    docker compose up -d --build

Run with:
    pytest test_endpoints.py -v

Test order matters — later tests depend on state created by earlier ones
(e.g. a job must be submitted before its status can be polled). The
pytest-ordering plugin or explicit state sharing via module-level variables
is used to enforce this.

Endpoint coverage:
    1. GET  /apis/stations                       → test_get_stations
    2. POST /apis/run                            → test_submit_job
    3. POST /apis/run (validation errors)        → test_submit_job_*
    4. GET  /apis/status?job_id=<id>             → test_poll_job_status
    5. GET  /apis/jobs/<job_id>/frames/<index>   → test_fetch_frames
"""

import time
import pytest
import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8001"
POLL_INTERVAL_SECONDS = 5
MAX_POLLS = 120

# ── shared state across ordered tests ────────────────────────────────────────
_state = {
    "station_id": None,
    "job_id": None,
    "num_frames": 0,
}


def _fmt_utc(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the API expects."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /apis/stations
# ─────────────────────────────────────────────────────────────────────────────
class TestStationsEndpoint:
    """Tests for GET /apis/stations."""

    def test_get_stations_returns_200(self):
        resp = requests.get(f"{BASE_URL}/apis/stations")
        assert resp.status_code == 200

    def test_get_stations_returns_geojson(self):
        resp = requests.get(f"{BASE_URL}/apis/stations")
        body = resp.json()
        assert "features" in body, "response should be a GeoJSON FeatureCollection"
        assert isinstance(body["features"], list)
        assert len(body["features"]) > 0, "station list should not be empty"

    def test_station_feature_has_expected_properties(self):
        resp = requests.get(f"{BASE_URL}/apis/stations")
        feature = resp.json()["features"][0]
        assert "properties" in feature
        assert "station_id" in feature["properties"]
        assert "geometry" in feature

        # stash a station for use in later tests
        _state["station_id"] = feature["properties"]["station_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. POST /apis/run — validation error cases
# ─────────────────────────────────────────────────────────────────────────────
class TestRunEndpointValidation:
    """Tests for POST /apis/run input validation (no job should be created)."""

    def test_missing_station_id_returns_400(self):
        now = datetime.now(timezone.utc)
        payload = {
            "startUtc": _fmt_utc(now - timedelta(minutes=45)),
            "endUtc": _fmt_utc(now - timedelta(minutes=25)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400
        assert "stationId" in resp.json().get("error", "").lower() or "stationid" in resp.json().get("error", "").lower()

    def test_invalid_station_id_returns_400(self):
        now = datetime.now(timezone.utc)
        payload = {
            "stationId": "ZZZZ_NOT_REAL",
            "startUtc": _fmt_utc(now - timedelta(minutes=45)),
            "endUtc": _fmt_utc(now - timedelta(minutes=25)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400

    def test_end_before_start_returns_400(self):
        now = datetime.now(timezone.utc)
        payload = {
            "stationId": _state["station_id"] or "KABX",
            "startUtc": _fmt_utc(now - timedelta(minutes=25)),
            "endUtc": _fmt_utc(now - timedelta(minutes=45)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400

    def test_future_end_time_returns_400(self):
        now = datetime.now(timezone.utc)
        payload = {
            "stationId": _state["station_id"] or "KABX",
            "startUtc": _fmt_utc(now - timedelta(minutes=30)),
            "endUtc": _fmt_utc(now + timedelta(hours=1)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400

    def test_end_time_too_recent_returns_400(self):
        """endUtc within 5 minutes of now triggers live-polling mode in the algorithm — must be rejected."""
        now = datetime.now(timezone.utc)
        payload = {
            "stationId": _state["station_id"] or "KABX",
            "startUtc": _fmt_utc(now - timedelta(minutes=30)),
            "endUtc": _fmt_utc(now - timedelta(minutes=2)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400

    def test_duration_too_short_returns_400(self):
        now = datetime.now(timezone.utc)
        payload = {
            "stationId": _state["station_id"] or "KABX",
            "startUtc": _fmt_utc(now - timedelta(minutes=10)),
            "endUtc": _fmt_utc(now - timedelta(minutes=8)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST /apis/run — happy path (submit a real job)
# ─────────────────────────────────────────────────────────────────────────────
class TestRunEndpointHappyPath:
    """Submit a valid job and store the job_id for downstream tests."""

    def test_submit_job_returns_202(self):
        assert _state["station_id"], "station_id must be populated by TestStationsEndpoint"

        now = datetime.now(timezone.utc)
        payload = {
            "stationId": _state["station_id"],
            "startUtc": _fmt_utc(now - timedelta(minutes=45)),
            "endUtc": _fmt_utc(now - timedelta(minutes=25)),
        }
        resp = requests.post(f"{BASE_URL}/apis/run", json=payload)
        assert resp.status_code == 202

        body = resp.json()
        assert "job_id" in body
        _state["job_id"] = body["job_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /apis/status
# ─────────────────────────────────────────────────────────────────────────────
class TestStatusEndpoint:
    """Tests for GET /apis/status."""

    def test_missing_job_id_returns_400(self):
        resp = requests.get(f"{BASE_URL}/apis/status")
        assert resp.status_code == 400

    def test_unknown_job_id_returns_404(self):
        resp = requests.get(f"{BASE_URL}/apis/status", params={"job_id": "nonexistent-id"})
        assert resp.status_code == 404

    def test_valid_job_returns_status(self):
        assert _state["job_id"], "job_id must be populated by TestRunEndpointHappyPath"
        resp = requests.get(f"{BASE_URL}/apis/status", params={"job_id": _state["job_id"]})
        assert resp.status_code == 200

        body = resp.json()
        assert "job_id" in body
        assert "status" in body
        assert body["status"] in {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}

    @pytest.mark.slow
    def test_poll_until_terminal(self):
        """Poll the job until it reaches COMPLETED or FAILED (may take minutes)."""
        assert _state["job_id"], "job_id must be populated by TestRunEndpointHappyPath"

        terminal = {"COMPLETED", "FAILED"}
        for _ in range(MAX_POLLS):
            time.sleep(POLL_INTERVAL_SECONDS)
            resp = requests.get(f"{BASE_URL}/apis/status", params={"job_id": _state["job_id"]})
            body = resp.json()
            if body.get("status") in terminal:
                _state["num_frames"] = int(body.get("num_frames", 0))
                return  # success — reached terminal state

        pytest.fail(f"Job {_state['job_id']} did not reach a terminal state after {MAX_POLLS} polls")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /apis/jobs/<job_id>/frames/<index>
# ─────────────────────────────────────────────────────────────────────────────
class TestFramesEndpoint:
    """Tests for GET /apis/jobs/<job_id>/frames/<index>."""

    def test_nonexistent_job_returns_404(self):
        resp = requests.get(f"{BASE_URL}/apis/jobs/fake-job-id/frames/0")
        assert resp.status_code == 404

    @pytest.mark.slow
    def test_fetch_all_frames(self):
        """After job completion, every advertised frame should be retrievable."""
        if _state["num_frames"] == 0:
            pytest.skip("No frames produced (job may have failed or not run yet)")

        job_id = _state["job_id"]
        for i in range(_state["num_frames"]):
            resp = requests.get(f"{BASE_URL}/apis/jobs/{job_id}/frames/{i}")
            assert resp.status_code == 200, f"frame {i} returned {resp.status_code}"
            assert len(resp.content) > 0, f"frame {i} was empty"
            assert resp.headers.get("Content-Type") == "image/tiff"

    @pytest.mark.slow
    def test_out_of_range_frame_returns_404(self):
        """Requesting a frame index beyond what the job produced should 404."""
        if _state["num_frames"] == 0:
            pytest.skip("No frames produced (job may have failed or not run yet)")

        resp = requests.get(
            f"{BASE_URL}/apis/jobs/{_state['job_id']}/frames/{_state['num_frames'] + 100}"
        )
        assert resp.status_code == 404
