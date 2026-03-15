#!/usr/bin/env python3
"""
End-to-end integration test:

1. GET /APIs/stations  → fetch the station list
2. Randomly select one station
3. POST /APIs/run      → submit a job for that station
4. Poll GET /APIs/status until the job reaches COMPLETED (or FAILED)
5. GET /api/jobs/<job_id>/frames/<index> → fetch every produced frame
"""

import json
import random
import time
import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8001"


def fmt_utc(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the API expects."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    now = datetime.now(timezone.utc)
    print(f"Baseline UTC time: {fmt_utc(now)}")

    # ── 1. Fetch station list ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1: GET /APIs/stations")
    resp = requests.get(f"{BASE_URL}/APIs/stations")
    print(f"  status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"  FAILED — {resp.text[:200]}")
        return

    features = resp.json().get("features", [])
    print(f"  total stations: {len(features)}")

    # ── 2. Pick one random station ───────────────────────────────────
    station = random.choice(features)
    station_id = station["properties"]["station_id"]
    print(f"\n{'='*60}")
    print(f"Step 2: Selected station → {station_id}")

    # ── 3. Submit a job ──────────────────────────────────────────────
    start = now - timedelta(minutes=45)
    end = now - timedelta(minutes=25)
    payload = {
        "stationId": station_id,
        "startUtc": fmt_utc(start),
        "endUtc": fmt_utc(end),
    }

    print(f"\n{'='*60}")
    print("Step 3: POST /APIs/run")
    print(f"  payload: {json.dumps(payload, indent=2)}")

    resp = requests.post(f"{BASE_URL}/APIs/run", json=payload)
    body = resp.json()
    print(f"  status:   {resp.status_code}")
    print(f"  response: {json.dumps(body, indent=2)}")

    if resp.status_code != 202:
        print("  Job was not accepted. Stopping.")
        return

    job_id = body["job_id"]

    # ── 4. Poll until terminal state ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: Polling job {job_id} until completion...")

    terminal_states = {"COMPLETED", "FAILED"}
    max_polls = 120
    status_body = {}

    for poll in range(1, max_polls + 1):
        time.sleep(5)
        print(f"\n  --- Poll #{poll} ---")

        resp = requests.get(f"{BASE_URL}/APIs/status", params={"job_id": job_id})
        try:
            status_body = resp.json()
        except requests.exceptions.JSONDecodeError:
            print(f"  (non-JSON) {resp.text[:200]}")
            continue

        status = status_body.get("status", "UNKNOWN")
        print(f"  status: {status}")
        print(f"  body:   {json.dumps(status_body, indent=2)}")

        if status in terminal_states:
            break
    else:
        print(f"\n  Gave up after {max_polls} polls.")
        return

    if status_body.get("status") != "COMPLETED":
        print(f"\n  Job ended with status: {status_body.get('status')}. Skipping frame fetch.")
        return

    # ── 5. Fetch frames ──────────────────────────────────────────────
    num_frames = int(status_body.get("num_frames", 0))
    print(f"\n{'='*60}")
    print(f"Step 5: Fetching {num_frames} frame(s) for job {job_id}")

    for index in range(num_frames):
        url = f"{BASE_URL}/api/jobs/{job_id}/frames/{index}"
        print(f"\n  GET {url}")
        resp = requests.get(url)
        if resp.status_code == 200:
            print(f"    ✓ frame {index}: {resp.status_code} ({len(resp.content)} bytes)")
        else:
            print(f"    ✗ frame {index}: {resp.status_code} — {resp.text[:200]}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
