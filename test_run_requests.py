#!/usr/bin/env python3
"""
Integration test script for the /APIs/run and /APIs/status endpoints.

Fetches the station list from /APIs/stations, picks 3 random stations,
sends run requests with 10-minute windows from the last hour (one per station),
plus a 4th request with a future endUtc that should be rejected.
Then polls until all valid jobs report COMPLETED (or FAILED).
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


def fetch_random_stations(count: int = 3) -> list[str]:
    """GET /APIs/stations and return `count` random station IDs."""
    print(f"{'='*60}")
    print(f"GET /APIs/stations  (selecting {count} random stations)")

    resp = requests.get(f"{BASE_URL}/APIs/stations")
    print(f"  status:  {resp.status_code}")

    if resp.status_code != 200:
        print(f"  Failed to fetch stations: {resp.text[:200]}")
        raise SystemExit(1)

    data = resp.json()
    features = data.get("features", [])
    print(f"  total stations available: {len(features)}")

    selected = random.sample(features, min(count, len(features)))
    station_ids = [f["properties"]["station_id"] for f in selected]
    print(f"  selected: {station_ids}")
    return station_ids


def send_run_request(label: str, station_id: str, start: datetime, end: datetime) -> str | None:
    """POST a run request and return the job_id, or None on failure."""
    payload = {
        "stationId": station_id,
        "startUtc": fmt_utc(start),
        "endUtc": fmt_utc(end),
    }
    print(f"\n{'='*60}")
    print(f"[{label}] POST /APIs/run")
    print(f"  payload: {json.dumps(payload, indent=2)}")

    resp = requests.post(f"{BASE_URL}/APIs/run", json=payload)
    body = resp.json()
    print(f"  status:  {resp.status_code}")
    print(f"  response: {json.dumps(body, indent=2)}")

    if resp.status_code == 202:
        return body.get("job_id")
    return None


def check_status(job_id: str, label: str = "") -> dict:
    """GET /APIs/status?job_id=<job_id> and return the parsed response body."""
    tag = f"[{label}] " if label else ""
    print(f"\n  {tag}GET /APIs/status?job_id={job_id}")

    resp = requests.get(f"{BASE_URL}/APIs/status", params={"job_id": job_id})
    print(f"  {tag}status:  {resp.status_code}")
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        print(f"  {tag}response: (non-JSON) {resp.text[:200]}")
        return {"status": "ERROR"}
    print(f"  {tag}response: {json.dumps(body, indent=2)}")
    return body


def main():
    now = datetime.now(timezone.utc)
    print(f"Baseline UTC time: {fmt_utc(now)}")

    # ── Fetch 3 random station IDs from the stations API ─────────────
    station_ids = fetch_random_stations(3)

    # ── Define 3 valid jobs: 10-minute windows within the last hour ──
    # ── Plus 1 invalid job with endUtc in the future ─────────────────
    jobs_config = [
        ("Job 1", station_ids[0], now - timedelta(minutes=90), now - timedelta(minutes=75)),
        ("Job 2", station_ids[1], now - timedelta(minutes=75), now - timedelta(minutes=60)),
        ("Job 3", station_ids[2], now - timedelta(minutes=60), now - timedelta(minutes=45)),
        ("Job 4", station_ids[0], now - timedelta(minutes=10),  now + timedelta(minutes=25)),  # future — expect 400
    ]

    job_ids: list[str | None] = []

    # ── Send all 4 requests ──────────────────────────────────────────
    for i, (label, sid, start, end) in enumerate(jobs_config):
        job_id = send_run_request(label, sid, start, end)
        job_ids.append(job_id)

        if i < len(jobs_config) - 1:
            time.sleep(2)

    # ── Summarize submission results ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Submission summary:")
    for i, (label, sid, _, _) in enumerate(jobs_config):
        status = f"job_id={job_ids[i]}" if job_ids[i] else "REJECTED"
        print(f"  {label} ({sid}): {status}")

    # ── Filter to valid jobs only for polling ────────────────────────
    valid_jobs = [
        (i, jid) for i, jid in enumerate(job_ids) if jid is not None
    ]

    if not valid_jobs:
        print("\nNo valid jobs to poll. Done.")
        return

    # ── Poll valid jobs until all reach a terminal state ─────────────
    print(f"\n{'='*60}")
    print(f"Polling {len(valid_jobs)} valid job(s) until completion...")
    print(f"{'='*60}")

    terminal_states = {"COMPLETED", "FAILED"}
    max_polls = 120
    poll_count = 0

    while poll_count < max_polls:
        poll_count += 1
        time.sleep(5)

        print(f"\n--- Poll #{poll_count} ---")
        all_done = True
        for i, jid in valid_jobs:
            label = jobs_config[i][0]
            body = check_status(jid, label=label)
            status = body.get("status", "UNKNOWN")
            if status not in terminal_states:
                all_done = False

        if all_done:
            print(f"\n{'='*60}")
            print("All jobs have reached a terminal state. Done!")
            print(f"{'='*60}")
            return

    print(f"\n{'='*60}")
    print(f"Gave up after {max_polls} polls. Some jobs may still be running.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

