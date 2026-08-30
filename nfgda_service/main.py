import asyncio
import logging
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import nexradaws
import redis
from process_output import generate_geotiff_output

from nfgda_service import NfgdaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), db=int(os.getenv("REDIS_DB", "0")), decode_responses=True)
RedisHashMapping = Mapping[str, str | int]

# semaphor manages how many jobs can run at once
job_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "2")))


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def listen_for_jobs() -> None:
    """Poll Redis for jobs and dispatch them as async tasks.

    Only dequeues a job when the semaphore has capacity, so jobs
    remain in job_queue and their queue position stays accurate.
    Expired-job cleanup runs between poll cycles.
    """
    logger.info("NFGDA service started")
    logger.info("listening for jobs (max %d concurrent)", int(os.getenv("MAX_CONCURRENT_JOBS", "2")))

    loop = asyncio.get_running_loop()

    while True:
        # periodic cleanup of expired job assets and autorefresh stations
        await loop.run_in_executor(None, cleanup_expired_job_assets)
        await loop.run_in_executor(None, disable_autorefresh_on_expired_stations)

        # wait until there's capacity to process a job
        await job_semaphore.acquire()

        # dequeue with a timeout, run cleanup when idle
        result = await loop.run_in_executor(None, pop_queued_job)

        if result is None:
            job_semaphore.release()
            continue

        _, job_id = result
        logger.info("dequeued job %s", job_id)

        # run that job and release the semaphore when done
        asyncio.create_task(run_and_release_job(job_id))

async def process_job(job_id: str) -> None:
    """Process a single job after being acquired from the queue."""

    job_key = f"job:{job_id}"
    job_fields = redis_hgetall(job_key)
    out_dir = format_output_directory(job_id)

    logger.info("processing job %s", job_id)
    service = NfgdaService(redis_client, job_id, job_fields, out_dir)
    await service.run()

def process_geotiff_output(job_id: str) -> None:
    """Process the output of the NFGDA algorithm for a given job
    into a stack of GeoTIFFs for final display on the frontend."""
    
    if redis_hget(f"job:{job_id}", "status") == "FAILED":
        return

    logger.info("generating GeoTIFF series for job %s", job_id)
    result = generate_geotiff_output(job_id, redis_client)
        
    if result is not None:
        logger.error("failed to generate GeoTIFF series for job %s. Error message: %s", job_id, result)
        redis_hset_mapping(f"job:{job_id}", {"status": "FAILED", "error_message": result})
    else:
        num_frames = len(os.listdir(f"/processed_data/{job_id}")) - 1  # subtract 1 for manifest.json
        logger.info("successfully generated GeoTIFF series for job %s", job_id)
        redis_hset_mapping(f"job:{job_id}", {"status": "COMPLETED", "num_frames": num_frames})

async def run_and_release_job(job_id: str) -> None:
    """Run a job and release the semaphore when finished.
    
    Also updates station:latest:<station_id> so the cooldown check in the
    backend stays current regardless of whether the job was manual or auto-refresh.
    """
    try:
        await process_job(job_id)
        process_geotiff_output(job_id)
    finally:
        # they took my jerb!
        job_semaphore.release()

    # Mirror the final job status into station:latest so the backend cooldown
    # check reflects the actual outcome (COMPLETED or FAILED).
    sync_station_latest_from_job(job_id)


def sync_station_latest_from_job(job_id: str) -> None:
    """Update station:latest:<station_id> to match the completed job's final status.

    This keeps the cooldown record accurate for subsequent requests regardless
    of how the job was triggered (manual or auto-refresh).
    """
    job_key = f"job:{job_id}"
    job_fields = redis_hgetall(job_key)
    if not job_fields:
        return

    station_id = job_fields.get("stationId", "")
    if not station_id:
        return

    latest_key = f"station:latest:{station_id}"
    existing = redis_hgetall(latest_key)

    # Only update if this job is still the current latest for the station.
    # Auto-refresh may have already advanced to a newer job_id.
    if existing.get("job_id") != job_id:
        return

    final_status = job_fields.get("status", "FAILED")
    num_frames = job_fields.get("num_frames", "")
    redis_hset_mapping(latest_key, {
        "status": final_status,
        "num_frames": num_frames,
    })

    autorefresh_key = f"autorefresh:{station_id}"
    autorefresh_fields = redis_hgetall(autorefresh_key)
    if autorefresh_fields.get("current_job_id") == job_id:
        redis_hset_mapping(autorefresh_key, {
            "status": "IDLE",
            "last_updated_at": utc_now_str(),
        })

    logger.info("updated station:latest:%s status=%s num_frames=%s", station_id, final_status, num_frames)


async def auto_refresh_loop() -> None:
    """Continuously poll S3 NEXRAD buckets for stations with auto-refresh enabled.

    Every AUTO_REFRESH_POLL_INTERVAL seconds, this coroutine:
      1. Scans Redis for autorefresh:* keys where refresh_enabled=true.
      2. For each enabled station, queries the NEXRAD AWS bucket for new scans
         since the station's last_scan_time.
      3. If new scans exist, creates and runs a new NFGDA processing job and
         updates both the autorefresh:<station_id> and station:latest:<station_id>
         records in Redis.

    The timebox for each auto-refresh run is a 25-minute rolling window ending
    10 minutes before now, matching the default timebox used for manual requests.
    """
    poll_interval = int(os.getenv("AUTO_REFRESH_POLL_INTERVAL", "120"))
    logger.info("auto-refresh loop started (poll interval=%ds)", poll_interval)

    aws_int = nexradaws.NexradAwsInterface()
    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(poll_interval)

        # Collect all stations with auto-refresh enabled
        enabled_stations = []
        for key in redis_scan_iter(match="autorefresh:*"):
            refresh_enabled = redis_hget(key, "refresh_enabled")
            if refresh_enabled == "true":
                station_id = key.split(":", 1)[1]
                enabled_stations.append(station_id)

        if not enabled_stations:
            logger.debug("auto-refresh: no stations currently enabled")
            continue

        logger.info("auto-refresh: checking %d station(s): %s", len(enabled_stations), enabled_stations)

        for station_id in enabled_stations:
            try:
                await _auto_refresh_station(station_id, aws_int, loop)
            except Exception:
                logger.exception("auto-refresh: unhandled error for station %s", station_id)


async def _auto_refresh_station(station_id: str, aws_int, loop) -> None:
    """Run a single auto-refresh cycle for one station.

    Checks S3 for new scans since last_scan_time, and if found, runs a new
    NFGDA job using the 25-minute rolling timebox.
    """
    ar_key = f"autorefresh:{station_id}"
    ar_fields = redis_hgetall(ar_key)

    # Skip if already processing — avoid overlapping jobs for the same station
    if ar_fields.get("status") == "PROCESSING":
        logger.info("auto-refresh: %s already PROCESSING, skipping this cycle", station_id)
        return

    # Determine the scan window to query S3
    now = datetime.now(timezone.utc)
    last_scan_str = ar_fields.get("last_scan_time", "")
    if last_scan_str:
        try:
            scan_since = datetime.strptime(last_scan_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            scan_since = now - timedelta(minutes=35)
    else:
        scan_since = now - timedelta(minutes=35)

    # Query NEXRAD S3 for new scans since last_scan_time
    logger.info("auto-refresh: checking %s for new scans since %s", station_id, scan_since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    try:
        scans = await loop.run_in_executor(
            None,
            lambda: aws_int.get_avail_scans_in_range(scan_since, now, station_id)
        )
    except Exception:
        logger.exception("auto-refresh: failed to query S3 for station %s", station_id)
        return

    if not scans:
        logger.info("auto-refresh: no new scans found for %s", station_id)
        return

    # Advance last_scan_time to just past the newest scan so we don't reprocess it
    latest_scan_time = scans[-1].scan_time  # datetime with tzinfo from nexradaws
    new_last_scan_time = (latest_scan_time + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("auto-refresh: %d new scan(s) found for %s, latest=%s", len(scans), station_id, latest_scan_time)

    # Build the 25-minute rolling timebox in the format NFGDA expects: YYYY-MM-DDTHH:MM:SSZ
    # end_utc is 10 minutes before now to keep the algorithm in historical (not live) mode.
    end_utc = now - timedelta(minutes=10)
    start_utc = end_utc - timedelta(minutes=25)
    start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Generate a deterministic job ID from station + timebox
    job_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, station_id + start_utc_str + end_utc_str))
    job_key = f"job:{job_id}"

    # Check if this exact job was already run (same station + same timebox window)
    if redis_exists(job_key):
        logger.info("auto-refresh: job %s already exists for %s, skipping", job_id, station_id)
        return

    # Register the job in Redis
    expiry_minutes = int(os.getenv("FILE_EXPIRATION_TIME", "1440"))
    expiry_timestamp = (now + timedelta(minutes=expiry_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    redis_hset_mapping(job_key, {
        "stationId": station_id,
        "startUtc": start_utc_str,
        "endUtc": end_utc_str,
        "status": "PENDING",
        "asset_expiry_timestamp": expiry_timestamp,
    })

    # Mark station as PROCESSING and update station:latest
    redis_hset_mapping(ar_key, {
        "status": "PROCESSING",
        "current_job_id": job_id,
        "last_scan_time": new_last_scan_time,
        "last_updated_at": utc_now_str(),
    })
    _write_station_latest_sync(station_id, job_id, source="auto_refresh")

    logger.info("auto-refresh: starting job %s for station %s [%s -> %s]",
                job_id, station_id, start_utc_str, end_utc_str)

    # Acquire the semaphore and run the job directly (not via job_queue)
    await job_semaphore.acquire()
    try:
        await process_job(job_id)
        process_geotiff_output(job_id)
    finally:
        job_semaphore.release()

    # Reflect the final outcome back to both records
    final_status = redis_hget(job_key, "status") or "FAILED"
    num_frames = redis_hget(job_key, "num_frames") or ""

    redis_hset_mapping(ar_key, {
        "status": "IDLE",
        "last_updated_at": utc_now_str(),
    })
    redis_hset_mapping(f"station:latest:{station_id}", {
        "status": final_status,
        "num_frames": num_frames,
    })
    logger.info("auto-refresh: job %s for %s finished with status=%s", job_id, station_id, final_status)


def _write_station_latest_sync(station_id: str, job_id: str, source: str) -> None:
    """Write station:latest:<station_id> synchronously (called from async context)."""
    now = utc_now_str()
    redis_hset_mapping(f"station:latest:{station_id}", {
        "job_id": job_id,
        "created_at": now,
        "status": "PENDING",
        "source": source,
        "num_frames": "",
    })


def format_output_directory(job_id: str) -> str:
    """Format the output directory for a job."""
    out_dir = f"/nfgda_output/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def disable_autorefresh_on_expired_stations():
    """Resets station redis values to disable auto-refresh after expiry period"""
    ar_stations = list(redis_scan_iter(match="autorefresh:*"))
    if not ar_stations:
        return

    now = datetime.now(timezone.utc)

    for station in ar_stations:
        expiry = redis_hget(station, "auto_refresh_expiry")
        if not expiry:
            continue
        try:
            expiry = datetime.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            logger.warning(
                "skipping %s due to malformed expiry timestamp: %s", station, expiry
            )
            continue

        if now < expiry:
            continue

        redis_hset_field(station, "refresh_enabled", "false")
        logger.info("disabled auto-refresh on station %s", station.split(":", 1)[1])


def cleanup_expired_job_assets() -> None:
    """Scan Redis for job records whose asset_expiry_timestamp has passed.

    For each expired job:
      1. Delete its output directories (/nfgda_output/<id> and /processed_data/<id>).
      2. Remove the job hash from Redis.
    """
    now = datetime.now(timezone.utc)

    for key in redis_scan_iter(match="job:*"):
        expiry_str = redis_hget(key, "asset_expiry_timestamp")
        if not expiry_str:
            continue

        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("skipping %s due to malformed expiry timestamp: %s", key, expiry_str)
            continue

        if now < expiry:
            continue

        # extract job_id from the key ("job:<id>")
        job_id = key.split(":", 1)[1]
        logger.info("cleaning up expired job %s (expired %s)", job_id, expiry_str)

        # remove output directories
        for base in ("/nfgda_output", "/processed_data"):
            path = os.path.join(base, job_id)
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info("- removed %s", path)

        # remove the job record from redis
        redis_delete(key)

def sync_redis_client() -> Any:
    """Return the sync Redis client with broad redis-py annotations erased."""
    return cast(Any, redis_client)


def pop_queued_job() -> tuple[str, str] | None:
    """Pop one queued job from Redis."""
    return cast(
        tuple[str, str] | None,
        sync_redis_client().brpop(["job_queue"], timeout=10),
    )


def redis_hgetall(key: str) -> dict[str, str]:
    """Read a Redis hash as decoded string fields."""
    return cast(dict[str, str], sync_redis_client().hgetall(key))


def redis_hget(key: str, field: str) -> str | None:
    """Read a single decoded Redis hash field."""
    return cast(str | None, sync_redis_client().hget(key, field))


def redis_hset_mapping(key: str, mapping: RedisHashMapping) -> None:
    """Write multiple Redis hash fields."""
    sync_redis_client().hset(key, mapping=mapping)


def redis_hset_field(key: str, field: str, value: str | int) -> None:
    """Write one Redis hash field."""
    sync_redis_client().hset(key, field, value)


def redis_scan_iter(match: str) -> Iterator[str]:
    """Scan Redis keys with decoded string results."""
    return cast(Iterator[str], sync_redis_client().scan_iter(match=match))


def redis_exists(key: str) -> bool:
    """Return whether a Redis key exists."""
    return bool(sync_redis_client().exists(key))


def redis_delete(key: str) -> None:
    """Delete a Redis key."""
    sync_redis_client().delete(key)


def main():
    asyncio.run(_run_all())

async def _run_all():
    """Run the job listener and auto-refresh polling loop concurrently."""
    await asyncio.gather(
        listen_for_jobs(),
        auto_refresh_loop(),
    )

if __name__ == "__main__":
    main()
