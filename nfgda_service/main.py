import os
import shutil
import asyncio
import redis
import logging
from datetime import datetime, timezone, timedelta
from nfgda_service import NfgdaService
from process_output import generate_geotiff_output


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", "6379")), db=int(os.getenv("REDIS_DB", "0")), decode_responses=True)

# semaphor manages how many jobs can run at once
job_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "2")))

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
        # periodic cleanup of expired job assets
        await loop.run_in_executor(None, cleanup_expired_jobs)

        # wait until there's capacity to process a job
        await job_semaphore.acquire()

        # dequeue with a timeout, run cleanup when idle
        result = await loop.run_in_executor(
            None, lambda: redis_client.brpop("job_queue", timeout=10)
        )

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
    job_fields = redis_client.hgetall(job_key)
    out_dir = format_output_directory(job_id)

    logger.info("processing job %s", job_id)
    service = NfgdaService(redis_client, job_id, job_fields, out_dir)
    await service.run()

def process_geotiff_output(job_id: str) -> None:
    """Process the output of the NFGDA algorithm for a given job
    into a stack of GeoTIFFs for final display on the frontend."""

    if redis_client.hget(f"job:{job_id}", "status") == "FAILED":
        return

    logger.info("generating GeoTIFF series for job %s", job_id)
    result = generate_geotiff_output(job_id, redis_client)

    if result is not None:
        logger.error("failed to generate GeoTIFF series for job %s. Error message: %s", job_id, result)
        redis_client.hset(f"job:{job_id}", mapping={"status": "FAILED", "error_message": result})
    else:
        logger.info("successfully generated GeoTIFF series for job %s", job_id)
        redis_client.hset(f"job:{job_id}", mapping={"status": "COMPLETED", "num_frames": len(os.listdir(f"/processed_data/{job_id}")) - 1}) # subtract 1 for the timestamps.json file



def _wipe_job_output(job_id: str) -> None:
    """Helper function to wipe the output of a job for rerunning if frames are <= 2"""
    for base in ("/nfgda_output", "/processed_data"):
        path = os.path.join(base, job_id)
        if os.path.isdir(path):
            shutil.rmtree(path)

def _expand_start_utc(start_str: str, end_str: str) -> str:
    """Helper to compute the expanded time window for a job"""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = datetime.strptime(start_str, fmt).replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, fmt).replace(tzinfo=timezone.utc)
    duration = end - start
    max_duration = timedelta(minutes=int(os.getenv("MAX_JOB_DURATION", "180")))
    new_start = max(start - duration, end - max_duration)
    return new_start.strftime(fmt)

async def run_and_release_job(job_id: str) -> None:
    """Run a job and release the semaphore when finished."""
    try:
        await process_job(job_id)
        process_geotiff_output(job_id)
        # Silently rerun if there is less than 2 frames in a completed job
        # I do not know how this will integrate with the new feature of forecasting
        num_frames = int(redis_client.hget(f"job:{job_id}", "num_frames") or 0)
        if num_frames <= 2:
            logger.warning("retrying the job with a larger time window")
            job_fields = redis_client.hgetall(f"job:{job_id}")
            new_start = _expand_start_utc(job_fields["startUtc"], job_fields["endUtc"])
            redis_client.hset(f"job:{job_id}", mapping={"startUtc": new_start, "status": "PENDING"})
            _wipe_job_output(job_id)
            await process_job(job_id)
            process_geotiff_output(job_id)


    finally:
        # they took my jerb!
        job_semaphore.release()

def format_output_directory(job_id: str) -> str:
    """Format the output directory for a job."""
    out_dir = f"/nfgda_output/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def cleanup_expired_jobs() -> None:
    """Scan Redis for job records whose asset_expiry_timestamp has passed.

    For each expired job:
      1. Delete its output directories (/nfgda_output/<id> and /processed_data/<id>).
      2. Remove the job hash from Redis.
    """
    now = datetime.now(timezone.utc)

    for key in redis_client.scan_iter(match="job:*"):
        expiry_str = redis_client.hget(key, "asset_expiry_timestamp")
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
        redis_client.delete(key)

def main():
    asyncio.run(listen_for_jobs())

if __name__ == "__main__":
    main()
