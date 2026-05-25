import os
import shutil
import asyncio
import redis
import logging
from redis.exceptions import RedisError
from datetime import datetime, timezone
from nfgda_service import NfgdaService
from process_output import generate_geotiff_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

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
    await wait_for_redis(loop)

    while True:
        # periodic cleanup of expired job assets
        try:
            await loop.run_in_executor(None, cleanup_expired_jobs)
        except RedisError as exc:
            logger.warning("Redis unavailable during cleanup: %s", exc)
            await asyncio.sleep(5)
            continue

        # wait until there's capacity to process a job
        await job_semaphore.acquire()

        # dequeue with a timeout, run cleanup when idle
        try:
            result = await loop.run_in_executor(
                None, lambda: redis_client.brpop("job_queue", timeout=10)
            )
        except RedisError as exc:
            job_semaphore.release()
            logger.warning("Redis unavailable while waiting for jobs: %s", exc)
            await asyncio.sleep(5)
            continue

        if result is None:
            job_semaphore.release()
            continue

        _, job_id = result
        logger.info("dequeued job %s", job_id)

        # run that job and release the semaphore when done
        asyncio.create_task(run_and_release_job(job_id))


async def wait_for_redis(loop: asyncio.AbstractEventLoop) -> None:
    """Wait until Redis accepts commands before starting the worker loop."""
    while True:
        try:
            await loop.run_in_executor(None, redis_client.ping)
            redis_kwargs = redis_client.connection_pool.connection_kwargs
            logger.info(
                "connected to Redis at %s:%s",
                redis_kwargs.get("host"),
                redis_kwargs.get("port"),
            )
            return
        except RedisError as exc:
            logger.warning("waiting for Redis: %s", exc)
            await asyncio.sleep(5)

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

async def run_and_release_job(job_id: str) -> None:
    """Run a job and release the semaphore when finished."""
    try:
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
