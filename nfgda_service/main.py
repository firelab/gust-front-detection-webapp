import os
import asyncio
import redis
import logging
from nfgda_service import NfgdaService
from process_output import generate_geotiff_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", "6379")), db=int(os.getenv("REDIS_DB", "0")), decode_responses=True)

# semaphor manages how many jobs can run at once
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

async def process_job(job_id: str) -> None:
    """Process a single job after being acquired from the queue."""
    job_key = f"job:{job_id}"
    job_fields = redis_client.hgetall(job_key)
    out_dir = format_output_directory(job_id)
    upate_redis_with_output_dir(job_id, out_dir)

    logger.info("processing job %s", job_id)
    service = NfgdaService(redis_client, job_id, job_fields, out_dir)
    await service.run()

def process_geotiff_output(job_id: str) -> None:
    """Process the output of the NFGDA algorithm for a given job
    into a stack of GeoTIFFs for final display on the frontend."""
    if redis_client.hget(f"job:{job_id}", "status") == "FAILED":
        return

    logger.info("Generating GeoTIFF series for job %s", job_id)
    result = generate_geotiff_output(job_id, redis_client)
        
    if result is not None:
        logger.error("Failed to generate GeoTIFF series for job %s. Error message: %s", job_id, result)
        redis_client.hset(f"job:{job_id}", mapping={"status": "FAILED", "error_message": result})
    else:
        logger.info("Successfully generated GeoTIFF series for job %s", job_id)
        redis_client.hset(f"job:{job_id}", mapping={"status": "COMPLETED", "num_frames": len(os.listdir(f"/processed_data/{job_id}"))})

async def run_and_release_job(job_id: str) -> None:
    """Run a job and release the semaphore when finished."""
    try:
        await process_job(job_id)
        process_geotiff_output(job_id)
    finally:
        # they took my jerb!
        job_semaphore.release()

def upate_redis_with_output_dir(job_id: str, out_dir: str) -> None:
    """Update Redis with the output directory for a job."""
    redis_client.hset(f"job:{job_id}", "outputDir", out_dir)

def format_output_directory(job_id: str) -> str:
    """Format the output directory for a job."""
    out_dir = f"/nfgda_output/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


async def listen_for_jobs() -> None:
    """Poll Redis for jobs and dispatch them as async tasks.

    Only dequeues a job when the semaphore has capacity, so jobs
    remain in job_queue and their queue position stays accurate.
    """
    logger.info("NFGDA service started")
    logger.info("listening for jobs (max %d concurrent)", MAX_CONCURRENT_JOBS)

    loop = asyncio.get_running_loop()

    while True:
        # Wait until there's capacity to process a job
        await job_semaphore.acquire()

        #then dequeue job details from redis
        _, job_id = await loop.run_in_executor(None, lambda: redis_client.brpop("job_queue", timeout=0))
        logger.info("dequeued job %s", job_id)

        # run that job and release the semaphore when done
        asyncio.create_task(run_and_release_job(job_id))

def main():
    asyncio.run(listen_for_jobs())

if __name__ == "__main__":
    main()