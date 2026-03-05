import os
import asyncio
import redis
import logging
from nfgda_service import NfgdaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", "6379")), db=int(os.getenv("REDIS_DB", "0")), decode_responses=True)

job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

async def process_job(job_id: str) -> None:
    """Process a single job, acquiring the concurrency semaphore first."""
    async with job_semaphore:
        job_key = f"job:{job_id}"
        job_fields = redis_client.hgetall(job_key)
        out_dir = format_output_directory(job_id)
        upate_redis_with_output_dir(job_id, out_dir)

        logger.info("processing job %s", job_id)
        service = NfgdaService(redis_client, job_id, job_fields, out_dir)
        await service.run()

def upate_redis_with_output_dir(job_id: str, out_dir: str) -> None:
    """Update Redis with the output directory for a job."""
    redis_client.hset(f"job:{job_id}", "outputDir", out_dir)

def format_output_directory(job_id: str) -> str:
    """Format the output directory for a job."""
    out_dir = f"/job_results/{job_id}/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


async def listen_for_jobs() -> None:
    """Poll Redis for jobs and dispatch them as async tasks.

    redis-py's brpop is blocking, so we run it in a thread to avoid
    stalling the event loop.  Each job is launched as a concurrent Task;
    the semaphore inside process_job throttles actual subprocess count.
    """
    logger.info("NFGDA service started")
    logger.info("listening for jobs (max %d concurrent)", MAX_CONCURRENT_JOBS)

    loop = asyncio.get_running_loop()

    while True:
        # brpop blocks — offload to a thread so the event loop stays free
        _, job_id = await loop.run_in_executor(
            None, lambda: redis_client.brpop("job_queue", timeout=0)
        )
        logger.info("dequeued job %s", job_id)

        # Fire-and-forget: each task self-manages via the semaphore
        asyncio.create_task(process_job(job_id))


def main():
    asyncio.run(listen_for_jobs())

if __name__ == "__main__":
    main()