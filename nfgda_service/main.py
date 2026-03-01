import redis
import logging
from concurrent.futures import ThreadPoolExecutor
from src.nfgda_service.nfgda_service import NfgdaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = 4

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)


def process_job(job_id: str) -> None:
    """Process a single job within a thread pool worker."""
    job_key = f"job:{job_id}"
    job_fields = redis_client.hgetall(job_key)
    out_dir = f"/job_results/{job_id}/"

    logger.info("processing job %s", job_id)
    service = NfgdaService(redis_client, job_id, job_fields, out_dir)
    service.run()

def main():
    logger.info("NFGDA service started")
    logger.info("listening for jobs (max %d concurrent)", MAX_CONCURRENT_JOBS)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        while True:
            # blocking pop, listens for jobs
            _, job_id = redis_client.brpop("job_queue", timeout=0)
            logger.info("dequeued job %s", job_id)

            # submit the job to the thread pool (non-blocking)
            executor.submit(process_job, job_id)
    
    logger.info("NFGDA service stopped")

if __name__ == "__main__":
    main()