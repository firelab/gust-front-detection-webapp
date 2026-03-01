import os
import logging

from src.nfgda_service.nfgda_runner import NfgdaRunner

logger = logging.getLogger(__name__)


class NfgdaService:
    """High-level service that orchestrates a single NFGDA run, including
    job lifecycle updates in Redis."""

    def __init__(self, redis_client, job_id: str, job_fields: dict, out_dir: str) -> None:
        self.redis_client = redis_client
        self.job_id = job_id
        self.job_key = f"job:{job_id}"
        self.job_fields = job_fields
        self.out_dir = out_dir

    def run(self) -> None:
        """Execute the NFGDA algorithm and update job status in Redis.

        This method is designed to be called from a worker thread.
        """
        try:
            self.redis_client.hset(self.job_key, "status", "PROCESSING")

            # create output directory
            os.makedirs(self.out_dir, exist_ok=True)

            # instantiate the runner and execute the algorithm
            runner = NfgdaRunner(
                self.job_fields["stationId"],
                self.job_fields["startUtc"],
                self.job_fields["endUtc"],
            )
            success = runner.run(self.out_dir)

            if success:
                self.redis_client.hset(self.job_key, "status", "COMPLETED")
                logger.info("Job %s completed successfully", self.job_id)
            else:
                self.redis_client.hset(self.job_key, "status", "FAILED")
                logger.warning("Job %s failed (runner returned falsy)", self.job_id)

        except Exception as e:
            self.redis_client.hset(self.job_key, "status", "FAILED")
            logger.exception("Job %s failed with exception: %s", self.job_id, e)
