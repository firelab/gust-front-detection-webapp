import logging
import os

from nfgda_runner import NfgdaRunner

logger = logging.getLogger(__name__)


class NfgdaService:
    """ high-level service that orchestrates a single NFGDA run, including job lifecycle updates in Redis. """

    def __init__(self, redis_client, job_id: str, job_fields: dict, out_dir: str) -> None:
        self.redis_client = redis_client
        self.job_id = job_id
        self.job_key = f"job:{job_id}"
        self.job_fields = job_fields
        self.out_dir = out_dir

    async def run(self) -> None:
        """ execute the NFGDA algorithm and update job status in Redis. """
        try:
            self.redis_client.hset(self.job_key, mapping={"status": "PROCESSING"})

            # create output directory
            os.makedirs(self.out_dir, exist_ok=True)

            # instantiate the runner and execute the algorithm
            runner = NfgdaRunner(
                self.job_fields["stationId"],
                self.job_fields["startUtc"],
                self.job_fields["endUtc"],
                self.job_id,
                self.out_dir
            )
            success, message = await runner.run()

            # update job status in redis
            if success:
                logger.info("algorithm processing for job %s completed successfully", self.job_id)
            else:
                # no, this is patrick
                self.redis_client.hset(self.job_key, mapping={"status": "FAILED", "error_message": message})
                logger.warning("job %s failed (runner returned falsy)", self.job_id)
                logger.warning("error message: %s", message)

        except Exception as e:
            self.redis_client.hset(self.job_key, mapping={"status": "FAILED", "error_message": str(e)})
            logger.exception("job %s failed with exception", self.job_id)
