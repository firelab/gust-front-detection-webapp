import os
import asyncio
import logging
import tempfile

logger = logging.getLogger(__name__)

class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, station_id: str, start_utc: str, end_utc: str) -> None:
        """
        Initialize the NfgdaRunner with the given parameters.

        Args:
            station_id (str): The station code.
            start_utc (str): The start time in UTC.
            end_utc (str): The end time in UTC.
        """

        self.algo_timeout_seconds = 120
        self.station_id = station_id
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.out_dir = out_dir

    async def run(self) -> bool:
        """
        Run the NFGDA process, writing output to specified out_dir.

        Uses asyncio.create_subprocess_exec so the event loop is never
        blocked while the algorithm runs. The calling coroutine simply
        awaits the child process's exit.

        Returns:
            bool: True if the NFGDA process completed successfully, False otherwise.
        """

        logger.info("creating temporary config file")
        config_path = self.create_temp_config(self.out_dir)
        if config_path is None:
            return False

        logger.info("running algorithm")
        try:
            # Build an env dict with the per-job config path
            env = os.environ.copy()
            env["NFGDA_CONFIG_PATH"] = config_path

            logger.info("setting environment variable NFGDA_CONFIG_PATH to %s", config_path)

            # Spawn the algorithm as a direct child process — no intermediate
            # Python worker process.  asyncio manages the wait without blocking.
            proc = await asyncio.create_subprocess_exec(
                "python", "nfgda_algorithm/scripts/NFGDA_Host.py",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.algo_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error("NFGDA algorithm timed out — killing process")
                proc.kill()
                await proc.wait()
                return False

            if proc.returncode != 0:
                logger.error(
                    "NFGDA algorithm exited with code %d\nstderr: %s",
                    proc.returncode,
                    stderr.decode(errors="replace") if stderr else "(empty)",
                )
                return False

            logger.info("algorithm processing completed")
            return True

        finally:
            if config_path and os.path.exists(config_path):
                os.unlink(config_path)

    def create_temp_config(self, out_dir: str) -> str:
        """Create a temporary NFGDA config file. Returns the path to the file."""

        if not os.path.exists("nfgda_algorithm/scripts/NFGDA.ini"):
            logger.warning("NFGDA.ini default config not found (proceeding anyway)")

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".ini", prefix="nfgda_") as f:
            f.write(f"""[Settings]
                radar_id = {self.station_id}
                export_preds_dir = {self.out_dir}nfgda_detection/
                export_forecast_dir = {self.out_dir}
                V06_dir = {self.out_dir}V06/
                custom_start_time = {self.start_utc}
                custom_end_time = {self.end_utc}
                evalbox_on = false

                [labels]
                label_on = false
                loc = 36.338467, -106.745250
                rloc = 35.14972305, -106.82389069
                """)

            logger.info("temporary config file created at %s", f.name)
            return f.name