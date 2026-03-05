import os
import asyncio
import logging
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, station_id: str, start_utc: str, end_utc: str, job_id: str, out_dir: str) -> None:
        """
        Initialize the NfgdaRunner with the given parameters.

        Args:
            station_id (str): The station code.
            start_utc (str): The start time in UTC.
            end_utc (str): The end time in UTC.
            job_id (str): The job ID.
            out_dir (str): The output directory.
            """

        self.algo_timeout_seconds = 600
        self.station_id = station_id
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.job_id = job_id
        self.out_dir = out_dir

    @staticmethod
    async def _stream_pipe(stream, label: str):
        """Read lines from an asyncio stream and log them in real time."""
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info("[NFGDA_Host %s] %s", label, line.decode(errors="replace").rstrip())

    async def run(self) -> bool:
        """
        Run the NFGDA process, writing output to specified out_dir.

        Uses asyncio.create_subprocess_exec so the event loop is never
        blocked while the algorithm runs.  Stdout and stderr are streamed
        line-by-line in real time so logs appear immediately in
        `docker compose logs -f`.

        Returns:
            bool: True if the NFGDA process completed successfully, False otherwise.
        """

        config_path = self.create_temp_config(self.out_dir)
        logger.info("setting environment variable NFGDA_CONFIG_PATH to %s", config_path)
        
        if config_path is None:
            return False

        logger.info("running algorithm for job %s", self.job_id)
        try:
            # Build an env dict with the per-job config path
            env = os.environ.copy()
            env["NFGDA_CONFIG_PATH"] = config_path

            # Spawn the algorithm as a direct child process — no intermediate
            # Python worker process.  asyncio manages the wait without blocking.
            proc = await asyncio.create_subprocess_exec(
                "python", "-u", "/app/scripts/NFGDA_Host.py",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Stream stdout and stderr line-by-line in real time
            stream_tasks = [
                asyncio.create_task(self._stream_pipe(proc.stdout, "stdout")),
                asyncio.create_task(self._stream_pipe(proc.stderr, "stderr")),
            ]

            try:
                await asyncio.wait_for(proc.wait(), timeout=self.algo_timeout_seconds)
            except asyncio.TimeoutError:
                logger.error("NFGDA algorithm timed out — killing process")
                proc.kill()
                await proc.wait()
                return False
            finally:
                # Ensure remaining buffered output is flushed
                await asyncio.gather(*stream_tasks)

            if proc.returncode != 0:
                logger.error(
                    "NFGDA algorithm exited with code %d",
                    proc.returncode,
                )
                return False

            logger.info("algorithm processing completed for job %s", self.job_id)
            return True

        finally:
            if config_path and os.path.exists(config_path):
                os.unlink(config_path)

    @staticmethod
    def _iso_to_csv_time(iso_str: str) -> str:
        """Convert an ISO 8601 timestamp (e.g. '2024-07-07T01:22:24Z') to the
        comma-separated format that NFGDA_load_config expects: 'year,month,day,hour,minute,second'.
        """
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return f"{dt.year},{dt.month},{dt.day},{dt.hour},{dt.minute},{dt.second}"

    def create_temp_config(self, out_dir: str) -> str:
        """Create a temporary NFGDA config file. Returns the path to the file."""

        if not os.path.exists("/app/scripts/NFGDA.ini"):
            logger.warning("NFGDA.ini default config not found (proceeding anyway)")

        csv_start = self._iso_to_csv_time(self.start_utc)
        csv_end = self._iso_to_csv_time(self.end_utc)
        logger.info("config times: start=%s -> %s, end=%s -> %s",
                     self.start_utc, csv_start, self.end_utc, csv_end)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".ini", prefix="nfgda_") as f:
            f.write(f"""[Settings]
                radar_id = {self.station_id}
                export_preds_dir = {self.out_dir}nfgda_detection/
                export_forecast_dir = {self.out_dir}
                V06_dir = {self.out_dir}V06/
                custom_start_time = {csv_start}
                custom_end_time = {csv_end}
                evalbox_on = false

                [labels]
                label_on = false
                loc = 36.338467, -106.745250
                rloc = 35.14972305, -106.82389069
                """)

            logger.info("temporary config file created at %s", f.name)
            return f.name