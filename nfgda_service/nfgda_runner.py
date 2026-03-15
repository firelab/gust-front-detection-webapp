import os
import asyncio
import logging
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MAX_NO_DATA_POLLS = int(os.getenv("MAX_NO_DATA_POLLS", "10"))

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

    async def run(self):
        """
        Run the NFGDA process, writing output to specified out_dir.
        Stdout and stderr are streamed line-by-line in real time so logs appear immediately in
        "docker compose logs -f".

        If the algorithm polls for NEXRAD data and finds nothing for MAX_NO_DATA_POLLS consecutive cycles 
        it is killed early so that the job slot is freed and the failure is reported.

        Returns:
            bool: True if the NFGDA process completed successfully, False otherwise.
        """
        
        logger.info(f"timebox parameters set to start_utc: {self.start_utc}, end_utc: {self.end_utc}")
        
        config_path = self.create_temp_config(self.out_dir)
        logger.info("setting environment variable NFGDA_CONFIG_PATH to %s", config_path)
        
        if config_path is None:
            return False, "Failed to create config file"

        logger.info("running algorithm for job %s", self.job_id)
        state = {"no_data_count": 0, "fatal_error_count": 0}

        try:
            # Build an env dict with the per-job config path
            env = os.environ.copy()
            env["NFGDA_CONFIG_PATH"] = config_path

            # asyncio manages the wait from the spawned algorithm subprocess(es)
            proc = await asyncio.create_subprocess_exec(
                "python", "-u", "/app/scripts/NFGDA_Host.py",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Stream stdout and stderr from algorithm subprocesses line-by-line in real time
            stream_tasks = [
                asyncio.create_task(self.stream_pipe(proc.stdout, "stdout")),
                asyncio.create_task(self.monitored_stream(proc.stderr, "stderr", proc, state)),
            ]

            # Wait for the algorithm to complete, with a timeout
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.algo_timeout_seconds)
            except asyncio.TimeoutError:
                logger.error("NFGDA algorithm timed out — killing process")
                proc.kill()
                await proc.wait()
                return False, "NFGDA algorithm timed out"
            finally:
                # flush remaining buffered output
                await asyncio.gather(*stream_tasks)

            # Check if the algorithm was killed due to a data gap
            if state["no_data_count"] >= MAX_NO_DATA_POLLS:
                logger.error(
                    "NFGDA process killed due to data gap — no scans found after %d consecutive polls",
                    MAX_NO_DATA_POLLS,
                )
                return False, f"No radar data found after {MAX_NO_DATA_POLLS} polls"

            # Check if the algorithm exited with a non-zero return code
            if proc.returncode != 0:
                logger.error(
                    "NFGDA algorithm exited with code %d",
                    proc.returncode,
                )
                return False, f"an error occurred processing the algorithm. Error code: {proc.returncode}"

            # Check if fatal errors were logged during processing
            if state["fatal_error_count"] > 0:
                logger.error(
                    "NFGDA algorithm reported %d fatal error(s) during processing",
                    state["fatal_error_count"],
                )
                return False, f"NFGDA algorithm encountered {state['fatal_error_count']} fatal error(s) during processing"

            # woo algorithm dun did its jerb
            return True, None

        finally:
            if config_path and os.path.exists(config_path):
                os.unlink(config_path)

    @staticmethod
    def iso_to_csv_time(iso_str: str) -> str:
        """Convert an ISO 8601 timestamp (e.g. '2024-07-07T01:22:24Z') to the
        comma-separated format that NFGDA config asks for (e.g. 'year,month,day,hour,minute,second').
        """
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return f"{dt.year},{dt.month},{dt.day},{dt.hour},{dt.minute},{dt.second}"

    def create_temp_config(self, out_dir: str) -> str:
        """Create a temporary NFGDA config file. Returns the path to the file."""

        if not os.path.exists("/app/scripts/NFGDA.ini"):
            logger.warning("NFGDA.ini default config not found (proceeding anyway)")

        csv_start = self.iso_to_csv_time(self.start_utc)
        csv_end = self.iso_to_csv_time(self.end_utc)
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

    @staticmethod
    async def stream_pipe(stream, label: str):
        """Read lines from an asyncio stream and log them in real time."""
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info("[NFGDA_Host %s] %s", label, line.decode(errors="replace").rstrip())

    @staticmethod
    async def monitored_stream(stream, label: str, proc, state: dict):
        """Read lines and, for stderr, count consecutive no-data polls.

        Args:
            stream: asyncio subprocess stream (stdout or stderr).
            MAX_NO_DATA_POLLS: Kill the process after this many consecutive
                               "no new scans found" messages.
        """
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            logger.info("[NFGDA_Host %s] %s", label, text)

            if label == "stderr":
                if "no new scans found" in text:
                    state["no_data_count"] += 1
                    if state["no_data_count"] >= MAX_NO_DATA_POLLS:
                        logger.error(
                            "no data found after %d consecutive polls — killing process",
                            state["no_data_count"],
                        )
                        proc.kill()
                        return
                elif "new volume" in text:
                    state["no_data_count"] = 0

                if "fatal error" in text.lower():
                    state["fatal_error_count"] += 1

    
















































































































































    # did you ever find bugs bunny attractive when he'd put on a dress and play a girl bunny?