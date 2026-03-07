import os  # using it to check if paths are correct
import subprocess  # Needed for sending commands

from src.nfgda_service.models import RunRequest


class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, req: RunRequest, job_id: str, out_dir: str) -> None:
        """
        Run the NFGDA process, writing output to *out_dir*.
        The outdir should be of the format {dir_name}/
        the slash at the end has to be there
        """
        # Make a dir associated with the job rather than all files in one dir
        # Because it is a module download it works from any dir.
        ini_dir = f"jobs/{job_id}/"
        os.makedirs(ini_dir, exist_ok=True)

        with open(f"{ini_dir}NFGDA.ini", "w") as f:
            f.write(f"""
                        [Settings]
                        radar_id = {req.station_id}
                        export_preds_dir = {out_dir}nfgda_detection/
                        export_forecast_dir = {out_dir}
                        V06_dir = {out_dir}V06/
                        custom_start_time = {req.start_utc}
                        custom_end_time = {req.end_utc}
                        evalbox_on = false

                        [labels]
                        label_on = false
                        loc = 36.338467, -106.745250
                        rloc = 35.14972305, -106.82389069
                    """)

        # Use full path to run the script
        subprocess.run(
            ["python", "/app/src/nfgda_service/nfgda_algorithm/scripts/NFGDA_Host.py"],
            cwd=ini_dir,
            timeout=15,
        )
