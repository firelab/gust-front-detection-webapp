import os  # using it to check if paths are correct
import subprocess  # Needed for sending commands

from src.nfgda_service.models import RunRequest


class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, req: RunRequest, out_dir: str) -> None:
        """
        Run the NFGDA process, writing output to *out_dir*.
        The outdir should be of the format {dir_name}/
        the slash at the end has to be there
        """
        """Should write the code under the source dir so I think that this is right"""

        if os.path.exists("nfgda_algorithm/scripts/NFGDA.ini"):
            with open("wills_log_file", "a") as f:
                f.write(
                    "The service was able to find the nfgda.ini file. from /src/nfgda_service/nfgda_runner.py"
                )
        else:
            with open("wills_log_file", "a") as f:
                f.write(
                    "The service was unable to find the nfgda.ini file. from /src/nfgda_service/nfgda_runner.py"
                )

        with open("nfgda_algorithm/scripts/NFGDA.ini", "w") as f:
            # I do not knwo what the export_preds_dir is
            # Different exprot dir
            # different time box

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
        """)  # maybe the loc and the rloc will need to passed

        # timeout after 15 seconds and this should be changed for viewing actual outputs
        try:
            subprocess.run(
                ["python", "nfgda_algorithm/scripts/NFGDA_Host.py"], timeout=15
            )
        except subprocess.TimeoutExpired:
            pass
