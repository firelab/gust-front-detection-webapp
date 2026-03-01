import os  # using it to check if paths are correct
import subprocess  # Needed for sending commands
import logging
import tempfile.NamedTemporaryFile as tempfile
from src.nfgda_service.models import RunRequest

class NfgdaRunner:
    """Executes the NFGDA algorithm for a given run request."""

    def __init__(self, station_id: str, start_utc: str, end_utc: str, out_dir: str) -> None:
        """
        Initialize the NfgdaRunner with the given parameters.

        Args:
            station_id (str): The station code.
            start_utc (str): The start time in UTC.
            end_utc (str): The end time in UTC.
            out_dir (str): The output directory.
        """
        
        self.algo_timeout_seconds = 120
        self.station_id = station_id
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.out_dir = out_dir

    def run(self) -> bool:
        """
        Run the NFGDA process, writing output to specified out_dir.

        Returns:
            bool: True if the NFGDA process completed successfully, False otherwise.
        """
        
        logging.info("creating temporary config file")  
        config_path = self.create_temp_config()
        if config_path is None:
            return False
        
        logging.info("running algorithm")
        try:
            # set the environment variable NFGDA_CONFIG_PATH to the path of the temporary config file
            logging.info("setting environment variable NFGDA_CONFIG_PATH to %s", config_path)
            env = os.environ.copy()
            env["NFGDA_CONFIG_PATH"] = config_path
            # Pass list of args instead of a string to subprocess.run
            subprocess.run(["python", "nfgda_algorithm/scripts/NFGDA_Host.py"], timeout=self.algo_timeout_seconds, env=env)
            logging.info("algorithm processing completed")
            return True
        except subprocess.TimeoutExpired:
            logging.error("NFGDA algorithm timed out")
            return False
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def create_temp_config(self) -> str:
        """ Create a temporary NFGDA config file. Returns the path to the file. """
        
        if not os.path.exists("nfgda_algorithm/scripts/NFGDA.ini"):
            logging.warning("NFGDA.ini default config not found (proceeding anyway)")

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
            
            logging.info("temporary config file created at %s", f.name)
            return f.name