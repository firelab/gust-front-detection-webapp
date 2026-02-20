import uuid
from flask import Flask, jsonify, request
from apis.stations import list_stations_api
from apis.run_request import start_algorithm_run
from src.nfgda_service.nfgda_service import NfgdaService
from src.nfgda_service.models import Job
app = Flask(__name__)

nfgda_service = NfgdaService()

# Station List API
@app.route("/APIs/stations", methods=["GET"])
def stations_endpoint():
    """
    Returns:
        GeoJSON FeatureCollection of stations with properties 
            station_id, name, and altitude, and Point geometry, with a 200 status code.
        
        OR (on failure)
        
        JSON error message with a 404 status code.
    """
    return list_stations_api()


# Algorithm Runner API
@app.route("/APIs/run", methods=["POST"])
def start_run():
    """Takes station and time frame args, kicks off an NFGDA processing job, and returns the new job ID and status."""
    return start_algorithm_run(dict(request.json))
    

# Frame Data API
@app.route("/APIs/frames/<job_id>", methods=["GET"])
def get_frames(job_id: str):
    """Takes job ID, returns list of frames."""
    pass


# Job Status API
@app.route("/APIs/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    """Takes job ID, returns status."""
    pass


if __name__ == '__main__':
    # Listen on all available network interfaces (0.0.0.0)
    app.run(debug=True, host='0.0.0.0', port=5000)
