from flask import Flask, jsonify, request

app = Flask(__name__)


# Station List API
@app.route("/APIs/stations", methods=["GET"])
def list_stations():
    """Returns list of stations and their coordinates."""
    pass


# Algorithm Runner API
@app.route("/APIs/run", methods=["POST"])
def start_run():
    """Takes station and time frame, starts NFGDA job, returns job ID."""
    pass


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
