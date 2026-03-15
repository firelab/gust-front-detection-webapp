import redis
from flask import Flask, jsonify, request
from apis.stations import list_stations_api
from apis.run_request import send_job_to_redis_queue
from apis.status import get_job_status
from apis.retrieve_frames import get_frames

app = Flask(__name__)

# Connect to the Redis container
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

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
    return list_stations_api(redis_client)


# Algorithm Runner API
@app.route("/APIs/run", methods=["POST"])
def run_endpoint():
    """Takes station and time frame args, kicks off an NFGDA processing job, and returns the new job ID and status code."""
    if not request.json:
        return jsonify({"error": "Missing request body"}), 400
    
    return send_job_to_redis_queue(redis_client, dict(request.json))
    

# Frame Data API
@app.route("/APIs/frames", methods=["GET"])
def frames_endpoint():
    """Takes job ID, returns list of frames."""
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
    return get_frames(redis_client, job_id)


# Job Status API
@app.route("/APIs/status", methods=["GET"])
def status_endpoint():
    """Takes job ID, returns status."""
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
    return get_job_status(redis_client, job_id)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)

