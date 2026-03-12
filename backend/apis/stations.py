from flask import jsonify
from geojson import FeatureCollection
from src.station_service.station_service import StationService

def list_stations_api(redis_client):
    """ Returns every available radar station and its coordinates. """

    station_service = StationService(redis_client)
    try:
        stations: FeatureCollection = station_service.retrieve_station_list()
        return stations, 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
