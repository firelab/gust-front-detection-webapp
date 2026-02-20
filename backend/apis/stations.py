from flask import jsonify
from geojson import FeatureCollection
from src.station_service.station_service import StationService

station_service = StationService()

def list_stations_api():
    """ Returns every available radar station and its coordinates. """

    try:
        stations: FeatureCollection = station_service.retrieve_station_list()
        return stations, 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
