from flask import jsonify
from geojson import FeatureCollection
from src.station_service.station_service import StationService


def list_stations(station_service: StationService):
    """ Returns every available radar station and its coordinates. """

    try:
        stations: FeatureCollection = station_service.list_stations()
        return stations, 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
