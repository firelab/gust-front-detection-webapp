"""
Station List API – returns every available radar station and its coordinates.
"""

from flask import jsonify
from src.station_service.station_service import StationService


def list_stations(station_service: StationService):
    """
    Response shape:
    [
        { "station_id": "KABX", "name": "...", "lat": 34.87, "lon": -106.82 },
        ...
    ]
    """
    pass
