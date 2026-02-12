from typing import List
from src.station_service.station import Station


class StationService:
    """Service responsible for retrieving available weather stations."""

    def __init__(self, station_retrieval_path: str) -> None:
        self._station_retrieval_path = station_retrieval_path

    def list_stations(self) -> List[Station]:
        """Return the list of available stations."""
        pass


