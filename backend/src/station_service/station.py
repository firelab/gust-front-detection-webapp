from dataclasses import dataclass


@dataclass
class Station:
    """Represents a weather station with its geographic coordinates."""

    station_id: str
    name: str
    lat: float
    lon: float
    altitude: float
