from dataclasses import dataclass


@dataclass
class Station:
    """Represents a weather station with its geographic coordinates."""

    id: str
    name: str
    lat: float
    lon: float
