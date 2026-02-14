import io
import re
import ssl
import zipfile
import xml.etree.ElementTree as ET
from typing import List
from geojson import FeatureCollection, Point, Feature
from urllib.request import urlopen
import certifi

# KML namespace used in the XML tags
_KML_NS = "{http://www.opengis.net/kml/2.2}"

# Regex to extract a parenthesized station ID, e.g. "ALBUQUERQUE (KABX)" -> "KABX"
_STATION_ID_RE = re.compile(r"\(([^)]+)\)")

class StationService:
    """Service responsible for retrieving available weather stations."""

    def __init__(self) -> None:
        self.station_retrieval_path = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.kmz"

    def list_stations(self) -> FeatureCollection:
        """Fetch a list of weather stations.
            Weather stations are stored in a db table. If the table is empty or records are more 
            than 1 day old, the table is repopulated from the NOAA KMZ file.
            This Requires downloading the newest KMZ file from NOAA, extracting the KML, and parsing the stations.

            For now we'll just fetch and process the KMZ every time on demand until getting a database 
            implementation working but
            TODO: add a db table of weather stations and throw it in hyah!

        """

        
        kml_content = self.download_and_extract_kml()
        return self.parse_stations_from_kml(kml_content)

    def download_and_extract_kml(self) -> bytes:
        """Fetch the KMZ archive from the configured URL and return raw KML bytes."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(self.station_retrieval_path, context=ssl_context) as response:
            if response.status != 200:
                raise ValueError("Failed to download KMZ file")
            kmz_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as zf:
            kml_filenames = [n for n in zf.namelist() if n.endswith(".kml")]
            if not kml_filenames:
                raise ValueError("KMZ archive does not contain a .kml file")
            return zf.read(kml_filenames[0])

    def parse_stations_from_kml(self, kml_content: bytes) -> FeatureCollection:
        """Parse KML XML and return a list of Station dataclass instances."""
        root = ET.fromstring(kml_content)
        stations: List[Feature] = []

        for placemark in root.iter(f"{_KML_NS}Placemark"):
            name_el = placemark.find(f"{_KML_NS}name")
            coords_el = placemark.find(f"{_KML_NS}Point/{_KML_NS}coordinates")

            if name_el is None or coords_el is None:
                raise ValueError(f"Invalid station: {placemark}")

            raw_name = name_el.text or ""
            id_match = _STATION_ID_RE.search(raw_name)
            if not id_match:
                raise ValueError(f"Invalid station name: {raw_name}")

            station_id = id_match.group(1)
            # The portion before the parenthesized ID is the human-readable name
            station_name = raw_name[: id_match.start()].strip()

            # KML coordinates are: longitude,latitude[,altitude]
            parts = coords_el.text.strip().split(",")
            lon = float(parts[0])
            lat = float(parts[1])
            altitude = float(parts[2]) if len(parts) > 2 else None
            point = Point((lon, lat))
            stations.append(
                Feature(
                    geometry=point,
                    properties={
                        "station_id": station_id,
                        "name": station_name,
                        "altitude": altitude,
                    }
                )
            )
        if len(stations) <= 1:
            raise ValueError("No stations found in KML file")
        stations = FeatureCollection(stations)

        return stations
