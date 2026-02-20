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
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"

# Regex to extract a parenthesized station ID, e.g. "ALBUQUERQUE (KABX)" -> "KABX"
STATION_ID_REGEX = re.compile(r"\(([^)]+)\)")

class StationService:
    """Service responsible for retrieving available weather stations."""

    def __init__(self) -> None:
        self.station_retrieval_path = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.kmz"

    def retrieve_station_list(self) -> FeatureCollection:
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
        # Download and retrieve the station KMZ file
        with urlopen(self.station_retrieval_path, context=ssl_context) as response:
            if response.status != 200:
                raise ValueError("Failed to download KMZ file")
            kmz_bytes = response.read()

        # Extract the KML file from the KMZ archive
        with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as file:
            kml_filenames = [n for n in file.namelist() if n.endswith(".kml")]
            if not kml_filenames:
                raise ValueError("KMZ archive does not contain a .kml file")
            kml_content = file.read(kml_filenames[0])
            file.close()
            return kml_content

    def parse_stations_from_kml(self, kml_content: bytes) -> FeatureCollection:
        """Parse KML/XML and return a GeoJSON FeatureCollection of stations."""
        root = ET.fromstring(kml_content)
        stations: List[Feature] = []

        # Iterate through all Placemark elements in the KML
        for placemark in root.iter(f"{KML_NAMESPACE}Placemark"):
            name = placemark.find(f"{KML_NAMESPACE}name")
            coords = placemark.find(f"{KML_NAMESPACE}Point/{KML_NAMESPACE}coordinates")

            if name is None or coords is None:
                # Skip invalid placemarks that lack a name or coordinates
                continue

            raw_name = name.text or ""
            # Extract the station ID from the name, e.g. "Station Name (KXXX)" -> KXXX
            id_match = STATION_ID_REGEX.search(raw_name)
            if not id_match:
                # If no ID pattern is found, skip this entry
                continue
            
            # If you have read this far, please audibly meow during the next group meeting :)

            station_id = id_match.group(1)
            # The portion before the parenthesized ID is the human-readable name
            station_name = raw_name[: id_match.start()].strip()

            # KML coordinates are: longitude,latitude[,altitude]
            if coords.text:
                parts = coords.text.strip().split(",")
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    # Altitude is optional
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
                except (ValueError, IndexError):
                    # Skip if coordinates are malformed
                    continue
        
        if not stations:
            raise ValueError("No stations found in KML file")
        stations = FeatureCollection(stations)

        return stations
