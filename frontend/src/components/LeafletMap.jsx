{
  /* Note that the leaflet map requires a defined height. */
  /* Vite hot reload has inconsistent behavior when making changes to the map, be sure to *fully* reload the page */
}

import { useCallback, useEffect, useState } from "react";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import GeoTiffAnimation from "./GeoTiffAnimation";

function DisplayPosition({ map, setMapLatLng }) {
  const onMove = useCallback(() => {
    setMapLatLng(map.getCenter());
  }, [map, setMapLatLng]);

  useEffect(() => {
    map.on("move", onMove);
    return () => {
      map.off("move", onMove);
    };
  }, [map, onMove]);
}

export default function LeafletMap({
  stations = [],
  selectedStation,
  setSelectedStation,
  frames = [],
  currentFrameIndex,
}) {
  const [map, setMap] = useState(null);
  const [mapLatLng, setMapLatLng] = useState({ lat: 40.0, lng: -98.0 });

  useEffect(() => {
    if (!map || !selectedStation) return;
    const [lng, lat] = selectedStation.geometry.coordinates;
    const latLng = [lat, lng];
    map.setView(latLng, 9);
  }, [selectedStation, map]);

  return (
    <div className="h-[600px] md:h-screen">
      {map && (
        <DisplayPosition
          map={map}
          mapLatLng={mapLatLng}
          setMapLatLng={setMapLatLng}
        />
      )}

      <MapContainer
        center={mapLatLng}
        zoom={4}
        scrollWheelZoom={true}
        style={{ height: "100%", width: "100%" }}
        ref={setMap}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {stations.map((station) => {
          const id = station?.properties?.station_id;
          const name = station?.properties?.name;
          const coords = station?.geometry?.coordinates;

          if (!id || !coords || coords.length < 2) return null;

          const [lng, lat] = coords;

          const isSelected = selectedStation?.properties?.station_id === id;

          return (
            <Marker key={id} position={[lat, lng]}>
              <Popup autoPan={false}>
                <span className="font-bold">{name}</span> ({id}) <br />
                <button
                  type="button"
                  className={`w-full font-bold py-1 mt-1 rounded ${isSelected
                    ? "bg-white outline-2 outline-gray-200"
                    : "bg-[#1976d2] hover:bg-[#1565c0] text-white cursor-pointer"
                    }`}
                  onClick={() => setSelectedStation(station)}
                >
                  {isSelected ? "Selected" : "Select"}
                </button>
              </Popup>
            </Marker>
          );
        })}
        <GeoTiffAnimation frames={frames} currentIndex={currentFrameIndex} />
      </MapContainer>
    </div>
  );
}
