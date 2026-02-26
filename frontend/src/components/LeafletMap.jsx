{/* Note that the leaflet map requires a defined height. */ }
{/* Vite hot reload has inconsistent behavior when making changes to the map, be sure to *fully* reload the page */ }

import { useState, useEffect, useCallback } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import GeotiffLayer from './GeoTiffLayer'


function DisplayPosition({ map, mapLatLng, setMapLatLng }) {

  const onMove = useCallback(() => {
    setMapLatLng(map.getCenter())
  }, [map, setMapLatLng])

  useEffect(() => {
    map.on('move', onMove)
    return () => {
      map.off('move', onMove)
    }
  }, [map, onMove])

  return (
    <p>
      latitude: {mapLatLng.lat.toFixed(4)}, longitude: {mapLatLng.lng.toFixed(4)}{' '}
    </p>
  )
}


export default function LeafletMap({
  stations = [],
  selectedStation,
  setSelectedStation
}) {
  const [map, setMap] = useState(null)
  const [mapLatLng, setMapLatLng] = useState({ "lat": 40.0, "lng": -98.0 });

  useEffect(() => {
    if (!map || !selectedStation) return
    const [lng, lat] = selectedStation.geometry.coordinates
    const latLng = [lat, lng]
    map.setView(latLng, 8)
    setMapLatLng(map.getCenter())
  }, [selectedStation, map, setMapLatLng])

  return (
    <div>
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
        scrollWheelZoom={false}
        style={{ height: '400px', width: '100%' }}
        ref={setMap}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {stations.map((station) =>( //loops through array
        <Marker
          key = {station.properties.station_id}
          position ={[station.geometry.coordinates[1],
                      station.geometry.coordinates[0],]}
        >
          <Popup autoPan={false} > {/*lets user choose station with drop down after clicking on marker */}
            {station.properties.name} <br />
            ({station.properties.station_id}) <br />
            <button type="button" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-1 rounded" onClick={()=> setSelectedStation(station)}>
              Select </button> {/*updates selected stations*/}

          </Popup>
        </Marker>
        ))}
        
        <GeotiffLayer />
      </MapContainer>
    </div>
  )
}