{/* Note that the leaflet map requires a defined height. */}
{/* When making changes, Vite HMR has inconsistent behavior, be sure to *fully* reload the page */}

import { useState, useEffect, useCallback, useMemo } from 'react'
import { MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import Button from '@mui/material/Button'

const center = [40.0, -98.0]
const zoom = 5

function DisplayPosition({ map }) {
  const [position, setPosition] = useState(() => map.getCenter())

  const onClick = useCallback(() => {
    map.setView(center, zoom)
  }, [map])

  const onMove = useCallback(() => {
    setPosition(map.getCenter())
  }, [map])

  useEffect(() => {
    map.on('move', onMove)
    return () => {
      map.off('move', onMove)
    }
  }, [map, onMove])

  return (
    <p>
      latitude: {position.lat.toFixed(4)}, longitude: {position.lng.toFixed(4)}{' '}
      <Button onClick={onClick} variant="outlined" color="primary">
        reset
      </Button>
    </p>
  )
}

export default function LeafletMap() {
  const [map, setMap] = useState(null)

  const displayMap = useMemo(
    () => (
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={false}
        style={{ height: '400px', width: '100%' }}
        ref={setMap}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
      </MapContainer>
    ),
    [],
  )

  return (
    <div>
      {map ? <DisplayPosition map={map} /> : null}
      {displayMap}
    </div>
  )
}
