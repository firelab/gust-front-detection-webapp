import Container from '@mui/material/Container';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css';
import LeafletMap from './LeafletMap'

export default function App() {

  return (
    <Container maxWidth="md">
        <p className='font-bold mt-8'>Leaflet Map</p>
        <Container className='bg-gray-200 min-h-100 mt-8' >
          <LeafletMap/>
        </Container>
        
    </Container>
  );
}
