import Container from '@mui/material/Container';
import 'leaflet/dist/leaflet.css';
import LeafletMap from './components/LeafletMap'

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
