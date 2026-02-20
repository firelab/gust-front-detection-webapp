import Container from '@mui/material/Container';
import 'leaflet/dist/leaflet.css';
import LeafletMap from './LeafletMap'
import RadarStations from "./assets/RadarStation";
import { useState } from 'react';

export default function App() {

  const [StationId, setStationId] = useState("");

  return (
    <Container maxWidth="md">
        <p className='font-bold mt-8'>Leaflet Map</p>
        <div className='mt-4'> {/*tailwind utility class*/}
          <RadarStations value={StationId} onChange={setStationId} />
        </div>
        <Container className='bg-gray-200 min-h-100 mt-8' >
          <LeafletMap/>
        </Container>
        
    </Container>
  );
}
