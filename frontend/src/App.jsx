import Container from '@mui/material/Container';
import 'leaflet/dist/leaflet.css';
import LeafletMap from './components/LeafletMap'
import RadarStationDropdown from "./components/RadarStationDropdown";
import { useState, useEffect } from 'react';

export default function App() {

  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState("");

  // fetch radar stations from backend API
  useEffect(() => {
    async function loadStations() {
      const response = await fetch("/APIs/stations");
      const stationJson = await response.json();

      const nextStations = Array.isArray(stationJson?.features)
        ? stationJson.features
        : [];

      setStations(nextStations);
    }

    loadStations();
  }, []);


  return (
    <Container maxWidth="md">
        <p className='font-bold mt-8'>Leaflet Map</p>
        <div className='mt-4'>
        <RadarStationDropdown
          stations={stations}
          value={selectedStation}
          onChange={setSelectedStation}
        />
        </div>
        <Container className='bg-gray-200 min-h-100 mt-8' >
          <LeafletMap/>
        </Container>
        
    </Container>
  );
}
