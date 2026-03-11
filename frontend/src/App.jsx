import Container from '@mui/material/Container';
import 'leaflet/dist/leaflet.css';
import LeafletMap from './components/LeafletMap'
import RadarStationDropdown from "./components/RadarStationDropdown";
import { useState, useEffect } from 'react';
import dayjs from 'dayjs';

// MUI
import { LocalizationProvider, DateTimePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import { Slider, Button } from '@mui/material';

export default function App() {

  const [stations, setStations] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedStation, setSelectedStation] = useState("");
  const [selectedDateTime, setSelectedDateTime] = useState(dayjs('2022-04-17T15:30'));

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
        <div className='mt-20 gap-4 flex'>
          <div className='flex-1'>
            <RadarStationDropdown
              stations={stations}
              selectedStation={selectedStation}
              setSelectedStation={setSelectedStation}
            />
          </div>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <DateTimePicker
              views={['year', 'month', 'day', 'hours']}
              label="Radar Data Start Time"
              value={selectedDateTime}
              onChange={(newValue) => setSelectedDateTime(newValue)}
              defaultValue={dayjs('2026-03-11T15:00')}
              className='flex-1'
            />
          </LocalizationProvider>
          <Button className="w-[20%]" variant="contained">Get Radar Data</Button>
        </div>

          <div className=" flex w-full mt-20 mb-2 items-center">
            <button 
              onClick={()=>{setIsPlaying(!isPlaying)}}
              className='mr-4 cursor-pointer text-white rounded-full bg-[#1976d2] hover:bg-[#1565c0] shadow hover:shadow-lg transition-all flex p-3 h-max'>
              {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
            </button>
            <Slider
              defaultValue={30}
              valueLabelDisplay="auto"
              shiftStep={30}
              step={10}
              marks
              min={10}
              max={110}
            />
            <p className='w-[20%] ml-4'>KABX 03:20 PM</p> {/* hardcoded for now */}
          </div>

        <Container className='bg-gray-50 min-h-100' >
          <LeafletMap
            stations={stations}
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
          />
        </Container>

    </Container>
  );
}
