import Container from '@mui/material/Container';
import 'leaflet/dist/leaflet.css';
import LeafletMap from './components/LeafletMap'
import RadarStationDropdown from "./components/RadarStationDropdown";
import { useState, useEffect } from 'react';
import dayjs from './utils/dayjsConfig';

// MUI
import { LocalizationProvider, DateTimePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import { Slider, Button, Select, MenuItem, FormControl, InputLabel, Checkbox } from '@mui/material';

export default function App() {

  const [stations, setStations] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentMode, setCurrentMode] = useState(true);
  const [selectedStation, setSelectedStation] = useState("");
  const [selectedDateTime, setSelectedDateTime] = useState(dayjs().tz(dayjs.tz.guess()).startOf('hour'));
  const [timezone, setTimezone] = useState(dayjs.tz.guess());
  
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

  function handleTimezoneChange(event) {
    const newTZ = event.target.value

    setTimezone(newTZ)

    if (selectedDateTime) {
      setSelectedDateTime(selectedDateTime.tz(newTZ))
    }
  }

  return (
    <Container maxWidth="md">
        <div className='mt-20 gap-4 flex items-end'>
          {/* Station Selector */}
          <div className='flex-1'>
            <RadarStationDropdown
              stations={stations}
              selectedStation={selectedStation}
              setSelectedStation={setSelectedStation}
            />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center">
              <Checkbox value={currentMode} onChange={()=>{setCurrentMode(!currentMode)}}>
              </Checkbox>
              <p>Use Current Data</p>
            </div>
            <div>
            {/* Timezone Selector */}
              <FormControl>
                <InputLabel>Timezone</InputLabel>
                <Select
                  disabled={!currentMode}
                  value={timezone}
                  className='min-w-24 mr-1'
                  label='Timezone'
                  onChange={handleTimezoneChange}
                >
                  <MenuItem value="UTC">UTC</MenuItem>
                  <MenuItem value="America/Los_Angeles">Pacific</MenuItem>
                  <MenuItem value="America/Denver">Mountain</MenuItem>
                  <MenuItem value="America/Chicago">Central</MenuItem>
                  <MenuItem value="America/New_York">Eastern</MenuItem>
                  <MenuItem value="America/Anchorage">Alaska</MenuItem>
                </Select>
              </FormControl>
              {/* Date Time Selector */}
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DateTimePicker
                  disabled={!currentMode}
                  views={['year', 'month', 'day', 'hours']}
                  label="Radar Data Start Time"
                  value={selectedDateTime}
                  onChange={(newValue) => setSelectedDateTime(dayjs(newValue).tz(timezone).startOf('hour'))}
                  defaultValue={dayjs('2026-03-11T15:00')}
                  className='flex-1'
                />
              </LocalizationProvider>
            </div>
          </div>
          {/* Fetch Button */}
          <Button
            className="w-[20%] h-14" 
            onClick={()=>{console.log(selectedStation.properties.station_id + selectedDateTime.utc().format())}}
            variant="contained">
              Get Radar Data
          </Button>
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
