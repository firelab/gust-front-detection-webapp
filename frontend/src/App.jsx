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
  const [selectedDateTime, setSelectedDateTime] = useState(dayjs().tz(dayjs.tz.guess()));
  const [timezone, setTimezone] = useState(dayjs.tz.guess());
  const [jobStatus, setjobStatus] = useState("NONE");
  const [jobId, setjobId] = useState("");
  const [numFrames, setNumFrames] = useState(0);
  const [frames, setFrames] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  
  // --------------------------------------- HANDLERS ----------------------------------------

  // requests a job from /backend/apis/run_request.py and recieves a job_id and response code
  const fetchRadarData = async () => {
    try {
      // ---- validate request ----
      if (!selectedStation?.properties?.station_id) {
        throw new Error("No station selected");
      }
      const requestBody = {
          stationId: selectedStation.properties.station_id
      };
      if (!currentMode){
        console.log("using historical data");
        // if not fetching current data the end time requested is t+30 minutes
        requestBody.startUtc = selectedDateTime.utc().format("YYYY-MM-DDTHH:mm:ss[Z]")
        requestBody.endUtc = selectedDateTime.add(30, 'minute').utc().format("YYYY-MM-DDTHH:mm:ss[Z]")
      } else {
        // currently using the same timebox as the default in run_request.py when no timebox is provided
        console.log("using current data");
        requestBody.startUtc = dayjs().subtract(45, 'minute').utc().format("YYYY-MM-DDTHH:mm:ss[Z]")
        requestBody.endUtc = dayjs().subtract(25, 'minute').utc().format("YYYY-MM-DDTHH:mm:ss[Z]")
      }
      // ---- make request ----
      setjobStatus("NONE");
      const response = await fetch("/APIs/run", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const data = await response.json();
      console.log(data)
      setjobId(data.job_id);
    } catch (err) {
      console.error(err)
    }
  };

  // fetch frames once the job is completed and the jobId and numFrames are set
useEffect(() => {
  async function fetchFrames() {
    if (jobStatus !== "COMPLETED" || !jobId || numFrames <= 0) return;

    console.log(`attempting to fetch ${numFrames} frames for job ${jobId}`);

    try {
      const promises = Array.from({ length: numFrames }, (_, i) =>
        fetch(`/APIs/jobs/${jobId}/frames/${i}`)
          .then(res => {
            if (!res.ok) throw new Error(`Failed frame ${i}`);
            return res.blob();
          })
          .then(blob => URL.createObjectURL(blob))
      );

      const urls = await Promise.all(promises);
      setFrames(urls);
      console.log("Frames fetched successfully: ", urls);
    } catch (err) {
      console.error("Error fetching frames:", err);
    }
  }

  fetchFrames();
}, [jobStatus, jobId, numFrames]);

  // fetch radar stations from backend at /APIs/stations
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

  // get the status of the job from APIs/job_status every 5 seconds until the job is completed or failed
  useEffect(() => {
    if (!jobId) return;
    if (jobStatus === "COMPLETED" || jobStatus === "FAILED") {
      return;
    }
    
    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(`/APIs/status?job_id=${jobId}`);
        const data = await response.json();
        console.log(data);
        setjobStatus(data.status);
        if (data.error_message){
          setErrorMessage(data.error_message);
        } else {
          setErrorMessage("");
        }
        if (data.num_frames){
          setNumFrames(data.num_frames);
        } else {
          setNumFrames(0);
        }
        } catch (err) {
          console.error(err);
        }
    }, 5000);
  
    return () => clearInterval(intervalId);
  }, [jobId, jobStatus]);

  // timezone change handler
  function handleTimezoneChange(event) {
    const newTZ = event.target.value

    setTimezone(newTZ)

    if (selectedDateTime) {
      setSelectedDateTime(selectedDateTime.tz(newTZ))
    }
  }

  // ---------------------------------------- JSX ----------------------------------------

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
              <Checkbox checked={currentMode} onChange={()=>{setCurrentMode(!currentMode)}}>
              </Checkbox>
              <p>Use Current Data</p>
            </div>
            <div>
            {/* Timezone Selector */}
              <FormControl>
                <InputLabel>Timezone</InputLabel>
                <Select
                  disabled={currentMode}
                  value={timezone}
                  className='min-w-24 mr-1'
                  label='Timezone'
                  onChange={handleTimezoneChange}
                >
                  <MenuItem value="UTC">UTC</MenuItem>
                  <MenuItem value="America/Anchorage">Alaska</MenuItem>
                  <MenuItem value="America/Los_Angeles">Pacific</MenuItem>
                  <MenuItem value="America/Denver">Mountain</MenuItem>
                  <MenuItem value="America/Chicago">Central</MenuItem>
                  <MenuItem value="America/New_York">Eastern</MenuItem>
                </Select>
              </FormControl>
              {/* Date Time Selector */}
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DateTimePicker
                  disabled={currentMode}
                  views={['year', 'month', 'day', 'hours', 'minutes']}
                  ampm={false}
                  label="Radar Data Start Time"
                  value={selectedDateTime}
                  onChange={(newValue) => setSelectedDateTime(dayjs(newValue).tz(timezone))}
                  defaultValue={dayjs('2026-03-11T15:00')}
                  className='flex-1'
                />
              </LocalizationProvider>
            </div>
          </div>
          {/* Fetch Button */}
          <Button
            className="w-[20%] h-14" 
            onClick={fetchRadarData}
            variant="contained">
              Get Radar Data
          </Button>
        </div>
        {jobId && <p>Job ID: {jobId}</p>}
        <p>Job Status: {jobStatus}</p>
        <p className='text-red-800 font-bold'>{errorMessage}</p>

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
            frames={frames}
          />
        </Container>

    </Container>
  );
}
