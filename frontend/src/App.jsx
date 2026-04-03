import Container from "@mui/material/Container";
import "leaflet/dist/leaflet.css";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import {
  Button,
  Checkbox,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Slider,
} from "@mui/material";
// MUI
import { DateTimePicker, LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { useEffect, useRef, useState } from "react";
import LeafletMap from "./components/LeafletMap";
import RadarStationDropdown from "./components/RadarStationDropdown";
import dayjs from "./utils/dayjsConfig";

export default function App() {
  // User Selection State
  const [stations, setStations] = useState([]);
  const [currentMode, setCurrentMode] = useState(true);
  const [selectedStation, setSelectedStation] = useState("");
  const [selectedDateTime, setSelectedDateTime] = useState(
    dayjs().tz(dayjs.tz.guess()),
  );
  const [timezone, setTimezone] = useState(dayjs.tz.guess());
  const [selectedDuration, setSelectedDuration] = useState("60");

  // API State
  const [jobStatus, setJobStatus] = useState("NONE");
  const [jobId, setjobId] = useState("");
  const [numFrames, setNumFrames] = useState(0);
  const [frames, setFrames] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");

  // Playback State
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playbackRef = useRef(null);

  // --------------------------------------- HANDLERS ----------------------------------------

  // requests a job from /backend/apis/run_request.py and recieves a job_id and response code
  const fetchRadarData = async () => {
    try {
      // ---- validate request ----
      if (!selectedStation?.properties?.station_id) {
        setErrorMessage("Please select a radar station first.");
        return;
      }
      setErrorMessage("");
      const durationMinutes = Number(selectedDuration);
      const requestBody = {
        stationId: selectedStation.properties.station_id,
      };
      if (!currentMode) {
        console.log("using historical data");
        requestBody.startUtc = selectedDateTime
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
        requestBody.endUtc = selectedDateTime
          .add(durationMinutes, "minute")
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
      } else {
        console.log("using current data");
        requestBody.startUtc = dayjs()
          .subtract(durationMinutes + 15, "minute")
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
        requestBody.endUtc = dayjs()
          .subtract(15, "minute")
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
      }

      // ---- reset state ----
      setJobStatus("REQUESTED");
      setjobId("");
      setNumFrames(0);
      setFrames([]);

      // ---- make request ----
      const response = await fetch("/apis/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      // ---- Handle Errors ----
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg =
          errorData.error ||
          errorData.message ||
          `Error ${response.status}: ${response.statusText}`;
        setErrorMessage(errorMsg);
        setJobStatus("FAILED");
        throw new Error(errorMsg); // This sends the message to the catch block
      }

      const data = await response.json();
      setjobId(data.job_id);
    } catch (err) {
      console.error("Fetch Error:", err);
      setJobStatus("FAILED");
      if (!errorMessage) {
        setErrorMessage("A network error occurred. Please try again.");
      }
    }
  };

  // fetch frames once the job is completed and the jobId and numFrames are set
  useEffect(() => {
    async function fetchFrames() {
      if (jobStatus !== "COMPLETED" || !jobId || numFrames <= 0) return;
      console.log(`attempting to fetch ${numFrames} frames for job ${jobId}`);
      try {
        const promises = Array.from({ length: numFrames }, (_, i) =>
          fetch(`/apis/jobs/${jobId}/frames/${i}`)
            .then((res) => {
              if (!res.ok) throw new Error(`Failed frame ${i}`);
              return res.blob();
            })
            .then((blob) => URL.createObjectURL(blob)),
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

  // fetch radar stations from backend at /apis/stations
  useEffect(() => {
    async function loadStations() {
      const response = await fetch("/apis/stations");
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
        const response = await fetch(`/apis/status?job_id=${jobId}`);
        const data = await response.json();
        console.log(data);
        setJobStatus(data.status);
        if (data.error) {
          setErrorMessage(data.error_message);
          console.log("here");
        } else {
          setErrorMessage("");
        }
        if (data.num_frames) {
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
    const newTZ = event.target.value;
    setTimezone(newTZ);
    if (selectedDateTime) {
      setSelectedDateTime(selectedDateTime.tz(newTZ));
    }
  }

  // Handle Playback
  useEffect(() => {
    if (isPlaying) {
      playbackRef.current = setInterval(() => {
        setCurrentFrameIndex((prev) => (prev + 1) % frames.length);
      }, 300);
    } else {
      clearInterval(playbackRef.current);
    }

    return () => clearInterval(playbackRef.current);
  }, [isPlaying, frames.length]);

  // Handle Slider Change
  const handleSliderChange = (event, newValue) => {
    setIsPlaying(false);
    setCurrentFrameIndex(newValue);
  };

  // ---------------------------------------- JSX ----------------------------------------

  return (
    <div>
      <div className="flex flex-col md:flex-row w-full">
        <div className="md:mt-12 p-4 gap-4 min-w-92 flex flex-col">
          {/* Station Selector */}
          <RadarStationDropdown
            stations={stations}
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
          />
          <div className="flex flex-col bg-white">
            <div className="flex items-center">
              <Checkbox
                checked={currentMode}
                onChange={() => {
                  setCurrentMode(!currentMode);
                }}
              ></Checkbox>
              <label
                className="mouse-pointer"
                onClick={() => {
                  setCurrentMode(!currentMode);
                }}
              >
                Get latest radar data
              </label>
            </div>
            <div>
              <FormControl>
                <InputLabel>Duration</InputLabel>
                <Select
                  label="Duration"
                  className="mr-1 mb-2"
                  value={selectedDuration}
                  onChange={(e) => setSelectedDuration(e.target.value)}
                >
                  <MenuItem value="30">30 minutes</MenuItem>
                  <MenuItem value="60">1 hour</MenuItem>
                  <MenuItem value="120">2 hours</MenuItem>
                </Select>
              </FormControl>
              {/* Timezone Selector */}
              <FormControl>
                <InputLabel>Timezone</InputLabel>
                <Select
                  disabled={currentMode}
                  value={timezone}
                  className="mr-1"
                  label="Timezone"
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
                <div className="max-w-92">
                  <DateTimePicker
                    disabled={currentMode}
                    views={["year", "month", "day", "hours", "minutes"]}
                    ampm={false}
                    label="Radar Data Start Time"
                    value={selectedDateTime}
                    onChange={(newValue) =>
                      setSelectedDateTime(dayjs(newValue).tz(timezone))
                    }
                    defaultValue={dayjs("2026-03-11T15:00")}
                    className="w-full"
                  />
                </div>
              </LocalizationProvider>
            </div>
          </div>
          {/* Fetch Button */}
          <Button
            className="w-full max-w-92 h-14"
            onClick={fetchRadarData}
            variant="contained"
          >
            Get Radar Data
          </Button>
          {jobStatus === "PROCESSING" && (
            <p>
              The radar data is being processed. This usually takes a couple
              minutes.
            </p>
          )}
          {jobStatus === "REQUESTED" && (
            <p>The radar data has been requested. Please wait.</p>
          )}
          {errorMessage && <p className="font-bold">{errorMessage}</p>}
        </div>

        <div className="bg-gray-50 min-h-100 w-full">
          <LeafletMap
            stations={stations}
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
            frames={frames}
            currentFrameIndex={currentFrameIndex}
          />
          <div className="flex w-full items-center p-8">
            <button
              type="button"
              onClick={() => setIsPlaying(!isPlaying)}
              className="mr-4 cursor-pointer text-white rounded-full bg-[#1976d2] hover:bg-[#1565c0] shadow hover:shadow-lg transition-all flex p-3 h-max"
            >
              {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
            </button>
            <Slider
              value={currentFrameIndex}
              min={0}
              max={frames.length > 0 ? frames.length - 1 : 0}
              step={1}
              onChange={handleSliderChange}
              valueLabelDisplay="auto"
              marks={frames.map((_, i) => ({ value: i }))}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
