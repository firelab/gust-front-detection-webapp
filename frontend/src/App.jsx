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
import { useState, useEffect, useRef } from 'react';
import dayjs from './utils/dayjsConfig';

// MUI
import { LocalizationProvider, DateTimePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import { Slider, Button, Select, MenuItem, FormControl, InputLabel, Checkbox } from '@mui/material';

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
  const [geotiffOpacity, setGeotiffOpacity] = useState("80");

  // API State
  const [jobStatus, setJobStatus] = useState("NONE");
  const [jobId, setjobId] = useState("");
  const [numFrames, setNumFrames] = useState(0);
  const [frames, setFrames] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");

  // Playback State
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [sliderValue, setSliderValue] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playbackRef = useRef(null);

  // Endpoint for Server Deployment
  const API_BASE = (import.meta.env.VITE_API_BASE || "/apis").replace(/\/$/, "");
  

  // --------------------------------------- HANDLERS ----------------------------------------

  // requests a job from /backend/apis/run_request.py and recieves a job_id and response code
  const fetchRadarData = async () => {
    try {
      // ---- validate request ----
      //check if station is selected
      if (!selectedStation?.properties?.station_id) {
        setErrorMessage("Please select a radar station first.");
        return;
      }
      // check if endTime is in the past
      if (!currentMode && selectedDateTime.isAfter(dayjs().subtract(Number(selectedDuration), "minute"))) {
        setErrorMessage(
          `Please select a start time at least ${selectedDuration} minutes in the past`,
        );
        return;
      }
      setErrorMessage("");
      const durationMinutes = Number(selectedDuration);
      const requestBody = {
        stationId: selectedStation.properties.station_id,
      };
      if (!currentMode) {
        requestBody.startUtc = selectedDateTime
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
        requestBody.endUtc = selectedDateTime
          .add(durationMinutes, "minute")
          .utc()
          .format("YYYY-MM-DDTHH:mm:ss[Z]");
      } else {
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
      const response = await fetch(`${API_BASE}/run`, {
        method: 'POST',

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
        fetch(`${API_BASE}/jobs/${jobId}/frames/${i}`)
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
  useEffect(() => {
    async function fetchFrames() {
      if (jobStatus !== "COMPLETED" || !jobId || numFrames <= 0) return;
      console.log(`attempting to fetch ${numFrames} frames for job ${jobId}`);
      try {
        const promises = Array.from({ length: numFrames }, async (_, i) => {
          const res = await fetch(`/apis/jobs/${jobId}/frames/${i}`);
          if (res.status === 404) {
            console.warn(`Frame ${i} gave 404 - skipping`);
            return null;
          }
          if (!res.ok) throw new Error(`Failed frame ${i}`);
          const timestamp = res.headers.get("x-frame-timestamp");
          const isForecast = res.headers.get("x-frame-is-forecast");
          const blob = await res.blob();
          return {
            url: URL.createObjectURL(blob),
            timestamp,
            isForecast,
            index: i,
          };
        });
        const frames = await Promise.all(promises);

        const processedFrames = frames
          .filter(Boolean)
          .sort((a, b) => a.index - b.index);

        const firstTime = dayjs(processedFrames[0].timestamp).valueOf();
        const lastTime = dayjs(processedFrames[processedFrames.length - 1].timestamp).valueOf();
        const totalSpan = lastTime - firstTime || 1;

        const sliderFrames = processedFrames.map((frame) => {
          const t = dayjs(frame.timestamp).valueOf();
          const relative = ((t - firstTime) / totalSpan) * 100;

          return {
            ...frame,
            sliderValue: relative,
          };
        });

        setFrames(sliderFrames);
        setIsPlaying(frames.length > 0);
        console.log("Frames fetched successfully: ", frames);
      } catch (err) {
        console.error("Error fetching frames:", err);
      }
    }
    fetchFrames();
  }, [jobStatus, jobId, numFrames]);

  // fetch radar stations from backend at /apis/stations
  useEffect(() => {
    async function loadStations() {
      const maxRetries = 10;

      for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
        try {
          const response = await fetch(`${API_BASE}/stations`);
          if (!response.ok) {
            throw new Error(`Station request failed with status ${response.status}`);
          }

          const stationJson = await response.json();
          const nextStations = Array.isArray(stationJson?.features)
            ? stationJson.features
            : [];
          setStations(nextStations);
          return;
        } catch (err) {
          console.error(`Station load attempt ${attempt} failed:`, err);

          if (attempt === maxRetries) {
            return;
          }

          const delayMs = Math.min(1000 * 2 ** (attempt - 1), 10000);
          await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
      }
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
        const response = await fetch(`${API_BASE}/status?job_id=${jobId}`);
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
    if (isPlaying && frames.length > 0) {
      playbackRef.current = setInterval(() => {
        setCurrentFrameIndex((prev) => {
          const nextIndex = (prev + 1) % frames.length;
          setSliderValue(frames[nextIndex].sliderValue);
          return nextIndex;
        });
      }, 300);
    } else {
      clearInterval(playbackRef.current);
    }
    return () => clearInterval(playbackRef.current);
  }, [isPlaying, frames]);

  // Handle Slider Change
  const handleSliderChange = (event, newValue) => {
    setIsPlaying(false);
    const { frame, index } = getNearestFrame(newValue);
    if (!frame) return;
    setSliderValue(frame.sliderValue);
    setCurrentFrameIndex(index);
  };

  const getNearestFrame = (val) => {
    if (!frames.length) return { frame: null, index: 0 };
    let nearestIndex = 0;
    let nearestDistance = Math.abs(frames[0].sliderValue - val);
    frames.forEach((frame, i) => {
      const dist = Math.abs(frame.sliderValue - val);
      if (dist < nearestDistance) {
        nearestDistance = dist;
        nearestIndex = i;
      }
    });
    return {
      frame: frames[nearestIndex],
      index: nearestIndex,
    };
  };

  // ---------------------------------------- JSX ----------------------------------------

  return (
    <div>
      <div className="flex flex-col md:flex-row w-full">
        <div className="md:mt-6 p-4 gap-4 md:w-92 w-full flex flex-col">
          <div className="mb-6 items-center gap-4">
            <h1 className="text-3xl font-light">Gust Front Web App</h1>
          </div>
          {/* Station Selector */}
          <RadarStationDropdown
            stations={stations}
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
          />
          <div className="flex flex-col">
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
                  className="mr-2 mb-3"
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
                  className="mr-2"
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
            loading={
              jobStatus === "REQUESTED" ||
              jobStatus === "PROCESSING" ||
              jobStatus === "PENDING"
            }
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
          {jobStatus === "PENDING" && (
            <p>The radar data is pending. Please wait.</p>
          )}
          {errorMessage && <p className="font-bold">{errorMessage}</p>}

          {/* Legend */}
          <div className="h-full max-h-10 ">
          </div>
          {numFrames !== 0 && <div className="hidden md:flex outline-1 outline-gray-300 rounded-md p-4 flex-col">
            <p className="text-xl font-bold pb-2">Legend</p>
            <div className="flex mb-2">
              <div className="rounded-full shadow-md bg-red-500 w-10 h-full mr-3 "></div>
              <p>Detected Gust Fronts</p>
            </div>
            <p className="font-bold mb-2">Forcasted Gust Front Probability</p>
            <div className="flex mb-2">
              <div className="rounded-full shadow-md bg-amber-500 w-10 h-full mr-3"></div>
              <p>{"> 30% Confidence"}</p>
            </div>
            <div className="flex mb-2">
              <div className="rounded-full shadow-md bg-cyan-300 w-10 h-full mr-3"></div>
              <p>{"> 50% Confidence"}</p>
            </div>
            <div className="flex">
              <div className="rounded-full shadow-md bg-purple-700 w-10 h-full mr-3"></div>
              <p>{"> 75% Confidence"}</p>
            </div>
          </div>}

          {/* Playback Controls */}
          {/* The CSS is a little cursed. */}
          <div className="flex h-full items-end">
            <div className=" flex md:min-w-[calc(100vw-1rem)] md:pl-92 z-998 w-full">
              {numFrames !== 0 && (
                <div className="flex flex-col items-center w-full max-w-[1200px] bg-white p-2 rounded-xl md:shadow-2xl md:mr-4 md:pr-4">
                  <div className="flex w-full items-center">
                    <button
                      type="button"
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="mr-6 ml-2 cursor-pointer text-white rounded-full bg-[#1976d2] hover:bg-[#1565c0] shadow hover:shadow-lg transition-all flex p-3 h-max"
                    >
                      {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                    </button>
                    <Slider
                      value={sliderValue}
                      min={0}
                      max={100}
                      step={null}
                      onChange={handleSliderChange}
                      valueLabelDisplay="auto"
                      valueLabelFormat={() =>
                        frames[currentFrameIndex]?.timestamp
                          ? dayjs(frames[currentFrameIndex].timestamp)
                            .tz(timezone)
                            .format("YYYY-MM-DD HH:mm z")
                          : "No timestamp"
                      }
                      marks={frames.map((frame, i) => ({
                        value: frame.sliderValue,
                      }))}
                    />
                  </div>
                  <div className="flex w-full pt-4">
                    <div className="flex items-center w-1/2">
                      <p className="min-w-fit px-3 text-sm">Opacity</p>
                      <Slider
                        value={geotiffOpacity}
                        min={0}
                        max={100}
                        step={1}
                        onChange={(event, newValue) =>
                          setGeotiffOpacity(newValue)
                        }
                        valueLabelDisplay="auto"
                      />
                      <p className="min-w-fit px-3">{geotiffOpacity}%</p>
                    </div>
                    <div className="w-1/2">
                      <p className="text-sm text-right">
                        {frames[currentFrameIndex]?.timestamp
                          ? `${dayjs(frames[currentFrameIndex].timestamp).tz(timezone).format("YYYY-MM-DD HH:mm z")}`
                          : "No timestamp available"}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-gray-50 min-h-100 w-full">
          <div
            className={
              jobStatus === "PROCESSING" ||
                jobStatus === "REQUESTED" ||
                jobStatus === "PENDING"
                ? "opacity-50"
                : ""
            }
          >
            <LeafletMap
              stations={stations}
              selectedStation={selectedStation}
              setSelectedStation={setSelectedStation}
              frames={frames}
              currentFrameIndex={currentFrameIndex}
              opacity={geotiffOpacity}
            />
          </div>
        </div>
      </div>
      <footer className="z-999 m-4 absolute bottom-0 left-0 hidden md:block shadow-xl hover:shadow-sm transition-all">
        <a className="outline-1 hover:text-black opacity-50 hover:opacity-100 transition-all rounded-md p-2 flex gap-2 items-center" href="https://github.com/firelab/gust-front-detection-webapp" target="_blank" rel="noopener noreferrer">
          <p className="">Code</p>
          <img src="/assets/github.svg" alt="GitHub" className="w-6 h-6" />
        </a>
      </footer>
    </div>
  );
}
