import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";

export default function RadarStationDropdown({
  stations = [],
  selectedStation,
  setSelectedStation
}) {
  const stationsArr = Array.isArray(stations) ? stations : []

  function handleChange(event) {
    const stationID = event.target.value

    const stationObj = stationsArr.find(
      s => s?.properties?.station_id === stationID
    )

    setSelectedStation(stationObj || null)
  }

  const stationID = selectedStation?.properties?.station_id || "";

  return (
    <FormControl fullWidth size="medium">
      <InputLabel id="radar-station-label">Select a station</InputLabel>
      <Select
        labelId="radar-station-label"
        value={stationID}
        label="Select a station"
        onChange={handleChange}
      >
        
        {stationsArr.map((feature) => {
          const props = feature?.properties;
          const stationId = props?.station_id ?? "";
          const name = props?.name ?? "";

          if (!stationId) return null;

          return (
            <MenuItem key={stationId} value={stationId}>
              {name} ({stationId})
            </MenuItem>
          );
        })}
      </Select>
    </FormControl>
  );
}