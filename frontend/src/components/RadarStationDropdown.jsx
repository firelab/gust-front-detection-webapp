export default function RadarStationDropdown({ stations = [], value, onChange }) {
  function handleChange(event) {
    onChange(event.target.value);
  }

  const stationsArr = Array.isArray(stations) ? stations : [];

  return (
    <select value={value} onChange={handleChange}>
      <option value="">Select a station</option>

      {stationsArr.map((feature) => {
        const props = feature?.properties;
        const stationId = props?.station_id ?? "";
        const name = props?.name ?? "";

        if (!stationId) return null;

        return (
          <option key={stationId} value={stationId}>
            {name} ({stationId})
          </option>
        );
      })}
    </select>
  );
}