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

  return (
    <select
      value={selectedStation?.properties?.station_id || ""}
      onChange={handleChange}
    >
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