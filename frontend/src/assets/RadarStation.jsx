import {useEffect, useState} from 'react'
//must run app.py first so the list is not blank

export default function RadarStations({value, onChange}){

    const[stations, setStations] = useState([]);

    useEffect(() => {
        async function loadStations() {
            let nextStations = [];

            const response = await fetch("/APIs/stations");
            const geoJson = await response.json();

            if(geoJson !== null && geoJson !== undefined){ //blank
                if(Array.isArray(geoJson.features)){
                    nextStations = geoJson.features;
                }    
            }
            setStations(nextStations);
        }
        loadStations();
       }, [])

       function handleChange(event){
        const selectedValue = event.target.value;
        onChange(selectedValue);
       }

       return (
        <select value={value} onChange={handleChange}>
            <option value="">Select a station</option>

            {stations.map((feature) => { //better than loop?
                const props = feature ? feature.properties : null;
                const stationId = props ? props.station_id : "";
                const name = props ? props.name : "";

                if (!stationId) return null;

                return(
                    <option key={stationId} value={stationId}>
                        {name} ({stationId})
                    </option>
                );
            })}
        </select>
       );
 }