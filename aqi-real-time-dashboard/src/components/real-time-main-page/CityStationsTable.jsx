import React, { useEffect, useState } from 'react';
import './CityStationsTable.css'; // Optional CSS for styling

function CityStationsTable({
  selectedStation,
  stations,
  calculateAQI,
  getAQICategory,
  fetchStationData // helper function to fetch current data for each station
}) {
  const [cityStationsData, setCityStationsData] = useState([]);
  const [loading, setLoading] = useState(false);
  const city = selectedStation?.city || '';

  useEffect(() => {
    if (!city) return;

    let isCancelled = false;
    setLoading(true);

    // Compute the filtered stations for the given city inside the effect to avoid extra dependencies
    const cityStations = stations.filter(station => station.city === city);

    const loadDataForCityStations = async () => {
      try {
        const dataPromises = cityStations.map(async (station) => {
          const stationPollutants = await fetchStationData(station);
          // Calculate AQI for this station
          const aqiValue = calculateAQI(stationPollutants);
          const aqiCategory = getAQICategory(aqiValue);
          return {
            ...station,
            ...stationPollutants,
            aqi: aqiValue,
            aqiCategory,
          };
        });

        const results = await Promise.all(dataPromises);

        if (!isCancelled) {
          // Sort stations by AQI in descending order and slice the top 10
          const topTenStations = results.sort((a, b) => b.aqi - a.aqi).slice(0, 10);
          setCityStationsData(topTenStations);
          setLoading(false);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Error fetching city stations data:', error);
          setLoading(false);
        }
      }
    };

    loadDataForCityStations();

    return () => {
      isCancelled = true;
    };
  }, [city, stations, calculateAQI, getAQICategory, fetchStationData]);

  if (!city) return null;

  return (
    <div className="city-stations-table-container">
      <h2 className="table-heading">
        {city} - Top 10 Stations by Air Quality
      </h2>
      <div className="table-wrapper" style={{ position: 'relative' }}>
        <table className="city-stations-table">
          <thead>
            <tr>
              <th>Location</th>
              <th>Status</th>
              <th>AQI</th>
              <th>PM2.5 (µg/m³)</th>
              <th>PM10 (µg/m³)</th>
              <th>Temp (°C)</th>
              <th>Humi (%)</th>
            </tr>
          </thead>
          <tbody>
            {cityStationsData.map((stationData) => {
              const { station_id, name, aqi, pm2_5, pm10, temp, hum, aqiCategory } = stationData;
              return (
                <tr key={station_id}>
                  <td>{name}</td>
                  <td style={{ color: aqiCategory?.color || '#000' }}>
                    {aqiCategory?.level || 'N/A'}
                  </td>
                  <td>{aqi ?? '--'}</td>
                  <td>{pm2_5 ?? '--'}</td>
                  <td>{pm10 ?? '--'}</td>
                  <td>{temp ?? '--'}</td>
                  <td>{hum ?? '--'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {loading && (
          <div className="loading-overlay">
            <div className="spinner">Loading...</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CityStationsTable;
