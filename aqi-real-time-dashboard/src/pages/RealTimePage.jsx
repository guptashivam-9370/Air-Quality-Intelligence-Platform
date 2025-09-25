// src/pages/Homepage.jsx
import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import Papa from 'papaparse';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import TopBar from '../components/real-time-main-page/TopBar';
import Header from '../components/real-time-main-page/Header';
import StationInfo from '../components/real-time-main-page/Stationinfo';
import AQIDisplay from '../components/real-time-main-page/AQIDisplay';
import MapView from '../components/real-time-main-page/MapView';
import PollutantCards from '../components/real-time-main-page/PollutantCards';
import HistoricalChart from '../components/real-time-main-page/HistoricalChart';
import HistoricalHistogram from '../components/real-time-main-page/HistoricalHistogram';
import CigaretteEquivalentCard from '../components/real-time-main-page/CigaretteEquivalentCard';
import Footer from '../components/real-time-main-page/Footer';
import HealthRisksComponent from '../components/real-time-main-page/HealthRisksComponent';
import CityStationsTable from '../components/real-time-main-page/CityStationsTable';

import './RealTimePage.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function RealTimePage() {
  // Retrieve passed state from the location (if any)
  const location = useLocation();
  const initialIsDark = location.state?.isdark ?? true;
  const initialSelectedStation = location.state?.selectedStation || {
    station_id: 361411,
    name: 'IITG, Guwahati - PCBA',
    latitude: 26.2028636,
    longitude: 91.70046436,
    city: 'Guwahati',
    state:'Assam',
    country: 'IN',
    };
  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState(initialSelectedStation);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredStations, setFilteredStations] = useState([]);
  const [currentAQI, setCurrentAQI] = useState(null);
  const [pollutantData, setPollutantData] = useState({});
  const [weatherData, setWeatherData] = useState({});
  const [historicalData, setHistoricalData] = useState({});
  const [selectedPollutant, setSelectedPollutant] = useState('aqi');
  const [loading, setLoading] = useState(initialIsDark);
  const [error, setError] = useState(null);
  const [isDark, setIsDark] = useState(initialIsDark);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [chartType, setChartType] = useState('histogram');
  const [stationsLoaded, setStationsLoaded] = useState(false);

  // Toggle dark mode and update body class
  useEffect(() => {
    if (isDark) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [isDark]);

  const toggleDarkMode = () => setIsDark(!isDark);

  // AQI categories and helper functions...
  const aqiCategories = [
    {
      range: [0, 50],
      level: 'Good',
      color: '#00e400',
      healthImplications:
        'Air quality is satisfactory, and air pollution poses little or no risk.',
    },
    {
      range: [51, 100],
      level: 'Moderate',
      color: '#ffff00',
      healthImplications:
        'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.',
    },
    {
      range: [101, 150],
      level: 'Unhealthy for Sensitive Groups',
      color: '#ff7e00',
      healthImplications:
        'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
    },
    {
      range: [151, 200],
      level: 'Unhealthy',
      color: '#ff0000',
      healthImplications:
        'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
    },
    {
      range: [201, 300],
      level: 'Very Unhealthy',
      color: '#8f3f97',
      healthImplications:
        'Health alert: The risk of health effects is increased for everyone.',
    },
    {
      range: [301, 500],
      level: 'Hazardous',
      color: '#7e0023',
      healthImplications:
        'Health warning of emergency conditions: everyone is more likely to be affected.',
    },
  ];

  const mapAQI = (concentration, c_low, c_high, i_low, i_high) => {
    return Math.round(
      ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    );
  };

  const calculateAQI = (pollutants) => {
    const pm25 = pollutants.pm2_5 || 0;
    const pm10 = pollutants.pm10 || 0;
    const o3 = pollutants.o3 || 0;
    const no2 = pollutants.no2 || 0;
    const so2 = pollutants.so2 || 0;
    const co = pollutants.co || 0;

    const pm25AQI = calculatePM25AQI(pm25);
    const pm10AQI = calculatePM10AQI(pm10);
    const o3AQI = calculateO3AQI(o3);
    const no2AQI = calculateNO2AQI(no2);
    const so2AQI = calculateSO2AQI(so2);
    const coAQI = calculateCOAQI(co);

    return Math.max(pm25AQI, pm10AQI, o3AQI, no2AQI, so2AQI, coAQI);
  };

  const calculatePM25AQI = (concentration) => {
    if (concentration <= 12) return mapAQI(concentration, 0, 12, 0, 50);
    if (concentration <= 35.4)
      return mapAQI(concentration, 12.1, 35.4, 51, 100);
    if (concentration <= 55.4)
      return mapAQI(concentration, 35.5, 55.4, 101, 150);
    if (concentration <= 150.4)
      return mapAQI(concentration, 55.5, 150.4, 151, 200);
    if (concentration <= 250.4)
      return mapAQI(concentration, 150.5, 250.4, 201, 300);
    if (concentration <= 500.4)
      return mapAQI(concentration, 250.5, 500.4, 301, 500);
    return 500;
  };

  const calculatePM10AQI = (concentration) => {
    if (concentration <= 54) return mapAQI(concentration, 0, 54, 0, 50);
    if (concentration <= 154) return mapAQI(concentration, 55, 154, 51, 100);
    if (concentration <= 254) return mapAQI(concentration, 155, 254, 101, 150);
    if (concentration <= 354) return mapAQI(concentration, 255, 354, 151, 200);
    if (concentration <= 424) return mapAQI(concentration, 355, 424, 201, 300);
    if (concentration <= 604) return mapAQI(concentration, 425, 604, 301, 500);
    return 500;
  };

  const calculateO3AQI = (concentration) => {
    const ppb = concentration * 0.5;
    if (ppb <= 54) return mapAQI(ppb, 0, 54, 0, 50);
    if (ppb <= 70) return mapAQI(ppb, 55, 70, 51, 100);
    if (ppb <= 85) return mapAQI(ppb, 71, 85, 101, 150);
    if (ppb <= 105) return mapAQI(ppb, 86, 105, 151, 200);
    if (ppb <= 200) return mapAQI(ppb, 106, 200, 201, 300);
    return 500;
  };

  const calculateNO2AQI = (concentration) => {
    const ppb = concentration * 0.53;
    if (ppb <= 53) return mapAQI(ppb, 0, 53, 0, 50);
    if (ppb <= 100) return mapAQI(ppb, 54, 100, 51, 100);
    if (ppb <= 360) return mapAQI(ppb, 101, 360, 101, 150);
    if (ppb <= 649) return mapAQI(ppb, 361, 649, 151, 200);
    if (ppb <= 1249) return mapAQI(ppb, 650, 1249, 201, 300);
    if (ppb <= 2049) return mapAQI(ppb, 1250, 2049, 301, 500);
    return 500;
  };

  const calculateSO2AQI = (concentration) => {
    const ppb = concentration * 0.38;
    if (ppb <= 35) return mapAQI(ppb, 0, 35, 0, 50);
    if (ppb <= 75) return mapAQI(ppb, 36, 75, 51, 100);
    if (ppb <= 185) return mapAQI(ppb, 76, 185, 101, 150);
    if (ppb <= 304) return mapAQI(ppb, 186, 304, 151, 200);
    if (ppb <= 604) return mapAQI(ppb, 305, 604, 201, 300);
    if (ppb <= 1004) return mapAQI(ppb, 605, 1004, 301, 500);
    return 500;
  };

  const calculateCOAQI = (concentration) => {
    const ppm = concentration * 0.000873;
    if (ppm <= 4.4) return mapAQI(ppm, 0, 4.4, 0, 50);
    if (ppm <= 9.4) return mapAQI(ppm, 4.5, 9.4, 51, 100);
    if (ppm <= 12.4) return mapAQI(ppm, 9.5, 12.4, 101, 150);
    if (ppm <= 15.4) return mapAQI(ppm, 12.4, 15.4, 151, 200);
    if (ppm <= 30.4) return mapAQI(ppm, 15.5, 30.4, 201, 300);
    if (ppm <= 50.4) return mapAQI(ppm, 30.5, 50.4, 301, 500);
    return 500;
  };

  const getAQICategory = (aqi) => {
    for (const category of aqiCategories) {
      if (aqi >= category.range[0] && aqi <= category.range[1]) {
        return category;
      }
    }
    return aqiCategories[aqiCategories.length - 1];
  };

  const fetchStationData = async (station) => {
    try {
      const API_KEY = 'aac405e628f9c30a047d3de13192a7f7';
      const airResponse = await axios.get(
        `https://api.openweathermap.org/data/2.5/air_pollution?lat=${station.latitude}&lon=${station.longitude}&appid=${API_KEY}`
      );
      const weatherResponse = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${station.latitude}&lon=${station.longitude}&appid=${API_KEY}&units=metric`
      );
      const pollutants = airResponse.data.list[0].components;
      const { temp, humidity } = weatherResponse.data.main;
      return {
        pm2_5: pollutants.pm2_5,
        pm10: pollutants.pm10,
        no2:pollutants.no2,
        o3: pollutants.o3,
        co: pollutants.co,
        so2: pollutants.so2,
        // nh3: pollutants.nh3,
        // no: pollutants.no,
        temp,         // Temperature in °C
        hum: humidity // Humidity percentage
      };
    } catch (error) {
      console.error('Error fetching station data:', error);
      return {};
    }
  };

  // Load station data from CSV
  useEffect(() => {
    const loadStations = async () => {
      try {
        const response = await fetch('/metadata.csv');
        const csvData = await response.text();

        console.log("CSV Data Preview:", csvData.substring(0, 200));

        Papa.parse(csvData, {
          header: true,
          complete: (results) => {
            const validStations = results.data.filter(
              station =>
                station.station_id && station.latitude && station.longitude && station.name && station.country && station.city && station.state 
            );
            setStations(validStations);
            setFilteredStations(validStations);

            setStationsLoaded(true);

            // Only set default station if none passed from another page
            if (!initialSelectedStation && validStations.length > 0) {
              setSelectedStation(validStations[0]);
            } else if (initialSelectedStation) {
              // Find the matching station in our loaded stations if possible
              const matchedStation = validStations.find(
                station => station.station_id === initialSelectedStation.station_id
              );

              // If we found a match, use the full station data
              if (matchedStation) {
                setSelectedStation(matchedStation);
              }
              // Otherwise keep using the passed station data
            }


            // if (validStations.length > 0) {
            //   setSelectedStation(validStations[452]);
            // }

            setLoading(false);
          },
          error: (error) => {
            console.error("CSV parsing error:", error);
            setError('Error parsing CSV: ' + error.message);
            setLoading(false);
          }
        });
      } catch (error) {
        console.error("Station data loading error:", error);
        setError('Error loading station data: ' + error.message);
        setLoading(false);
      }
    };

    loadStations();
  }, []);

  // Fetch data when a station is selected
  useEffect(() => {
    if (selectedStation) {
      fetchCurrentData();
      fetchHistoricalData();
    }
  }, [selectedStation]);

  // Filter stations based on search term
  useEffect(() => {
    if (searchTerm) {
      const filtered = stations.filter(station =>
        (station.name && station.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (station.city && station.city.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (station.state && station.state.toLowerCase().includes(searchTerm.toLowerCase()))
      );
      setFilteredStations(filtered);
    } else {
      setFilteredStations(stations);
    }
  }, [searchTerm, stations]);

  // Fetch current AQI data for selected station
  const fetchCurrentData = async () => {
    if (!selectedStation) return;
    setLoading(true);
    try {
      const API_KEY = 'aac405e628f9c30a047d3de13192a7f7';
      const response = await axios.get(
        `https://api.openweathermap.org/data/2.5/air_pollution?lat=${selectedStation.latitude}&lon=${selectedStation.longitude}&appid=${API_KEY}`
      );

      const weatherResponse = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${selectedStation.latitude}&lon=${selectedStation.longitude}&appid=${API_KEY}&units=metric`
      );
      const weatherdata = weatherResponse.data.main;
      setWeatherData(weatherdata)

      const pollutants = response.data.list[0].components;
      const aqi = calculateAQI(pollutants);
      setCurrentAQI(aqi);
      setPollutantData(pollutants);
      
      setLoading(false);
    } catch (error) {
      setError('Error fetching current data: ' + error.message);
      setLoading(false);
    }
  };

  // Fetch historical data for selected station
  const fetchHistoricalData = async () => {
    if (!selectedStation) return;
    setLoading(true);
    try {
      const API_KEY = 'aac405e628f9c30a047d3de13192a7f7';
      const now = Math.floor(Date.now() / 1000);
      const oneDayAgo = now - (24 * 60 * 60);
      const response = await axios.get(
        `https://api.openweathermap.org/data/2.5/air_pollution/history?lat=${selectedStation.latitude}&lon=${selectedStation.longitude}&start=${oneDayAgo}&end=${now}&appid=${API_KEY}`
      );
      const history = {
        labels: [],
        aqi: [],
        pm2_5: [],
        pm10: [],
        o3: [],
        no2: [],
        so2: [],
        co: [],
        no: [],
        nh3: []
      };

      response.data.list.forEach(item => {
        const datetime = new Date(item.dt * 1000);
        history.labels.push(datetime.toLocaleTimeString());
        const currentAQI = calculateAQI(item.components);
        history.aqi.push(currentAQI);
        history.pm2_5.push(item.components.pm2_5);
        history.pm10.push(item.components.pm10);
        history.o3.push(item.components.o3);
        history.no2.push(item.components.no2);
        history.so2.push(item.components.so2);
        history.co.push(item.components.co);
        history.no.push(item.components.no);
        history.nh3.push(item.components.nh3);
      });

      setHistoricalData(history);
      setLoading(false);
    } catch (error) {
      setError('Error fetching historical data: ' + error.message);
      setLoading(false);
    }
  };

  const handleStationSelect = (station) => {
    setSelectedStation(station);
    setSearchTerm('');
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleRefresh = () => {
    if (selectedStation) {
      fetchCurrentData();
      fetchHistoricalData();
      setLastUpdated(new Date().toLocaleString());
    }
  };

  const aqiCategory = currentAQI ? getAQICategory(currentAQI) : null;

  // Chart data for historical display
  const chartData = {
    labels: historicalData.labels || [],
    datasets: [
      {
        label: selectedPollutant === 'aqi' ? 'AQI' : selectedPollutant,
        data: historicalData[selectedPollutant] || [],
        fill: false,
        borderColor: selectedPollutant === 'aqi' && aqiCategory ? aqiCategory.color : '#4bc0c0',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: `Past 24 Hours ${selectedPollutant.toUpperCase()}`
      },
    },
    scales: {
      y: { beginAtZero: true }
    }
  };
  // console.log(setSelectedStation)
  // Inline Historical Data Display component
  const HistoricalDataDisplay = ({
    historicalData,
    selectedPollutant,
    setSelectedPollutant,
    chartData,
    chartOptions,
    chartType,
    setChartType
  }) => {
    return (
      <div className="historical-data-container">
        <div className="chart-controls-wrapper">
          <div className="chart-type-dropdown">
            <label htmlFor="chartTypeSelect" className="chart-type-label">Chart Type:</label>
            <select
              id="chartTypeSelect"
              value={chartType}
              onChange={(e) => setChartType(e.target.value)}
              className="chart-select"
            >
              <option value="line">Line Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </div>
        </div>
        {chartType === 'line' ? (
          <HistoricalChart
            historicalData={historicalData}
            selectedPollutant={selectedPollutant}
            setSelectedPollutant={setSelectedPollutant}
            chartData={chartData}
            chartOptions={chartOptions}
          />
        ) : (
          <HistoricalHistogram
            historicalData={{
              labels: historicalData.labels || [],
              [selectedPollutant]: historicalData[selectedPollutant] || []
            }}
            selectedPollutant={selectedPollutant}
            setSelectedPollutant={setSelectedPollutant}
          />
        )}
        <style jsx>{`
          .historical-data-container {
            background-color: #ffffff;
            border-radius: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            padding: 24px;
            margin-bottom: 24px;
          }
          .chart-controls-wrapper {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            margin-bottom: 16px;
          }
          .chart-type-dropdown {
            display: flex;
            align-items: center;
          }
          .chart-type-label {
            font-size: 14px;
            font-weight: 500;
            color: #4A5568;
            margin-right: 8px;
          }
          .chart-select {
            background-color: #F7FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 35px;
            padding: 13px 14px;
            font-size: 14px;
            color: #2D3748;
            cursor: pointer;
            min-width: 140px;
            transition: all 0.2s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%234A5568' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            padding-right: 32px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
          }
          .chart-select:hover {
            border-color: #CBD5E0;
            background-color: #EDF2F7;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
          }
          .chart-select:focus {
            outline: none;
            border-radius: 25px;
            border-color: #4299E1;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.3);
          }
          @media (max-width: 640px) {
            .chart-type-dropdown {
              width: 100%;
            }
            .chart-select {
              width: 100%;
            }
          }
        `}</style>
      </div>
    );
  };

  if (loading && !selectedStation) {
    return <div className="loading">Loading station data...</div>;
  }

  if (error && !selectedStation) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="app">
      {/* <div>
        <button onClick={toggleDarkMode} className="dark-mode-button">
          {isDark ? 'Dark Mode' : 'Light Mode'}
        </button>
      </div> */}

      <TopBar toggleDarkMode={toggleDarkMode} isDark={isDark} />


      <Header
        searchTerm={searchTerm}
        filteredStations={filteredStations}
        handleSearchChange={handleSearchChange}
        handleStationSelect={handleStationSelect}
        handleRefresh={handleRefresh}
        lastUpdated={lastUpdated}
        isDark={isDark}
      />
      {selectedStation && (
        <div className="dashboard">
          <StationInfo selectedStation={selectedStation} isdark={isDark} pollutantData={pollutantData} historicalData={historicalData}/>

          <div style={{ display: 'flex', width: '100%' }}>
            <div style={{ flex: '0 0 70%' }}>
              <AQIDisplay currentAQI={currentAQI} aqiCategory={aqiCategory} weatherData={weatherData} />
            </div>
            <div style={{ flex: '0 0 30%', marginLeft: '10px', paddingRight: '10px' }}>
              <MapView selectedStation={selectedStation} currentAQI={currentAQI} />
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <PollutantCards pollutantData={pollutantData} isdark={isDark} selectedStation={selectedStation} historicalData={historicalData}/>
          </div>

          <div style={{ marginTop: '20px' }}>
            <HistoricalDataDisplay
              historicalData={historicalData}
              selectedPollutant={selectedPollutant}
              setSelectedPollutant={setSelectedPollutant}
              chartData={chartData}
              chartOptions={chartOptions}
              chartType={chartType}
              setChartType={setChartType}
            />
          </div>

          <div style={{ marginTop: '-30px' }}>
            {selectedStation && (
              <CityStationsTable
                selectedStation={selectedStation}
                stations={stations}
                calculateAQI={calculateAQI}
                getAQICategory={getAQICategory}
                fetchStationData={fetchStationData}
              />
            )}
          </div>

          {pollutantData.pm2_5 && (
            <CigaretteEquivalentCard pm25Level={pollutantData.pm2_5} />
          )}

          <div style={{ marginTop: '20px' }}>
            <HealthRisksComponent currentAQI={currentAQI} />
          </div>
        </div>
      )}

      {!selectedStation && (
        <div className="no-station">
          <p>Please select a station to view AQI data.</p>
        </div>
      )}

      <Footer />
    </div>
  );
}

export default RealTimePage;
