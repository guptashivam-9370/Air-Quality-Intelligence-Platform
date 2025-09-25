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

// Import components
import TopBar from '../../components/real-time-main-page/TopBar';
import Header from '../../components/real-time-main-page/Header';
import StationInfo from '../../components/real-time-main-page/Stationinfo';
import MapView from '../../components/real-time-main-page/MapView';
import Dialer from '../../components/real-time-pollutant-page/pm2_5/Dialer';
import Component1 from '../../components/real-time-pollutant-page/pm2_5/component1';
import Component2 from '../../components/real-time-pollutant-page/pm2_5/component2';
import Component3 from '../../components/real-time-pollutant-page/pm2_5/component3';
import Component4 from '../../components/real-time-pollutant-page/pm2_5/component4';
import Component5 from '../../components/real-time-pollutant-page/pm2_5/component5';
import HistoricalChart from '../../components/real-time-pollutant-page/pm2_5/HistoricalChart';
import HistoricalHistogram from '../../components/real-time-pollutant-page/pm2_5/HistoricalHistogram';
import CigaretteEquivalentCard from '../../components/real-time-main-page/CigaretteEquivalentCard';
import Footer from '../../components/real-time-main-page/Footer';
import '../HomePage.css';

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

function PM25Page() {
  // Retrieve passed state from the location (if any)
  const location = useLocation();
  const initialIsDark = location.state?.isdark ?? true;
  const initialPM25Value =
    location.state?.pollutantValue || location.state?.pollutantData?.pm2_5 || null;
  const initialPollutantData = location.state?.pollutantData || {};
  const initialselectedStation = location.state?.selectedStation || null;
  const initialhistoricalData = location.state?.historicalData || null;
  // Removed initialPM25Value from location state and set PM2.5 initial value from passed state
  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState(initialselectedStation);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredStations, setFilteredStations] = useState([]);
  const [pm25Value, setPM25Value] = useState(initialPM25Value);
  const [historicalData, setHistoricalData] = useState(initialhistoricalData);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDark, setIsDark] = useState(initialIsDark);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [chartType, setChartType] = useState('histogram');
  const [pollutantData, setPollutantData] = useState(initialPollutantData);
  const [selectedPollutant, setSelectedPollutant] = useState('pm2_5');
  const [stationsLoaded, setStationsLoaded] = useState(false);

  // Toggle dark mode
  useEffect(() => {
    if (isDark) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [isDark]);

  const toggleDarkMode = () => setIsDark(!isDark);

  const pm25Categories = [
    {
      range: [0, 50],
      level: 'Good',
      color: 'linear-gradient(to right, #43e97b, #a8e05f)',
      healthImplications: 'Air quality is satisfactory, and PM2.5 pollution poses little or no risk.'
    },
    {
      range: [50, 100],
      level: 'Moderate',
      color: 'linear-gradient(to right, #a8e05f, #f6d365)',
      healthImplications: 'Air quality is acceptable. However, there may be a moderate health concern for a very small number of individuals who are unusually sensitive to air pollution.'
    },
    {
      range: [100, 250],
      level: 'Poor',
      color: 'linear-gradient(to right, #f6d365, #fa709a)',
      healthImplications: 'Air quality is poor. Prolonged exposure may cause discomfort or mild health issues for the general population.'
    },
    {
      range: [250, 350],
      level: 'Unhealthy',
      color: 'linear-gradient(to right, #fa709a, #ff0844)',
      healthImplications: 'Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects.'
    },
    {
      range: [350, 430],
      level: 'Severe',
      color: 'linear-gradient(to right, #ff0844, #6a11cb)',
      healthImplications: 'Health warnings of emergency conditions. The entire population is more likely to be affected.'
    },
    {
      range: [430, 510],
      level: 'Hazardous',
      color: 'linear-gradient(to right, #6a11cb, #7f0000)',
      healthImplications: 'Serious health effects for everyone. Avoid outdoor exposure and physical activity.'
    }
  ];

  // Get PM2.5 category based on PM2.5 value (in μg/m³)
  const getPM25Category = (pm25) => {
    for (const category of pm25Categories) {
      if (pm25 >= category.range[0] && pm25 <= category.range[1]) {
        return category;
      }
    }
    return pm25Categories[pm25Categories.length - 1]; // Return the highest category for very high values
  };

  // Helper function to fetch data for a given station (for detailed station info)
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
        temp, // Temperature in °C
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

        Papa.parse(csvData, {
          header: true,
          complete: (results) => {
            const validStations = results.data.filter(
              station =>
                station.station_id && station.latitude && station.longitude && station.name
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

  // When a station is selected, only fetch historical data.
  useEffect(() => {
    if (selectedStation) {
      // fetchHistoricalData();
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

  // Remove fetching current PM2.5 data from API.
  // The current PM2.5 value and pollutant data are taken from the passed state.
  // We still fetch historical data.
 

  const handleStationSelect = (station) => {
    setSelectedStation(station);
    setSearchTerm('');
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  // Refresh now only updates historical data and lastUpdated time
  const handleRefresh = () => {
    if (selectedStation) {
      
    }
  };

  const pm25Category = pm25Value ? getPM25Category(pm25Value) : null;

  // Chart data for historical display
  const chartData = {
    labels: historicalData.labels || [],
    datasets: [
      {
        label: selectedPollutant.toUpperCase().replace('_', '.'),
        data: historicalData[selectedPollutant] || [],
        fill: false,
        borderColor: '#4bc0c0',
        tension: 0.1
      }
    ]
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: `Past 24 Hours ${selectedPollutant.toUpperCase().replace('_', '.')} Levels`
      },
    },
    scales: {
      y: { beginAtZero: true }
    }
  };

  // Inline Historical Data Display component
  const HistoricalDataDisplay = ({
    historicalData,
    chartData,
    chartOptions,
    chartType,
    setChartType,
    selectedPollutant,
    setSelectedPollutant
  }) => {
    const pollutantOptions = [
      { label: 'PM2.5', value: 'pm2_5' },
      { label: 'PM10', value: 'pm10' },
      { label: 'CO', value: 'co' },
      { label: 'NO', value: 'no' },
      { label: 'NO2', value: 'no2' },
      { label: 'O3', value: 'o3' },
      { label: 'SO2', value: 'so2' },
      { label: 'NH3', value: 'nh3' }
    ];
    return (
      <div className="historical-data-container">
        <div className="chart-controls-wrapper">
          <div className="chart-type-dropdown">
            <label htmlFor="pollutantSelect" className="chart-type-label">Pollutant:</label>
            <select
              id="pollutantSelect"
              value={selectedPollutant}
              onChange={(e) => setSelectedPollutant(e.target.value)}
              className="chart-select"
            >
              {pollutantOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="chart-type-dropdown" style={{ marginLeft: '10px' }}>
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

  if (loading && !stationsLoaded) {
    return <div className="loading">Loading station data...</div>;
  }

  if (error && !selectedStation) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="app">
      {/* <button onClick={toggleDarkMode} className="dark-mode-button">
        {isDark ? 'Dark Mode' : 'Light Mode'}
      </button> */}

     <TopBar toggleDarkMode={toggleDarkMode} isDark={isDark}/>

      <Header
        searchTerm={searchTerm}
        filteredStations={filteredStations}
        handleSearchChange={handleSearchChange}
        handleStationSelect={handleStationSelect}
        handleRefresh={handleRefresh}
        lastUpdated={lastUpdated}
      />

      {selectedStation && (
        <div className="dashboard">
          <StationInfo selectedStation={selectedStation} isdark={isDark} pollutantData={pollutantData} historicalData={historicalData} />

          <div style={{ display: 'flex', width: '100%' }}>
            <div style={{ flex: '0 0 70%' }}>
              <Dialer currentAQI={pm25Value} aqiCategory={pm25Category} />
            </div>
            <div style={{ flex: '0 0 30%', marginLeft: '10px', paddingRight: '10px' }}>
              <MapView selectedStation={selectedStation} currentAQI={null} pm25Value={pm25Value} />
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <HistoricalDataDisplay
              historicalData={historicalData}
              chartData={chartData}
              chartOptions={chartOptions}
              chartType={chartType}
              setChartType={setChartType}
              selectedPollutant={selectedPollutant}
              setSelectedPollutant={setSelectedPollutant}
            />
          </div>

          <div style={{ marginTop: '20px' }}>
            <Component1 pm25Value={pm25Value} />
          </div>

          <div style={{ marginTop: '20px' }}>
            <Component2 />
          </div>

          <div style={{ marginTop: '20px' }}>
            <Component3 />
          </div>

          <div style={{ marginTop: '20px' }}>
            <Component4 />
          </div>

          <div style={{ marginTop: '20px' }}>
            <Component5 />
          </div>

          {pm25Value && (
            <CigaretteEquivalentCard pm25Level={pm25Value} />
          )}
        </div>
      )}

      {!selectedStation && (
        <div className="no-station">
          <p>Please select a station to view PM2.5 data.</p>
        </div>
      )}

      <Footer />
    </div>
  );
}

export default PM25Page;
