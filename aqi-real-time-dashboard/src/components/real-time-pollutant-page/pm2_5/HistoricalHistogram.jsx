import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const HistoricalHistogram = ({ historicalData, selectedPollutant, setSelectedPollutant }) => {
  // Define pollutant-specific thresholds and colors
  const pollutantThresholds = {
    aqi: [50, 100, 150, 200, 300], // AQI breakpoints
    pm2_5: [50, 100, 250, 350, 450], // PM2.5 in µg/m³
    pm10: [50, 100, 250, 350, 450], // PM10 in µg/m³
    o3: [50, 100, 162, 208, 748], // Ozone in ppb
    no2: [50, 100, 150, 200, 300], // NO2 in ppb
    so2: [40, 80, 380, 800, 1600], // SO2 in ppb
    co: [8330, 16670, 25000, 33330, 41670], // CO in ppm
    no: [40, 80, 200, 400, 600], // NO in ppb
    nh3: [200, 400, 800, 1200, 1600], // NH3 in ppb
  };

  const colors = ['#A8E05F', '#FDD74B', '#FB9B57', '#F66A67', '#A97ABC', '#A87383']; // Colors from Good to Hazardous

  // Get color based on value for the selected pollutant
  const getBarColor = (value) => {
    const thresholds = pollutantThresholds[selectedPollutant] || pollutantThresholds.aqi; // Default to AQI if unknown
    for (let i = 0; i < thresholds.length; i++) {
      if (value <= thresholds[i]) return colors[i];
    }
    return colors[colors.length - 1]; // Highest category
  };

  // Format data for Chart.js
  const prepareChartData = () => {
    if (!historicalData || !historicalData.labels || !historicalData[selectedPollutant]) {
      return {
        labels: [],
        datasets: [
          {
            data: [],
            backgroundColor: [],
            borderWidth: 0,
            borderRadius: 2,
            barThickness: 16,
          }
        ]
      };
    }

    return {
      labels: historicalData.labels,
      datasets: [
        {
          data: historicalData[selectedPollutant],
          backgroundColor: historicalData[selectedPollutant].map(value => getBarColor(value)),
          borderWidth: 0,
          borderRadius: 2,
          barThickness: 16,
        }
      ]
    };
  };

  // Chart data
  const chartData = prepareChartData();

  // Calculate min and max values for the selected pollutant over the last 24 hours
  let minValue = null;
  let maxValue = null;
  if (historicalData && historicalData[selectedPollutant] && historicalData[selectedPollutant].length > 0) {
    const dataArray = historicalData[selectedPollutant];
    minValue = Math.min(...dataArray);
    maxValue = Math.max(...dataArray);
  }

  return (
    <div className="historical-chart">
      <h2 className="chart-title">Historical Air Quality Data</h2>

      {/* <div className="chart-controls">
        <select
          value={selectedPollutant}
          onChange={(e) => setSelectedPollutant(e.target.value)}
          className="pollutant-select"
        >
          {Object.keys(pollutantThresholds).map((pollutant) => (
            <option key={pollutant} value={pollutant}>{pollutant.toUpperCase()}</option>
          ))}
        </select>
      </div> */}

      <div className="chart-container" style={{ position: 'relative' }}>
        {/* Display min/max stats if data is available */}
        {minValue !== null && maxValue !== null && (
          <div
            className="chart-stats"
            style={{
              position: 'absolute',
              top: '-60px',
              right: '20px',
              padding: '10px 15px',
              borderRadius: '6px',
              boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
              fontSize: '0.9rem',
              color: '#fff',
              fontWeight: 'bold',
              background: `linear-gradient(135deg, ${getBarColor(maxValue)}, ${getBarColor(minValue)})`,
            }}
          >
            <div>🔺 Max {selectedPollutant.toUpperCase()}: {maxValue}</div>
            <div>🔻 Min {selectedPollutant.toUpperCase()}: {minValue}</div>
          </div>
        )}

        {chartData.labels.length > 0 ? (
          <Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false }} height={300} />
        ) : (
          <div className="no-data">No data available</div>
        )}
      </div>
    </div>
  );
};

export default HistoricalHistogram;
