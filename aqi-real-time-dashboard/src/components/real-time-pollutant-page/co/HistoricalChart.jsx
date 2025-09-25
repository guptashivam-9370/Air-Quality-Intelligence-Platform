import React, { useRef } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const HistoricalChart = ({ historicalData, selectedPollutant, setSelectedPollutant }) => {
  const chartRef = useRef(null);
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
  const colors = ['#A8E05F', '#FDD74B', '#FB9B57', '#F66A67', '#A97ABC', '#A87383'];

  // Function to get color based on value
  const getLineColor = (value) => {
    const thresholds = pollutantThresholds[selectedPollutant] || pollutantThresholds.aqi;
    for (let i = 0; i < thresholds.length; i++) {
      if (value <= thresholds[i]) return colors[i];
    }
    return colors[colors.length - 1];
  };

  // Generate dataset with dynamic colors
  const prepareChartData = () => {
    if (!historicalData || !historicalData.labels || !historicalData[selectedPollutant]) {
      return { labels: [], datasets: [{ data: [], borderColor: '#ccc', pointBackgroundColor: '#ccc', tension: 0.3 }] };
    }

    const values = historicalData[selectedPollutant];

    return {
      labels: historicalData.labels,
      datasets: [
        {
          data: values,
          borderColor: values.map(getLineColor),
          pointBackgroundColor: values.map(getLineColor),
          pointBorderColor: values.map(getLineColor),
          pointRadius: 4,
          tension: 0.3,
          borderWidth: 2,
        }
      ]
    };
  };

  // Calculate min and max values
  let minValue = null;
  let maxValue = null;
  if (historicalData && historicalData[selectedPollutant] && historicalData[selectedPollutant].length > 0) {
    const dataArray = historicalData[selectedPollutant];
    minValue = Math.min(...dataArray);
    maxValue = Math.max(...dataArray);
  }

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => `${selectedPollutant.toUpperCase()}: ${context.raw.toFixed(1)}`
        }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#8E9BAE', font: { size: 11 } } },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(142, 155, 174, 0.15)' },
        ticks: { color: '#8E9BAE', font: { size: 11 }, stepSize: 20 },
        title: { display: true, text: selectedPollutant.toUpperCase(), color: '#8E9BAE', font: { size: 12 } }
      }
    }
  };

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
        {/* Display Min/Max Stats */}
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
              background: `linear-gradient(135deg, ${getLineColor(maxValue)}, ${getLineColor(minValue)})`,
            }}
          >
            <div>🔺 Max {selectedPollutant.toUpperCase()}: {maxValue}</div>
            <div>🔻 Min {selectedPollutant.toUpperCase()}: {minValue}</div>
          </div>
        )}

        {historicalData.labels && historicalData.labels.length > 0 ? (
          <Line ref={chartRef} data={prepareChartData()} options={chartOptions} />
        ) : (
          <div className="no-data">No historical data available</div>
        )}
      </div>
    </div>
  );
};

export default HistoricalChart;
