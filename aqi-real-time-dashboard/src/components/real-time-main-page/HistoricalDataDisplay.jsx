import React from 'react';

// This component doesn't manage its own state anymore
// Instead it receives chartType and setChartType from the parent
const HistoricalDataDisplay = ({
  historicalData,
  selectedPollutant,
  setSelectedPollutant,
  chartData,
  chartOptions,
  chartType,  // Now received as prop
  setChartType // Now received as prop
}) => {
  return (
    <div className="historical-data-container">
      <div className="chart-controls-wrapper">
        {/* Chart type selector */}
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
      
      {/* Render appropriate chart based on selection */}
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
      
      {/* CSS styles */}
      <style jsx>{`
        .historical-data-container {
          background-color: #ffffff;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          padding: 20px;
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
          border-radius: 12px; /* increased curvature */
          padding: 8px 12px;
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
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* subtle shadow */
        }
        
        .chart-select:hover {
          border-color: #CBD5E0;
          background-color: #EDF2F7;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        
        .chart-select:focus {
          outline: none;
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

export default HistoricalDataDisplay;
