import React from 'react';
import './Component1.css'; // Import the CSS file

const Component1 = ({ pm25Value }) => {
  // Use 58 as default if no pm25Value is provided
  const actualPm25Value = pm25Value || 58;
  const whoGuideline = 15.27;

  // Calculate how many times above WHO guideline (with 1 decimal place)
  const timesAboveWHO = (actualPm25Value / whoGuideline).toFixed(1);

  // Determine health risk level based on SO₂ value
  const getHealthRiskLevel = (value) => {
    if (value <= 40) return { level: "Good", color: "#4CAF50" };
    if (value <= 80) return { level: "Moderate", color: "#FFC107" };
    if (value <= 380) return { level: "Poor", color: "#FF9800" };
    if (value <= 800) return { level: "Unhealthy", color: "#F44336" };
    if (value <= 1600) return { level: "Severe", color: "#9C27B0" };
    return { level: "Hazardous", color: "#7D0000" };
  };



  const risk = getHealthRiskLevel(actualPm25Value);

  // Dynamic status message based on SO₂ value
  const getStatusMessage = (value) => {
    if (value <= 40)
      return "which is within safe SO₂ limits, but sensitive individuals should be cautious.";
    if (value <= 80)
      return "indicating moderate SO₂ levels; sensitive groups should consider reducing outdoor activities.";
    if (value <= 380)
      return "indicating poor SO₂ levels; everyone should take precautions.";
    if (value <= 800)
      return "indicating unhealthy SO₂ levels; avoid outdoor activities.";
    if (value <= 1600)
      return "indicating very unhealthy SO₂ levels; everyone should avoid outdoor exposure.";
    return "indicating hazardous SO₂ levels; stay indoors with windows closed.";
  };

  const statusMessage = getStatusMessage(actualPm25Value);

  return (
    <div className="pm25-containers">
      {/* Header section */}
      <h1 className="pm25-title">What is the Current SO2 Level?</h1>


      <div className="content-container">
        {/* Left side with image */}
        {/* Left side with image */}
        <div className="image-container-1">
          <img
            src="/so2/component1.png"
            alt="City pollution illustration showing buildings, car emitting PM2.5, and a tree"
            className="pollution-image"
          />
        </div>
        {/* Right side with information */}
        <div className="info-container">
          <div className="status-message">
            <p>
              The current SO2 level here is{' '}
              <span className="pm25-value">{actualPm25Value} µg/m³</span> {statusMessage}
            </p>
          </div>

          {/* Red box with WHO comparison */}
          <div className="who-comparison">
            <div className="who-badge" style={{
              backgroundColor: timesAboveWHO > 1 ? '#dc2626' : '#4CAF50', // red if >1, green otherwise
            }}>
              <span className="times-above">{timesAboveWHO}x<br />Above</span>
            </div>
            <div className="who-info" >
              <p>
                The current SO2 level here is {timesAboveWHO}x Above the recommended WHO guideline of{' '}
                <span className="who-value">{whoGuideline} µg/m³</span>.
              </p>
            </div>
          </div>

          {/* Health risk level */}
          <div className="risk-container">
            <div className="risk-badge" style={{ backgroundColor: risk.color }}>
              <p>
                Health Risk: {risk.level}
              </p>
            </div>
          </div>

          {/* Recommendations based on level */}
          <div className="recommendations">
            <h3>Recommendations:</h3>
            {actualPm25Value > 55 && (
              <ul>
                <li>Wear N95 masks outdoors</li>
                <li>Use air purifiers indoors</li>
                <li>Limit outdoor activities</li>
                <li>Keep windows closed</li>
              </ul>
            )}
            {actualPm25Value <= 55 && actualPm25Value > 35 && (
              <ul>
                <li>Sensitive groups should reduce outdoor activities</li>
                <li>Consider using air purifiers</li>
              </ul>
            )}
            {actualPm25Value <= 35 && (
              <ul>
                <li>Air quality is acceptable</li>
                <li>Sensitive individuals should monitor conditions</li>
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Component1;