import React, { useState } from 'react';
import './CigaretteEquivalentCard.css';

const CigaretteEquivalentCard = ({ pm25Level, isDarkMode = false }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  // Calculate cigarette equivalents based on Berkeley Earth's rule of thumb
  const cigarettesPerDay = parseFloat((pm25Level / 22).toFixed(1));
  const cigarettesPerWeek = parseFloat((cigarettesPerDay * 7).toFixed(1));
  const cigarettesPerMonth = parseFloat((cigarettesPerDay * 30).toFixed(1));

  // Add dark-mode class if isDarkMode is true
  const cardClassName = `cigarette-equivalent-card ${isDarkMode ? 'dark-mode' : ''}`;

  return (
    <div className={cardClassName}>
      <div className="card-content">
        <div className="stats-container">
          <div className="daily-stats">
            <span className="daily-count">{cigarettesPerDay}</span>
            <span className="daily-label">Cigarettes</span>
            <span className="daily-label">per day</span>
          </div>

          <div className="cigarette-icon">
            {/* Simple SVG cigarette icon */}
            <svg width="200" height="120" viewBox="0 0 200 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="200" height="120" />
              <rect x="20" y="80" width="140" height="15" fill="white" stroke="black" strokeWidth="0.5" />
              <rect x="20" y="80" width="30" height="15" fill="#D4A76A" />
              <path d="M160 80 L175 78 Q180 85 175 92 L160 95 Z" fill="darkred" />
              <circle cx="172" cy="85" r="5" fill="gray" />
              <circle cx="170" cy="87" r="4" fill="lightgray" />
              <circle cx="175" cy="65" r="10" fill="gray" opacity="0.5" />
              <circle cx="180" cy="50" r="15" fill="gray" opacity="0.5" />
              <circle cx="185" cy="35" r="12" fill="gray" opacity="0.5" />
              <circle cx="190" cy="25" r="8" fill="gray" opacity="0.5" />
              <circle cx="192" cy="20" r="5" fill="gray" opacity="0.5" />
            </svg>
          </div>

          <div className="weekly-stats">
            <span className="period-label">Weekly</span>
            <div>
              <span className="period-count">{cigarettesPerWeek}</span>
              <span className="period-unit"> Cigarettes</span>
            </div>
          </div>

          <div className="monthly-stats">
            <span className="period-label">Monthly</span>
            <div>
              <span className="period-count">{cigarettesPerMonth}</span>
              <span className="period-unit"> Cigarettes</span>
            </div>
          </div>
        </div>

        <div className="card-footer">
          <p className="info-text">Breathing the air in this location is as harmful as smoking {cigarettesPerDay} cigarettes a day.</p>
          <div className="source-container">
            <span className="source-label">Source:</span>
            <span className="source-name">Berkeley Earth</span>
            <span
              className="info-icon"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
            >
              i
              {showTooltip && (
                <div className="tooltip">
                  <strong>Berkeley Earth</strong><br />
                  According to Berkeley Earth's rule of thumb, one cigarette per day is equivalent to 22 μg/m3 of PM2.5 level.
                </div>
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CigaretteEquivalentCard;