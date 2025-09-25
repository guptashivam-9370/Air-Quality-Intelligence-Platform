import React from 'react';
import './Component3.css'; // <-- Import the new CSS file

const PM25SizeComparison = () => {
  return (
    <div className="pm25-size-comparison">
      <div className="relative">
        {/* Info box with red side border */}
        <div className="info-box">
          <p>
            PM10 particles are with a diameter of 10 micrometers (µm) or smaller and human hair has a diameter of 50 to 70 (µm). It makes PM10 1/5 to 1/7 of the human hair size.
          </p>
          <div className="line-container">
            <svg width="200" height="30" viewBox="0 0 200 30">
              <path
                d="M10,15 Q50,5 100,15 T190,15"
                stroke="#f04d5d"
                strokeWidth="2"
                fill="none"
              />
            </svg>
          </div>
        </div>

        {/* SVG illustration replaced by an <img> tag */}
        <div className="illustration">
          <img
            src="/pm10/c3.svg"
            alt="Component 3 Illustration"
          />
        </div>
      </div>
    </div>
  );
};

export default PM25SizeComparison;
