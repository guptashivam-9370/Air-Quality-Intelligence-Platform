import React from 'react';
import './Component3.css'; // <-- Import the new CSS file

const PM25SizeComparison = () => {
  return (
    <div className="pm25-size-comparison">
      <div className="relative">
        {/* Info box with red side border */}
        <div className="info-box">
          <p>
          Do you know CO emissions in the atmosphere can affect the greenhouse gas level? It is because CO emission causes global and climate changes in the atmosphere by increasing or changing the sea and land temperatures. It makes changes in ecosystems and storm activity and affects other weather events..
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
            src="/co/c3.svg"
            alt="Component 3 Illustration"
          />
        </div>
      </div>
    </div>
  );
};

export default PM25SizeComparison;
