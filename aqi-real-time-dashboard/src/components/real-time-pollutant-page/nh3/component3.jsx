import React from 'react';
import './Component3.css'; // <-- Import the new CSS file

const PM25SizeComparison = () => {
  return (
    <div className="pm25-size-comparison">
      <div className="relative">
        {/* Info box with red side border */}
        <div className="info-box">
          <p>
            PM2.5 particles, with a diameter of 2.5 micrometers or less,
            are approximately 30 times smaller than a human hair, making them
            a significant yet often unseen threat to air quality.
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
            src="/nh3/c3.svg"
            alt="Component 3 Illustration"
          />
        </div>
      </div>
    </div>
  );
};

export default PM25SizeComparison;
