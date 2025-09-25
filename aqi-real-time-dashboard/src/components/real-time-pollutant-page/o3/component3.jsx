import React from 'react';
import './Component3.css'; // <-- Import the new CSS file

const PM25SizeComparison = () => {
  return (
    <div className="pm25-size-comparison">
      <div className="relative">
        {/* Info box with red side border */}
        <div className="info-box">
          <p>
          Do you know that Ozone's total mass is 3 billion metric tons in the atmosphere? It looks like a lot but has only 0.00006% of the atmosphere. Besides, Ozone's peak concentration is 32 kilometres above the Earth's surface.
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
            src="/o3/c3.svg"
            alt="Component 3 Illustration"
          />
        </div>
      </div>
    </div>
  );
};

export default PM25SizeComparison;
