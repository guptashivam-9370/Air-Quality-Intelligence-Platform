import React from 'react';
import './Component3.css'; // <-- Import the new CSS file

const PM25SizeComparison = () => {
  return (
    <div className="pm25-size-comparison">
      <div className="relative">
        {/* Info box with red side border */}
        <div className="info-box">
          <p>
          Sulfur Dioxide (SO₂): Sulfur dioxide is the major component in acid rain as it easily dissolves in the water and forms sulfuric acid. And SO2 in acid rain can harm forests, crops, and aquatic ecosystems, change soil acidity, and deteriorate building materials and historical monuments.
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
            src="/so2/c3.svg"
            alt="Component 3 Illustration"
          />
        </div>
      </div>
    </div>
  );
};

export default PM25SizeComparison;
