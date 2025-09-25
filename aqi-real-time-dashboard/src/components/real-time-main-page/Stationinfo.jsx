import React from 'react';
import LiveAQIIndicator from './LiveAQIIndicator';
import NavigationBar from './NavigationBar';

const StationInfo = ({ selectedStation, isdark ,pollutantData,historicalData}) => {
  return (
    <>
      <NavigationBar isdark={isdark} pollutantData={pollutantData} selectedStation={selectedStation} historicalData={historicalData}/>
      <div className="station-info">
        {/* Live AQI indicator */}
        <LiveAQIIndicator />
        <h2>{selectedStation.name}</h2>
        <p>
          {selectedStation.city || 'Unknown'}, {selectedStation.state || 'Unknown'},{' '}
          {selectedStation.country || 'Unknown'}
        </p>
      </div>
    </>
  );
};

export default StationInfo;
