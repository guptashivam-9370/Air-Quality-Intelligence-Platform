import React, { useState, useEffect } from 'react';
import { AlertCircle, Loader } from 'lucide-react';
import './AQIMap.css';

const AQIMap = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapUrl, setMapUrl] = useState('');
  const [summaryData, setSummaryData] = useState(null);
  
  // Backend API endpoints
  const MAP_API_URL = 'http://127.0.0.1:5000/api/aqi-map';
  const DATA_API_URL = 'http://127.0.0.1:5000/api/aqi-data';
  
  useEffect(() => {
    // Check if the backend is available
    const checkBackend = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch summary data
        const dataResponse = await fetch(DATA_API_URL);
        if (dataResponse.ok) {
          const data = await dataResponse.json();
          setSummaryData(data.summary);
        }
        
        // Set map URL
        setMapUrl(MAP_API_URL);
        setLoading(false);
      } catch (err) {
        console.error('Error connecting to backend:', err);
        setError('Failed to connect to the AQI backend service');
        setLoading(false);
      }
    };
    
    checkBackend();
  }, []);
  
  // Function to get AQI category color
  const getAqiColor = (aqi) => {
    if (aqi <= 50) return '#00e400';      // Good
    if (aqi <= 100) return '#ffff00';     // Moderate
    if (aqi <= 150) return '#ff7e00';     // Unhealthy for Sensitive Groups
    if (aqi <= 200) return '#ff0000';     // Unhealthy
    if (aqi <= 300) return '#99004c';     // Very Unhealthy
    return '#7e0023';                     // Hazardous
  };
  
  // Function to get AQI category name
  const getAqiCategory = (aqi) => {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
  };
  
  return (
    <div className="flex flex-col w-full h-full map-container">
      <div className="mb-4">
        {/* <h2 className="text-2xl font-bold mb-2">India Air Quality Map</h2> */}
        {/* <p className="text-gray-600 dark:text-gray-300">
          Real-time air quality data across India. The map shows AQI (Air Quality Index) values.
        </p> */}
      </div>
      
      
      
      {/* Map container */}
      <div className="relative flex-grow rounded-lg overflow-hidden shadow-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700" style={{ minHeight: '70vh', width: '100%' }}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800 bg-opacity-75 dark:bg-opacity-75">
            <div className="flex flex-col items-center">
              <Loader className="w-10 h-10 text-blue-500 animate-spin" />
              <p className="mt-2 text-lg">Loading AQI data...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
            <div className="flex flex-col items-center text-center p-4">
              <AlertCircle className="w-12 h-12 text-red-500 mb-2" />
              <h3 className="text-xl font-bold mb-2">Error Loading Map</h3>
              <p className="text-gray-600 dark:text-gray-300">{error}</p>
              <button 
                onClick={() => window.location.reload()} 
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Retry
              </button>
            </div>
          </div>
        )}
        
        {!loading && !error && mapUrl && (
          <iframe 
            src={mapUrl}
            className="w-full h-full"
            style={{ height: '70vh', width: '100%', border: 'none' }}
            frameBorder="0"
            title="India AQI Map"
            loading="lazy"
          />
        )}
      </div>
      
      {/* Legend */}
      {/* <div className="mt-4 bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">AQI Legend</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#00e400' }}></div>
            <span>Good (0-50)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#ffff00' }}></div>
            <span>Moderate (51-100)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#ff7e00' }}></div>
            <span>Unhealthy for Sensitive Groups (101-150)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#ff0000' }}></div>
            <span>Unhealthy (151-200)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#99004c' }}></div>
            <span>Very Unhealthy (201-300)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: '#7e0023' }}></div>
            <span>Hazardous (301+)</span>
          </div>
        </div>
      </div> */}
    </div>
  );
};

export default AQIMap;