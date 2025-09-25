
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Header.css';

const Header = ({ searchTerm, filteredStations, handleSearchChange, handleStationSelect, handleRefresh, lastUpdated, isDark }) => {
  const navigate = useNavigate();
  const [searchInput, setSearchInput] = useState(searchTerm || '');

  // Update local state when prop changes
  useEffect(() => {
    setSearchInput(searchTerm);
  }, [searchTerm]);

  const navigateToHome = () => {
    navigate('/', {
      state: { isdark: isDark }
    });
  };

  const navigateToAQIMap = () => {
    navigate('/aqi-map', {
      state: { isdark: isDark }
    });
  };


  const navigateToWorldMap = () => {
    navigate('/world-map', {
      state: { isdark: isDark }
    });
  };

  const navigateToGlobe = () => {
    navigate('/globe', {
      state: { isdark: isDark }
    });
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    // If we have filtered stations and the user presses enter, select the first one
    if (filteredStations && filteredStations.length > 0) {
      const firstStation = filteredStations[0];
      navigate('/', { state: { selectedStation: firstStation, isdark: isDark } });
    }
  };

  const handleStationSelectAndNavigate = (station) => {
    // First handle the selection with the parent component's handler
    if (handleStationSelect) {
      handleStationSelect(station);
    }

    // Then navigate to home with the selected station
    navigate('/', { state: { selectedStation: station, isdark: isDark } });
  };

  const handleLocalSearchChange = (e) => {
    setSearchInput(e.target.value);
    if (handleSearchChange) {
      handleSearchChange(e);
    }
  };

  return (
    <header className="header">
      <div className="header-container">
        <div className="header-top">
          <div className="logo-section">
            <h1 onClick={navigateToHome} className="site-title">
              Real-Time Analysis
            </h1>
          </div>


          <div className="controls-section">
            <button className="refresh-button" onClick={handleRefresh}>
              Refresh Data
            </button>

            <div className="navsigation-buttons">
              <div className="navs-button" onClick={navigateToAQIMap} title="AQI Map">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z"></path>
                  <path d="M8 2v16"></path>
                  <path d="M16 6v16"></path>
                </svg>
                <span>AQI Map</span>
              </div>
              {/* World Map Button */}
              <div className="navs-button" onClick={navigateToWorldMap} title="World Map">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z"></path>
                  <path d="M8 2v16"></path>
                  <path d="M16 6v16"></path>
                </svg>
                <span>Map</span>
              </div>

              {/* 3D Globe Button */}
              <div className="navs-button" onClick={navigateToGlobe} title="3D Globe">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <ellipse cx="12" cy="12" rx="10" ry="4"></ellipse>
                  <path d="M2 12h20"></path>
                  <path d="M12 2v20"></path>
                </svg>
                <span>Globe</span>
              </div>
            </div>
          </div>
          {/* {lastUpdated && <p className="last-updated">Last updated: {lastUpdated}</p>} */}
        </div>

        <div className="search-wrapper">
          <form onSubmit={handleSearchSubmit} className="search-container">
            <input
              type="text"
              placeholder="Search stations by name, city, or state..."
              value={searchInput}
              onChange={handleLocalSearchChange}
              className="search-input"
            />
            <div className="search-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  stroke="currentColor"
                  fill="none"
                />
              </svg>
            </div>
          </form>

          {searchInput && (
            <div className="search-results">
              {filteredStations.slice(0, 10).map((station) => (
                <div
                  key={station.station_id || station.id}
                  className="search-result-item"
                  onClick={() => handleStationSelectAndNavigate(station)}
                >
                  <div className="station-name">{station.name}</div>
                  <div className="station-location">
                    {station.city || 'Unknown'}, {station.state || 'Unknown'}
                  </div>
                </div>
              ))}
              {filteredStations.length === 0 && (
                <div className="search-result-item no-results">No results found</div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;