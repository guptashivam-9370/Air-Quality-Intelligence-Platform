import React from 'react';
import { NavLink } from 'react-router-dom';
import './NavigationBar.css';

const NavigationBar = ({ isdark, pollutantData ,selectedStation,historicalData}) => {
  const navItems = [
    { name: 'AQI', path: '/' },
    { name: 'PM2.5', path: '/pollutant/pm2_5' },
    { name: 'PM10', path: '/pollutant/pm10' },
    { name: 'CO', path: '/pollutant/co' },
    { name: 'SO2', path: '/pollutant/so2' },
    { name: 'NO2', path: '/pollutant/no2' },
    { name: 'O3', path: '/pollutant/o3' },
    { name: 'NO', path: '/pollutant/no' },
    { name: 'NH3', path: '/pollutant/nh3' }
  ];

  return (
    <nav className="navigation-bar">
      <div className="nav-container">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            state={{ isdark: isdark, pollutantData: pollutantData ,selectedStation:selectedStation,historicalData:historicalData}}
            className={({ isActive }) =>
              isActive ? "nav-item active" : "nav-item"
            }
          >
            {item.name}
          </NavLink>
        ))}
      </div>
    </nav>
  );
};

export default NavigationBar;
