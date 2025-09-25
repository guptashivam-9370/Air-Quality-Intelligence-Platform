import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';
import AQIMap from '../components/HomePage/AQIMap';
import './HomePage.css';

function AQIMapPage() {
  const location = useLocation();
  const initialIsDark = location.state?.isdark ?? true;
  const [isDark, setIsDark] = useState(initialIsDark);
  
  const toggleDarkMode = () => {
    setIsDark(!isDark);
  };
  
  // Apply dark mode class to the body
  useEffect(() => {
    document.body.classList.toggle('dark-mode', isDark);
    return () => {
      document.body.classList.remove('dark-mode');
    };
  }, [isDark]);

  return (
    <div className={`app ${isDark ? 'dark-mode' : ''}`}>
      <TopBar toggleDarkMode={toggleDarkMode} isDark={isDark} />
      
      <div className="content-container full-width-map">
        <AQIMap />
      </div>
      
      <Footer />
    </div>
  );
}

export default AQIMapPage;