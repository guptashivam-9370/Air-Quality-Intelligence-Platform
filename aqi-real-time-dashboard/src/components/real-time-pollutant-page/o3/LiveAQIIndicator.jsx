import React, { useState, useEffect } from 'react';

const LiveAQIIndicator = () => {
  const [isVisible, setIsVisible] = useState(true);
  
  // Create blinking effect
  useEffect(() => {
    const interval = setInterval(() => {
      setIsVisible(prev => !prev);
    }, 600); // Toggle every 800ms for a noticeable but not too fast blink
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      position: 'absolute',
      top: '10px',
      right: '10px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      zIndex: 10,
      padding: '6px 12px',
      backgroundColor: 'rgba(0, 0, 0, 0.6)',
      borderRadius: '4px'
    }}>
      <div style={{ 
        width: '12px', 
        height: '12px', 
        backgroundColor: '#ff3b30',
        borderRadius: '50%',
        boxShadow: '0 0 6px #ff3b30',
        opacity: isVisible ? 1 : 0.2,
        transition: 'opacity 0.2s ease-in-out'
      }} />
      <span style={{
        color: '#fff',
        fontSize: '0.9rem',
        fontWeight: 'bold',
        whiteSpace: 'nowrap'
      }}>
        Live O3
      </span>
    </div>
  );
};

export default LiveAQIIndicator;