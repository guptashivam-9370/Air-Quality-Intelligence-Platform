// import React from 'react';
// import { useNavigate, useLocation } from 'react-router-dom';

// function TopBar({ isDark, toggleDarkMode, navigateToHome = '/' }) {
//   const navigate = useNavigate();
//   const location = useLocation();
//   const currentPath = location.pathname;

//   const handleNavigation = (path) => {
//     navigate(path, {
//       state: { isdark: isDark }
//     });
//   };

//   // Determine active button based on current path
//   const isActive = (path) => {
//     return currentPath === path;
//   };

//   // Inline styles for the AQI logo
//   const logoContainerStyle = {
//     display: 'flex',
//     alignItems: 'center',
//     cursor: 'pointer',
//     marginRight: '1rem',
//   };

//   const logoStyle = {
//     fontSize: '3.5rem',
//     fontWeight: 'bold',
//     margin: 0,
//     lineHeight: 1,
//   };

//   const logoBlueStyle = { color: '#63B3ED' };
//   const logoGreenStyle = { color: '#68D391' };
//   const logoYellowStyle = { color: '#F6E05E' };

//   const taglineStyle = {
//     color: '#1e79b3',
//     fontSize: '1rem',
//     margin: 0,
//     lineHeight: 1.2,
//   };

//   return (
//     <div className="nav-controls">
//       {/* AQI Logo Block */}
//       <div style={logoContainerStyle} onClick={() => handleNavigation('/')}>
//         <div style={logoStyle}>
//           <span style={logoBlueStyle}>A</span>
//           <span style={logoGreenStyle}>Q</span>
//           <span style={logoYellowStyle}>I</span>
//         </div>
//         <div style={{ marginLeft: '0.5rem' }}>
//           <p style={taglineStyle}>
//             <b>Air Quality Intelligence</b> <br /> Platform
//           </p>
//         </div>
//       </div>

//       <div className="nav-buttons">
//         <button
//           className={`nav-button home-btn ${isActive('/') ? 'active' : ''}`}
//           onClick={() => handleNavigation('/')}>
//           <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
//             <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
//           </svg>
//           Home
//         </button>
//         <button
//           className={`nav-button realtime-btn ${isActive('/realtime') ? 'active' : ''}`}
//           onClick={() => handleNavigation('/')}>
//           <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
//             <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.5-11a.5.5 0 00-1 0v4a.5.5 0 00.146.354l2.5 2.5a.5.5 0 10.708-.708L10.5 10.293V7z" clipRule="evenodd" />
//           </svg>
//           Real-time Analysis
//         </button>
//         <button 
//           className={`nav-button historical-btn ${isActive('/historical') ? 'active' : ''}`}
//           onClick={() => handleNavigation('/')}>
//           <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
//             <path d="M3 3a1 1 0 011-1h1a1 1 0 011 1v13a1 1 0 11-2 0V4H4a1 1 0 01-1-1zm5 4a1 1 0 011-1h1a1 1 0 011 1v9a1 1 0 11-2 0V7H9a1 1 0 01-1-1zm5-3a1 1 0 011-1h1a1 1 0 011 1v12a1 1 0 11-2 0V4h-1a1 1 0 01-1-1z" />
//           </svg>
//           Historical Analysis
//         </button>
//         <button
//           className={`nav-button prediction-btn ${isActive('/prediction') ? 'active' : ''}`}
//           onClick={() => handleNavigation('/')}
//         >
//           <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
//             <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
//             <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
//           </svg>
//           Prediction Model
//         </button>
//         <button
//           className={`nav-button chatbot-btn ${isActive('/aqi-chatbot') ? 'active' : ''}`}
//           onClick={() => handleNavigation('/aqi-chatbot')}
//         >
//           <svg
//             xmlns="http://www.w3.org/2000/svg"
//             className="nav-icon"
//             viewBox="0 0 20 20"
//             fill="currentColor"
//           >
//             <path
//               fillRule="evenodd"
//               d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z"
//               clipRule="evenodd"
//             />
//           </svg>
//           AQI Chatbot
//         </button>
//       </div>
//       <div>
//         <button onClick={toggleDarkMode} className="dark-mode-button">
//           {isDark ? (
//             <svg xmlns="http://www.w3.org/2000/svg" className="mode-icon" viewBox="0 0 20 20" fill="currentColor">
//               <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
//             </svg>
//           ) : (
//             <svg xmlns="http://www.w3.org/2000/svg" className="mode-icon" viewBox="0 0 20 20" fill="currentColor">
//               <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
//             </svg>
//           )}
//           {isDark ? 'Dark Mode' : 'Light Mode'}
//         </button>
//       </div>

//       {/* CSS for the navigation controls */}
//       <style jsx>{`
//         .nav-controls {
//           display: flex;
//           justify-content: space-between;
//           align-items: center;
//           margin-bottom: 1rem;
//           flex-wrap: wrap;
//           gap: 1rem;
//           padding: 0.75rem;
//           border-radius: 0.5rem;
//           background-color: #f7fafc;
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
//         }

//         .dark-mode .nav-controls {
//           background-color: #2d3748;
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
//         }

//         .nav-buttons {
//           display: flex;
//           gap: 0.75rem;
//           flex-wrap: wrap;
//         }

//         .nav-button {
//           display: flex;
//           align-items: center;
//           justify-content: center;
//           background-color: #3182ce;
//           color: white;
//           border: none;
//           border-radius: 0.5rem;
//           padding: 0.625rem 1rem;
//           font-size: 0.875rem;
//           font-weight: 600;
//           cursor: pointer;
//           transition: all 0.2s ease;
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
//           letter-spacing: 0.025em;
//         }

//         .nav-icon, .mode-icon {
//           width: 1rem;
//           height: 1rem;
//           margin-right: 0.375rem;
//         }

//         .nav-button:hover {
//           background-color: #2c5282;
//           transform: translateY(-2px);
//           box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
//         }

//         .nav-button:active {
//           transform: translateY(0);
//           box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
//         }

//         /* Active button styles */
//         .nav-button.active {
//           transform: translateY(-3px);
//           box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
//           position: relative;
//           font-weight: 700;
//           border: 2px solid white;
//         }

//         .dark-mode .nav-button.active {
//           border: 2px solid #1a202c;
//         }

//         .nav-button.active::after {
//           content: '';
//           position: absolute;
//           bottom: -8px;
//           left: 50%;
//           transform: translateX(-50%);
//           width: 8px;
//           height: 8px;
//           background-color: white;
//           border-radius: 50%;
//         }

//         .dark-mode .nav-button.active::after {
//           background-color: #1a202c;
//         }

//         /* Button-specific colors */
//         .home-btn {
//           background-color: #38b2ac;
//           background-image: linear-gradient(135deg, #38b2ac, #319795);
//         }

//         .realtime-btn {
//           background-color: #4299e1;
//           background-image: linear-gradient(135deg, #4299e1, #3182ce); 
//         }

//         .historical-btn {
//           background-color: #805ad5;
//           background-image: linear-gradient(135deg, #805ad5, #6b46c1);
//         }

//         .prediction-btn {
//           background-color: #ed8936;
//           background-image: linear-gradient(135deg, #ed8936, #dd6b20);
//         }

//         .chatbot-btn {
//           background-color: #48bb78;
//           background-image: linear-gradient(135deg, #48bb78, #38a169);
//         }

//         .home-btn:hover, .home-btn.active {
//           background-image: linear-gradient(135deg, #319795, #2c7a7b);
//         }

//         .realtime-btn:hover, .realtime-btn.active {
//           background-image: linear-gradient(135deg, #3182ce, #2c5282);
//         }

//         .historical-btn:hover, .historical-btn.active {
//           background-image: linear-gradient(135deg, #6b46c1, #553c9a);
//         }

//         .prediction-btn:hover, .prediction-btn.active {
//           background-image: linear-gradient(135deg, #dd6b20, #c05621);
//         }

//         .chatbot-btn:hover, .chatbot-btn.active {
//           background-image: linear-gradient(135deg, #38a169, #2f855a);
//         }

//         .dark-mode-button {
//           display: flex;
//           align-items: center;
//           background-color: #4a5568;
//           color: white;
//           border: none;
//           border-radius: 0.5rem;
//           padding: 0.625rem 1rem;
//           font-size: 0.875rem;
//           font-weight: 600;
//           cursor: pointer;
//           transition: all 0.2s ease;
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
//         }

//         .dark-mode-button:hover {
//           background-color: #2d3748;
//           transform: translateY(-2px);
//           box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
//         }

//         .dark-mode-button:active {
//           transform: translateY(0);
//           box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
//         }

//         /* Dark mode styles */
//         .dark-mode .nav-button {
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
//         }

//         .dark-mode .home-btn {
//           background-image: linear-gradient(135deg, #2c7a7b, #285e61);
//         }

//         .dark-mode .realtime-btn {
//           background-image: linear-gradient(135deg, #2b6cb0, #2c5282);
//         }

//         .dark-mode .historical-btn {
//           background-image: linear-gradient(135deg, #553c9a, #44337a);
//         }

//         .dark-mode .prediction-btn {
//           background-image: linear-gradient(135deg, #c05621, #9c4221);
//         }

//         .dark-mode .chatbot-btn {
//           background-image: linear-gradient(135deg, #38a169, #2f855a);
//         }

//         .dark-mode .home-btn:hover, .dark-mode .home-btn.active {
//           background-image: linear-gradient(135deg, #2f855a, #276749);
//         }

//         .dark-mode .realtime-btn:hover, .dark-mode .realtime-btn.active {
//           background-image: linear-gradient(135deg,rgb(44, 84, 112),rgb(35, 59, 82));
//         }

//         .dark-mode .historical-btn:hover, .dark-mode .historical-btn.active {
//           background-image: linear-gradient(135deg, #44337a, #322659);
//         }

//         .dark-mode .prediction-btn:hover, .dark-mode .prediction-btn.active {
//           background-image: linear-gradient(135deg, #9c4221, #7b341e);
//         }

//         .dark-mode .chatbot-btn:hover, .dark-mode .chatbot-btn.active {
//           background-image: linear-gradient(135deg, #276749, #1e4e3b);
//         }

//         .dark-mode .dark-mode-button {
//           background-color: #1a202c;
//           color: #fff;
//           box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
//         }

//         .dark-mode .dark-mode-button:hover {
//           background-color: #171923;
//         }

//         @media (max-width: 768px) {
//           .nav-controls {
//             flex-direction: column;
//             align-items: stretch;
//           }

//           .nav-buttons {
//             justify-content: center;
//           }

//           .dark-mode-button {
//             margin-top: 0.5rem;
//             justify-content: center;
//           }
//         }
//       `}</style>
//     </div>
//   );
// }

// export default TopBar;


import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

function TopBar({ isDark, toggleDarkMode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const currentPath = location.pathname;
  const [menuOpen, setMenuOpen] = useState(false);

  const handleNavigation = (path) => {
    navigate(path, { state: { isdark: isDark } });
    setMenuOpen(false); // Close menu after navigating
  };

  // Determine active button based on current path
  const isActive = (path) => currentPath === path;

  // Inline styles for the AQI logo block
  const logoContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    marginRight: '1rem',
  };

  const logoStyle = {
    fontSize: '3.5rem',
    fontWeight: 'bold',
    margin: 0,
    lineHeight: 1,
  };

  const logoBlueStyle = { color: '#63B3ED' };
  const logoGreenStyle = { color: '#68D391' };
  const logoYellowStyle = { color: '#F6E05E' };

  const taglineStyle = {
    color: '#1e79b3',
    fontSize: '1rem',
    margin: 0,
    lineHeight: 1.2,
  };

  return (
    <div className="nav-controls">
      {/* Left Block: Logo */}
      <div style={logoContainerStyle} onClick={() => handleNavigation('/')}>
        <div style={logoStyle}>
          <span style={logoBlueStyle}>A</span>
          <span style={logoGreenStyle}>Q</span>
          <span style={logoYellowStyle}>I</span>
        </div>
        <div style={{ marginLeft: '0.5rem' }}>
          <p style={taglineStyle}>
            <b>Air Quality Intelligence</b> <br /> Platform
          </p>
        </div>
      </div>

      {/* Center Block: Navigation Buttons */}
      <div className="nav-menu-container">
        <div className={`nav-buttons ${menuOpen ? 'open' : ''}`}>
          {/* <button
            className={`nav-button home-btn ${isActive('/') ? 'active' : ''}`}
            onClick={() => handleNavigation('/')}>
            <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
            </svg>
            Home
          </button> */}
          <button
            className={`nav-button realtime-btn ${isActive('/') ? 'active' : ''}`}
            onClick={() => handleNavigation('/')}>
            <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.5-11a.5.5 0 00-1 0v4a.5.5 0 00.146.354l2.5 2.5a.5.5 0 10.708-.708L10.5 10.293V7z" clipRule="evenodd" />
            </svg>
            Real-time Analysis
          </button>
          <button
            className={`nav-button prediction-btn ${isActive('/prediction') ? 'active' : ''}`}
            onClick={() => handleNavigation('/prediction')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
              <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
              <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
            </svg>
            Prediction Model
          </button>
          <button
            className={`nav-button chatbot-btn ${isActive('/aqi-chatbot') ? 'active' : ''}`}
            onClick={() => handleNavigation('/aqi-chatbot')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z"
                clipRule="evenodd"
              />
            </svg>
            AQI Chatbot
          </button>

          <button
            className={`nav-button about-us-btn ${isActive('/about-us') ? 'active' : ''}`}
            onClick={() => handleNavigation('/about-us')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
              <circle cx="9" cy="7" r="4"></circle>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            About Us
          </button>
        </div>
      </div>

      {/* Right Block: Dark Mode Toggle and Hamburger Menu */}
      <div className="right-controls">
        <button onClick={toggleDarkMode} className="dark-mode-icon">
          {isDark ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="mode-icon" viewBox="0 0 20 20" fill="currentColor">
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="mode-icon" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
            </svg>
          )}
          <span className="dark-mode-text">
            {isDark ? 'Dark Mode' : 'Light Mode'}
          </span>
        </button>
        <div className="hamburger-container" onClick={() => setMenuOpen(!menuOpen)}>
          <svg xmlns="http://www.w3.org/2000/svg" className="hamburger-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </div>
      </div>

      {/* CSS styles */}
      <style jsx>{`
        .nav-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
          flex-wrap: wrap;
          gap: 1rem;
          padding: 0.75rem;
          border-radius: 0.5rem;
          background-color: #f7fafc;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .dark-mode .nav-controls {
          background-color: #2d3748;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        /* Navigation menu container */
        .nav-menu-container {
          position: relative;
          flex-grow: 1;
          display: flex;
          justify-content: center;
        }
        
        .nav-buttons {
          display: flex;
          gap: 0.75rem;
          flex-wrap: wrap;
        }
        
        /* Mobile styles for nav buttons */
        @media (max-width: 1200px) {
          .nav-buttons {
            display: ${menuOpen ? 'flex' : 'none'};
            flex-direction: column;
            position: absolute;
            top: 100%;
            left: 0;
            background: #f7fafc;
            width: 100%;
            padding: 0.5rem 0;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 10;
          }
          
          .dark-mode .nav-buttons {
            background: #2d3748;
          }
        }
        
        .nav-button {
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: #3182ce;
          color: white;
          border: none;
          border-radius: 0.5rem;
          padding: 0.625rem 1rem;
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          letter-spacing: 0.025em;
        }
        
        .nav-icon, .mode-icon {
          width: 1rem;
          height: 1rem;
          margin-right: 0.375rem;
        }
        
        .nav-button:hover {
          background-color: #2c5282;
          transform: translateY(-2px);
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }
        
        .nav-button:active {
          transform: translateY(0);
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .nav-button.active {
          transform: translateY(-3px);
          box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
          position: relative;
          font-weight: 700;
          border: 2px solid white;
        }
        
        .dark-mode .nav-button.active {
          border: 2px solid #1a202c;
        }
        
        .nav-button.active::after {
          content: '';
          position: absolute;
          bottom: -8px;
          left: 50%;
          transform: translateX(-50%);
          width: 8px;
          height: 8px;
          background-color: white;
          border-radius: 50%;
        }
        
        .dark-mode .nav-button.active::after {
          background-color: #1a202c;
        }
        
        /* Button-specific colors */
        .home-btn {
          background-color: #38b2ac;
          background-image: linear-gradient(135deg, #38b2ac, #319795);
        }
        
        .realtime-btn {
          background-color: #4299e1;
          background-image: linear-gradient(135deg, #4299e1, #3182ce); 
        }
        
        .historical-btn {
          background-color: #805ad5;
          background-image: linear-gradient(135deg, #805ad5, #6b46c1);
        }
        
        .prediction-btn {
          background-color: #ed8936;
          background-image: linear-gradient(135deg, #ed8936, #dd6b20);
        }
        
        .chatbot-btn {
          background-color: #48bb78;
          background-image: linear-gradient(135deg, #48bb78, #38a169);
        }
        
        .home-btn:hover, .home-btn.active {
          background-image: linear-gradient(135deg, #319795, #2c7a7b);
        }
        
        .realtime-btn:hover, .realtime-btn.active {
          background-image: linear-gradient(135deg, #3182ce, #2c5282);
        }
        
        .historical-btn:hover, .historical-btn.active {
          background-image: linear-gradient(135deg, #6b46c1, #553c9a);
        }
        
        .prediction-btn:hover, .prediction-btn.active {
          background-image: linear-gradient(135deg, #dd6b20, #c05621);
        }
        
        .chatbot-btn:hover, .chatbot-btn.active {
          background-image: linear-gradient(135deg, #38a169, #2f855a);
        }
        
        /* Right block for dark mode and hamburger icons */
        .right-controls {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .dark-mode-icon {
          display: flex;
          align-items: center;
          background-color: #4a5568;
          color: white;
          border: none;
          border-radius: 0.5rem;
          padding: 0.625rem 1rem;
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .dark-mode-text {
          display: none;
        }
        
        /* Show dark mode text on laptop/desktop view */
        @media (min-width: 1200px) {
          .dark-mode-text {
            display: inline;
            margin-left: 0.25rem;
          }
        }
        
        .hamburger-container {
          cursor: pointer;
          padding: 0.5rem;
          display: none;
          align-items: center;
        }
        
        /* Only show hamburger on mobile and tablet */
        @media (max-width: 1200px) {
          .hamburger-container {
            display: flex;
          }
        }
        
        .hamburger-icon {
          width: 1.5rem;
          height: 1.5rem;
          color: #333;
        }
        
        .dark-mode .hamburger-icon {
          color: #fff;
        }
        
        .mode-icon {
          width: 1.25rem;
          height: 1.25rem;
        }
      `}</style>
    </div>
  );
}

export default TopBar;
