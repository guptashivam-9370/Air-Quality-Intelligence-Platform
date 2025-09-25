import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';
import './AboutUsPage.css';

// Team member placeholder images - replace with actual team member photos
const TEAM_MEMBERS = [
  {
    name: "Rishab Sonthalia",
    rollNumber: "220150035",
    email: "s.rishab@iitg.ac.in",
    photo: "/images/i1.jpeg" // Replace with actual photo path
  },
  {
    name: "Shivam Gupta",
    rollNumber: "220107099",
    email: "shivamgupta@iitg.ac.in",
    photo: "/images/i3.jpeg" // Updated photo path
  },
  {
    name: "Sudhanshu Raj",
    rollNumber: "220107085",
    email: "r.sudhanshu@iitg.ac.in",
    photo: "/images/i2.jpeg" // Updated photo path
  },
];

function AboutUsPage() {
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

      <div className="about-us-container">
        <div className="about-section">
          <h1>About Our Project</h1>
          <div className="project-description">
            <p>
              Welcome to our Air Quality Intelligence (AQI) Platform. We have created a comprehensive system that analyzes AQI data across 662 stations throughout India.
            </p>

            <h2>Project Features</h2>
            <ul className="feature-list">
              <Link to="/" style={{ textDecoration: 'none' }}>
                <li>
                  <span className="feature-icon">📊</span>
                  <div className="feature-details">
                    <h3>Real-time Analysis</h3>
                    <p>Live monitoring and visualization of air quality data from 662 stations across India</p>
                  </div>
                </li>
              </Link>

              <Link to="/prediction" style={{ textDecoration: 'none' }}>
                <li style={{ cursor: 'pointer' }}>
                  <span className="feature-icon">🔮</span>
                  <div className="feature-details">
                    <h3>Prediction Modeling</h3>
                    <p>Advanced machine learning models to forecast future air quality trends</p>
                  </div>
                </li>
              </Link>

              <Link to="/aqi-chatbot" style={{ textDecoration: 'none' }}>
                <li>
                  <span className="feature-icon">🤖</span>
                  <div className="feature-details">
                    <h3>AQI Chatbot</h3>
                    <p>Intelligent assistant for answering queries related to air quality data</p>
                  </div>
                </li>
              </Link>
            </ul>
          </div>
        </div>

        <div className="team-section">
          <h2>Our Team</h2>
          <div className="team-members">
            {TEAM_MEMBERS.map((member, index) => (
              <div className="team-member-card" key={index}>
                <div className="member-photo">
                  <img src={member.photo} alt={`${member.name}`} />
                </div>
                <div className="member-info">
                  <h3>{member.name}</h3>
                  <p><strong>Roll:</strong> {member.rollNumber}</p>
                  <p><strong>Email:</strong> {member.email}</p>
                </div>
              </div>
            ))}
          </div>
        </div>


      </div>

      <Footer />
    </div>
  );
}

export default AboutUsPage;