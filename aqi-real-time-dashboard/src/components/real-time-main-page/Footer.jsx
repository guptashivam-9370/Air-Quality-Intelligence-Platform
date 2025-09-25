import React from 'react';
import { Facebook, Instagram, Twitter, Linkedin, Youtube } from 'lucide-react';
import './Footer.css'; // Import the stylesheet

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="grid-container">
          {/* Logo and Description Column */}
          <div>
            <div className="logo">
              <span className="logo-blue">A</span>
              <span className="logo-green">Q</span>
              <span className="logo-yellow">I</span>
            </div>
            <p className="tagline">
              <strong>Air Quality Intelligence</strong> <br /> Platform
            </p>
          </div>

          {/* About AQI Column */}
          <div>
            <h3 className="heading">About AQI</h3>
            <ul className="list">
              <li className="list-item">
                <a href="/about-us" className="link">About Us</a>
              </li>
              <li className="list-item">
                <a href="/about-us" className="link">Contact Us</a>
              </li>
              <li className="list-item">
                <a href="/about-us" className="link">Blogs</a>
              </li>
            </ul>
          </div>

          {/* Analysis Column */}
          <div>
            <h3 className="heading">Analysis</h3>
            <ul className="list">
              <li className="list-item">
                <a href="/" className="link">Real-time Analysis</a>
              </li>
              <li className="list-item">
                <a href="/aqi-map" className="link">AQI Heatmap</a>
              </li>
            </ul>
          </div>

          {/* AI Analysis Column */}
          <div>
            <h3 className="heading">AI Analysis</h3>
            <ul className="list">
              <li className="list-item">
                <a href="/prediction" className="link">Prediction</a>
              </li>
              <li className="list-item">
                <a href="/aqi-chatbot" className="link">AQI Chatbot</a>
              </li>
            </ul>
          </div>
        </div>

        {/* Social Media Links */}
        <div className="divider">
          <div className="bottom-row">
            <p className="copyright">
              © 2025 AQI Dashboard. All rights reserved | Data collected from OpenWeatherMap API
            </p>
            <div className="social-container">
              <span className="social-text">Find us on:</span>
              <div className="social-icons">
                <a href="https://instagram.com" aria-label="Instagram" className="social-icon">
                  <Instagram size={20} />
                </a>
                <a href="https://twitter.com" aria-label="Twitter" className="social-icon">
                  <Twitter size={20} />
                </a>
                <a href="https://linkedin.com" aria-label="LinkedIn" className="social-icon">
                  <Linkedin size={20} />
                </a>
                <a href="https://youtube.com" aria-label="YouTube" className="social-icon">
                  <Youtube size={20} />
                </a>
                <a href="https://facebook.com" aria-label="Facebook" className="social-icon">
                  <Facebook size={20} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
