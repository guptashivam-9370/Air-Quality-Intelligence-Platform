import React, { useState, useRef, useEffect } from 'react';
import { TextField, Button, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';

import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';
import SendIcon from '@mui/icons-material/Send';
import MenuIcon from '@mui/icons-material/Menu';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import CloseIcon from '@mui/icons-material/Close';
import ReactMarkdown from 'react-markdown';
import Plot from 'react-plotly.js';
import ExampleQuestions from '../components/AQIChatbot/ExampleQuestions';
import SourceViewer from '../components/AQIChatbot/SourceViewer';
import './AQIChatbot.css';
import axios from 'axios';
import { useLocation } from 'react-router-dom';

const API_URL = "http://localhost:8000/api/query";

function AQIChatbot() {
  const location = useLocation()

  const initialIsDark = location.state?.isdark ?? true;
  // const initialIsDark = localStorage.getItem('darkMode') === 'true' || true;

  const [chatHistory, setChatHistory] = useState([
    {
      role: "assistant",
      content: "Hello! I'm your Air Quality Index (AQI) assistant. I can help you explore air quality data from 662 stations across India from 2020 to 2025. Feel free to ask me any questions about air quality trends, pollutant levels, or specific cities.",
      id: 0
    }
  ]);

  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [messageId, setMessageId] = useState(1);
  const [queryData, setQueryData] = useState({});
  const [darkMode, setDarkMode] = useState(initialIsDark);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when chat history changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // Focus input field when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Toggle theme
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode);
    document.body.className = newDarkMode ? 'dark-mode' : 'light-mode';
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Set initial theme
  useEffect(() => {
    document.body.className = darkMode ? 'dark-mode' : 'light-mode';
  }, []);

  const handleSendMessage = async () => {
    if (!userInput.trim() || isProcessing) return;

    // Add user message to chat history
    const userMessageId = messageId;
    setChatHistory(prev => [...prev, { role: "user", content: userInput, id: userMessageId }]);
    setMessageId(prev => prev + 1);
    setUserInput('');
    setIsProcessing(true);

    try {
      // Send query to API
      const response = await axios.post(API_URL, { query: userInput });
      const result = response.data;

      // Create assistant message ID
      const assistantMessageId = messageId + 1;
      setMessageId(prev => prev + 2); // Increment by 2 to account for both messages

      // Add assistant response to chat history
      setChatHistory(prev => [...prev, {
        role: "assistant",
        content: result.answer,
        id: assistantMessageId
      }]);

      // Process and store visualization data if available
      if (result.answer) {
        const visualizationData = extractPollutantData(result.answer);

        setQueryData(prev => ({
          ...prev,
          [assistantMessageId]: {
            sources: result.sources || [],
            visualizationData
          }
        }));
      }
    } catch (error) {
      console.error("Error querying API:", error);

      // Add error message to chat history
      setChatHistory(prev => [...prev, {
        role: "assistant",
        content: `Error: ${error.message || "Failed to process your query"}`,
        id: messageId + 1
      }]);
      setMessageId(prev => prev + 2);
    } finally {
      setIsProcessing(false);
      inputRef.current?.focus();
    }
  };

  const handleExampleQuestion = (question) => {
    setUserInput(question);
    setSidebarOpen(false);
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };

  // Function to extract pollutant data from response text for visualization
  const extractPollutantData = (text) => {
    const patterns = {
      "PM2.5": /PM2\.5:?\s*(\d+\.?\d*)/g,
      "PM10": /PM10:?\s*(\d+\.?\d*)/g,
      "NO2": /NO2:?\s*(\d+\.?\d*)/g,
      "SO2": /SO2:?\s*(\d+\.?\d*)/g,
      "O3": /O3:?\s*(\d+\.?\d*)/g,
      "AQI": /AQI:?\s*(\d+\.?\d*)/g
    };

    const extractedData = [];

    for (const [pollutant, pattern] of Object.entries(patterns)) {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        extractedData.push({
          Pollutant: pollutant,
          Value: parseFloat(match[1])
        });
      }
    }

    return extractedData.length > 0 ? extractedData : null;
  };

  return (
    <div>
      <div className={`app-container ${darkMode ? 'dark-mode' : 'light-mode'}`}>
        {/* Header */}
        <TopBar isDark={darkMode} toggleDarkMode={toggleDarkMode} />

        {/* <div className="header">
          <div className="header-content">
            <div className="header-left">
              <button className="hamburger-menu" onClick={toggleSidebar} aria-label="Menu">
                <MenuIcon />
              </button>
              <h1 className="app-title">
                <span className="gradient-text">AQI <b>Chatbot</b></span>
              </h1>
            </div>

          </div>
        </div> */}

        <div className="main-container">
          {/* Sidebar with Examples */}
          <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
            <div className="sidebar-header">
              <h3 className="sidebar-title">AQI Chatbot</h3>
              <button className="close-sidebar" onClick={toggleSidebar}>
                <CloseIcon />
              </button>
            </div>
            <div className="sidebar-content">
              <ExampleQuestions onSelectQuestion={handleExampleQuestion} isDisabled={isProcessing} isDarkMode={darkMode} />
            </div>
          </div>

          {/* Overlay for mobile when sidebar is open */}
          {sidebarOpen && <div className="sidebar-overlay" onClick={toggleSidebar}></div>}

          {/* Main Chat Area */}
          <div className="chat-container">
            <div className="messages-container">
              {chatHistory.map((message, index) => (
                <div key={index} className={`message-wrapper ${message.role === 'user' ? 'user-message-wrapper' : 'assistant-message-wrapper'}`}>
                  <div className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}>
                    {message.role === 'assistant' && (
                      <div className="message-avatar">
                        <span>AQI</span>
                      </div>
                    )}

                    <div className="message-content">
                      <ReactMarkdown>{message.content}</ReactMarkdown>

                      {/* Render visualizations if available */}
                      {queryData[message.id]?.visualizationData && (
                        <div className="visualization-container">
                          <Plot
                            data={[
                              {
                                x: queryData[message.id].visualizationData.map(d => d.Pollutant),
                                y: queryData[message.id].visualizationData.map(d => d.Value),
                                type: 'bar',
                                marker: {
                                  color: darkMode ? '#4dabf5' : '#1976d2'
                                }
                              }
                            ]}
                            layout={{
                              title: 'Pollutant Levels',
                              paper_bgcolor: 'transparent',
                              plot_bgcolor: 'transparent',
                              font: {
                                color: darkMode ? '#e0e0e0' : '#212121'
                              },
                              margin: {
                                l: 50,
                                r: 50,
                                b: 50,
                                t: 50,
                              }
                            }}
                            style={{ width: '100%', height: '300px' }}
                          />
                        </div>
                      )}

                      {/* Show sources if available */}
                      {queryData[message.id]?.sources && queryData[message.id].sources.length > 0 && (
                        <Accordion className="sources-accordion">
                          <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            className="sources-summary"
                          >
                            <span>View Sources ({queryData[message.id].sources.length})</span>
                          </AccordionSummary>
                          <AccordionDetails className="sources-details">
                            <SourceViewer sources={queryData[message.id].sources} />
                          </AccordionDetails>
                        </Accordion>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-container">
              <div className="input-wrapper">
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="Ask about India's air quality data..."
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  disabled={isProcessing}
                  inputRef={inputRef}
                  className="input-field"
                  InputProps={{
                    className: "input-field-inner"
                  }}
                />
                <Button
                  variant="contained"
                  color="primary"
                  endIcon={isProcessing ? <div className="spinner"></div> : <SendIcon />}
                  onClick={handleSendMessage}
                  disabled={isProcessing || !userInput.trim()}
                  className="send-button"
                >
                  Send
                </Button>
              </div>
              <div className="input-footer">
                <p className="input-footer-text">
                  Data from 662 AQI monitoring stations across India (2020-2025)
                </p>
              </div>
            </div>
          </div>
        </div>


      </div>
      <div style={{marginTop: '20px'}}>
        <Footer />
      </div>
    </div>
  );
}

export default AQIChatbot;