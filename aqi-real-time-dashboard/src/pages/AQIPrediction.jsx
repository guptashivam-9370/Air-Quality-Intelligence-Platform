import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Line } from 'recharts';
import { BarChart, Bar, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Clock, MapPin, Calendar, BarChart2, ThermometerSun, Wind } from 'lucide-react';
import TopBar from '../components/real-time-main-page/TopBar';
import Footer from '../components/real-time-main-page/Footer';
import './AQIPrediction.css';

const AQIPrediction = () => {
    const location = useLocation();
    const initialIsDark = location.state?.isdark ?? true;
    const [darkMode, setDarkMode] = useState(initialIsDark);
    const [states, setStates] = useState([]);
    const [stations, setStations] = useState([]);
    const [selectedState, setSelectedState] = useState('');
    const [selectedStation, setSelectedStation] = useState('');
    const [predictionDays, setPredictionDays] = useState(1);
    const [loading, setLoading] = useState(false);
    const [predictions, setPredictions] = useState(null);
    const [modelMetrics, setModelMetrics] = useState(null);
    const [modelImages, setModelImages] = useState({});
    const [overallMetrics, setOverallMetrics] = useState(null);
    const [activeTab, setActiveTab] = useState('prediction');
    const [aqiCategory, setAqiCategory] = useState(null);

    useEffect(() => {
        // Fetch states from backend
        fetch('http://localhost:8501/api/states')
            .then(response => response.json())
            .then(data => {
                setStates(data.states);
            })
            .catch(error => console.error('Error fetching states:', error));

        // Fetch overall metrics
        fetch('http://localhost:8501/api/overall_metrics')
            .then(response => response.json())
            .then(data => {
                setOverallMetrics(data);
            })
            .catch(error => console.error('Error fetching overall metrics:', error));

        // Check for dark mode preference
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        setDarkMode(isDarkMode);
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
        }
    }, []);

    useEffect(() => {
        // Save dark mode preference
        localStorage.setItem('darkMode', darkMode);
        if (darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }, [darkMode]);

    useEffect(() => {
        if (selectedState) {
            // Fetch stations for selected state
            fetch(`http://localhost:8501/api/stations?state=${selectedState}`)
                .then(response => response.json())
                .then(data => {
                    setStations(data.stations);
                    setSelectedStation('');
                })
                .catch(error => console.error('Error fetching stations:', error));

            // Fetch model metrics for selected state
            fetch(`http://localhost:8501/api/model_metrics?state=${selectedState}`)
                .then(response => response.json())
                .then(data => {
                    setModelMetrics(data);
                })
                .catch(error => console.error('Error fetching model metrics:', error));

            // Fetch model visualization images
            fetch(`http://localhost:8501/api/model_images?state=${selectedState}`)
                .then(response => response.json())
                .then(data => {
                    setModelImages(data);
                })
                .catch(error => console.error('Error fetching model images:', error));
        }
    }, [selectedState]);

    const handlePredict = () => {
        if (!selectedState || !selectedStation || !predictionDays) {
            alert('Please select a state, station, and number of days');
            return;
        }

        setLoading(true);
        setPredictions(null);

        // Make prediction API call
        fetch('http://localhost:8501/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                state: selectedState,
                station_id: selectedStation,
                days: predictionDays,
            }),
        })
            .then(response => response.json())
            .then(data => {
                setPredictions(data.predictions);
                setAqiCategory(getAQICategory(data.predictions[0].aqi));
                setLoading(false);
            })
            .catch(error => {
                console.error('Error making prediction:', error);
                setLoading(false);
                alert('Error making prediction. Please try again.');
            });
    };

    const getAQICategory = (aqi) => {
        if (aqi <= 50) return { name: 'Good', color: '#55a84f', description: 'Air quality is considered satisfactory, and air pollution poses little or no risk.' };
        if (aqi <= 100) return { name: 'Moderate', color: '#fff964', description: 'Air quality is acceptable; however, some pollutants may be a concern for a very small number of people.' };
        if (aqi <= 150) return { name: 'Unhealthy for Sensitive Groups', color: '#ff9a52', description: 'Members of sensitive groups may experience health effects.' };
        if (aqi <= 200) return { name: 'Unhealthy', color: '#fe6a69', description: 'Everyone may begin to experience health effects.' };
        if (aqi <= 300) return { name: 'Very Unhealthy', color: '#a97abc', description: 'Health warnings of emergency conditions. The entire population is more likely to be affected.' };
        return { name: 'Hazardous', color: '#a87383', description: 'Health alert: everyone may experience more serious health effects.' };
    };

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    const formatDate = (daysFromNow) => {
        const date = new Date();
        date.setDate(date.getDate() + daysFromNow);
        return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    };
    // Actual model architecture based on the trained models
    const modelLayers = [
        { name: "Input Layer", description: "10 days × 8 pollutants", color: "#e3f2fd" },
        { name: "Conv1D", description: "64 filters, kernel size 3", color: "#bbdefb" },
        { name: "LeakyReLU", description: "Alpha = 0.1", color: "#90caf9" },
        { name: "LSTM Layer 1", description: "128 units, L2 regularization", color: "#64b5f6" },
        { name: "BatchNorm + Dropout", description: "Dropout rate: 0.3", color: "#42a5f5" },
        { name: "LSTM Layer 2", description: "128 units, L2 regularization", color: "#64b5f6" },
        { name: "BatchNorm + Dropout", description: "Dropout rate: 0.3", color: "#42a5f5" },
        { name: "Dense Layer", description: "8 outputs (one per pollutant)", color: "#90caf9" },
        { name: "Softplus Activation", description: "Ensures positive values", color: "#bbdefb" }
    ];

    // Actual model parameters from the trained models
    const MODEL_PARAMS = {
        'lstm_units': 128,
        'l2_reg': 0.001,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 200,
        'patience': 20
    };

    // Pollutants list
    const POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Pb"];
    return (
        <div className="prediction-page">
            <TopBar isDark={darkMode} toggleDarkMode={toggleDarkMode} />



            <div className="container">
                <h1 className="page-title">Air Quality Prediction System</h1>

                <div className="tabs">
                    <button
                        className={`tab ${activeTab === 'prediction' ? 'active' : ''}`}
                        onClick={() => setActiveTab('prediction')}
                    >
                        Prediction
                    </button>
                    <button
                        className={`tab ${activeTab === 'model-performance' ? 'active' : ''}`}
                        onClick={() => setActiveTab('model-performance')}
                    >
                        Model Performance
                    </button>
                    <button
                        className={`tab ${activeTab === 'model-architecture' ? 'active' : ''}`}
                        onClick={() => setActiveTab('model-architecture')}
                    >
                        Model Architecture
                    </button>
                </div>

                {activeTab === 'prediction' && (
                    <div className="prediction-section">
                        <div className="prediction-form">
                            <h2>Predict Air Quality</h2>
                            <div className="form-group">
                                <label>State:</label>
                                <select
                                    value={selectedState}
                                    onChange={e => setSelectedState(e.target.value)}
                                >
                                    <option value="">Select a state</option>
                                    {states.map(state => (
                                        <option key={state} value={state}>{state}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Monitoring Station:</label>
                                <select
                                    value={selectedStation}
                                    onChange={e => setSelectedStation(e.target.value)}
                                    disabled={!selectedState}
                                >
                                    <option value="">Select a station</option>
                                    {stations.map(station => (
                                        <option key={station.id} value={station.id}>
                                            {station.name} ({station.city})
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Number of days to predict:</label>
                                <select
                                    value={predictionDays}
                                    onChange={e => setPredictionDays(parseInt(e.target.value))}
                                >
                                    {[1, 2, 3, 4, 5].map(day => (
                                        <option key={day} value={day}>{day} day{day > 1 ? 's' : ''}</option>
                                    ))}
                                </select>
                            </div>

                            <button
                                className="predict-button"
                                onClick={handlePredict}
                                disabled={loading || !selectedState || !selectedStation}
                            >
                                {loading ? 'Processing...' : 'Predict Air Quality'}
                            </button>
                        </div>

                        {loading && (
                            <div className="loading-container">
                                <div className="loading-spinner"></div>
                                <p>Analyzing air quality data and making predictions...</p>
                            </div>
                        )}

                        {predictions && (
                            <div className="prediction-results">
                                <h2>Air Quality Predictions</h2>

                                <div className="aqi-highlight">
                                    <div className="aqi-values" style={{ backgroundColor: aqiCategory.color }}>
                                        <h3>Tomorrow's AQI</h3>
                                        <div className="aqi-number">{Math.round(predictions[0].aqi)}</div>
                                        <div className="aqi-category">{aqiCategory.name}</div>
                                    </div>
                                    <div className="aqi-description">
                                        <p>{aqiCategory.description}</p>
                                        <div className="station-info">
                                            <p><MapPin size={16} /> {stations.find(s => s.id === selectedStation)?.name}</p>
                                            <p><Calendar size={16} /> {formatDate(1)}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="prediction-chart-container">
                                    <h3>AQI Forecast (Next {predictionDays} Days)</h3>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart
                                            data={predictions.map((prediction, index) => ({
                                                name: formatDate(index + 1),
                                                aqi: Math.round(prediction.aqi),
                                            }))}
                                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="aqi"
                                                stroke="#8884d8"
                                                name="Air Quality Index"
                                                strokeWidth={2}
                                                activeDot={{ r: 8 }}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>

                                <div className="pollutants-section">
                                    <h3>Pollutant Predictions</h3>
                                    <div className="pollutant-grid">
                                        {['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3'].map(pollutant => {
                                            const displayName = {
                                                'pm2_5': 'PM2.5',
                                                'pm10': 'PM10',
                                                'o3': 'Ozone (O₃)',
                                                'no2': 'Nitrogen Dioxide (NO₂)',
                                                'so2': 'Sulfur Dioxide (SO₂)',
                                                'co': 'Carbon Monoxide (CO)',
                                                'no': 'Nitric Oxide (NO)',
                                                'nh3': 'Ammonia (NH₃)'
                                            }[pollutant];

                                            return (
                                                <div key={pollutant} className="pollutant-card">
                                                    <h4>{displayName}</h4>
                                                    <ResponsiveContainer width="100%" height={150}>
                                                        <LineChart
                                                            data={predictions.map((prediction, index) => ({
                                                                name: formatDate(index + 1),
                                                                value: prediction[pollutant],
                                                            }))}
                                                            margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                                                        >
                                                            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                                                            <YAxis tick={{ fontSize: 10 }} />
                                                            <Tooltip />
                                                            <Line
                                                                type="monotone"
                                                                dataKey="value"
                                                                stroke="#82ca9d"
                                                                strokeWidth={2}
                                                                dot={{ r: 3 }}
                                                            />
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                    <div className="pollutant-value">
                                                        <span className="value-label">Tomorrow:</span>
                                                        <span className="value-number">{predictions[0][pollutant].toFixed(2)} μg/m³</span>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'model-performance' && (
                    <div className="performance-section">
                        <h2>Model Performance Analysis</h2>

                        <div className="state-selector">
                            <label>Select State to View Model Performance:</label>
                            <select
                                value={selectedState}
                                onChange={e => setSelectedState(e.target.value)}
                            >
                                <option value="">Select a state</option>
                                {states.map(state => (
                                    <option key={state} value={state}>{state}</option>
                                ))}
                            </select>
                        </div>

                        {activeTab === 'model-performance' && (
                            <div className="performance-section">



                                {selectedState && modelMetrics && (

                                    <div className="metrics-container">

                                        <h2>Performance Metrics for {selectedState}</h2>

                                        <div className="metrics-cards">
                                            <div className="metric-card">
                                                <h4>MAE</h4>
                                                <div className="metric-value">{modelMetrics.mae.toFixed(2)}</div>
                                                <div className="metric-description">Mean Absolute Error</div>
                                            </div>
                                            <div className="metric-card">
                                                <h4>RMSE</h4>
                                                <div className="metric-value">{modelMetrics.rmse.toFixed(2)}</div>
                                                <div className="metric-description">Root Mean Square Error</div>
                                            </div>
                                            <div className="metric-card">
                                                <h4>R²</h4>
                                                <div className="metric-value">{modelMetrics.r2.toFixed(2)}</div>
                                                <div className="metric-description">Coefficient of Determination</div>
                                            </div>
                                        </div>

                                        <div className="performance-visuals">
                                            <div className="model-image">
                                                <h4>All Pollutants Prediction</h4>
                                                {modelImages.all_predictions && (
                                                    <img
                                                        src={`data:image/png;base64,${modelImages.all_predictions}`}
                                                        alt="All pollutants prediction visualization"
                                                    />
                                                )}
                                            </div>

                                            <div className="pollutant-metrics">
                                                <h4>Performance by Pollutant</h4>
                                                <div className="metrics-table-container">
                                                    <table className="metrics-table">
                                                        <thead>
                                                            <tr>
                                                                <th>Pollutant</th>
                                                                <th>MAE</th>
                                                                <th>RMSE</th>
                                                                <th>R²</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {modelMetrics.pollutants && Object.keys(modelMetrics.pollutants).map(pollutant => (
                                                                <tr key={pollutant}>
                                                                    <td>{pollutant}</td>
                                                                    <td>{modelMetrics.pollutants[pollutant].mae.toFixed(2)}</td>
                                                                    <td>{modelMetrics.pollutants[pollutant].rmse.toFixed(2)}</td>
                                                                    <td>{modelMetrics.pollutants[pollutant].r2.toFixed(2)}</td>
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="individual-pollutants">
                                            <h4>Individual Pollutant Predictions</h4>
                                            <div className="pollutant-images-grid">
                                                {['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3'].map(pollutant => (
                                                    <div key={pollutant} className="pollutant-image">
                                                        <h5>{pollutant.toUpperCase()}</h5>
                                                        {/* Use the exact keys from the API response */}
                                                        {modelImages[`${pollutant}_prediction`] && (
                                                            <img
                                                                src={`data:image/png;base64,${modelImages[`${pollutant}_prediction`]}`}
                                                                alt={`${pollutant} prediction visualization`}
                                                            />
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {overallMetrics && (
                            <div className="overall-metrics">
                                <h3>Overall Model Performance</h3>

                                <div className="overall-metrics-charts">
                                    <div className="chart-container">
                                        <h4>Average Metrics by State</h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart
                                                data={overallMetrics.avg_by_state}
                                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                                barSize={30} /* Wider bars */
                                                barGap={5} /* Less gap between bars */
                                            >
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis
                                                    dataKey="state"
                                                    tick={{ fill: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                    tickLine={{ stroke: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                />
                                                <YAxis
                                                    tick={{ fill: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                    tickLine={{ stroke: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                />
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: darkMode ? '#2d3748' : '#fff',
                                                        color: darkMode ? '#e0e0e0' : '#4a5568',
                                                        border: darkMode ? '1px solid #4a5568' : '1px solid #e2e8f0'
                                                    }}
                                                />
                                                <Legend
                                                    wrapperStyle={{
                                                        paddingTop: '10px',
                                                        color: darkMode ? '#e0e0e0' : '#4a5568'
                                                    }}
                                                />
                                                <Bar
                                                    dataKey="mae"
                                                    fill="#8884d8"
                                                    name="MAE"
                                                    radius={[4, 4, 0, 0]} /* Rounded tops */
                                                    stroke={darkMode ? '#9b96e3' : '#7b78c4'} /* Add stroke */
                                                    strokeWidth={1}
                                                />
                                                <Bar
                                                    dataKey="rmse"
                                                    fill="#82ca9d"
                                                    name="RMSE"
                                                    radius={[4, 4, 0, 0]}
                                                    stroke={darkMode ? '#a3dbb6' : '#6db58b'}
                                                    strokeWidth={1}
                                                />
                                                <Bar
                                                    dataKey="r2"
                                                    fill="#ffc658"
                                                    name="R²"
                                                    radius={[4, 4, 0, 0]}
                                                    stroke={darkMode ? '#ffd47f' : '#e6b041'}
                                                    strokeWidth={1}
                                                />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="chart-container">
                                        <h4>Average Metrics by Pollutant</h4>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart
                                                data={overallMetrics.avg_by_pollutant}
                                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                                barSize={30} /* Wider bars */
                                                barGap={5} /* Less gap between bars */
                                            >
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis
                                                    dataKey="pollutant"
                                                    tick={{ fill: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                    tickLine={{ stroke: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                />
                                                <YAxis
                                                    tick={{ fill: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                    tickLine={{ stroke: darkMode ? '#e0e0e0' : '#4a5568' }}
                                                />
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: darkMode ? '#2d3748' : '#fff',
                                                        color: darkMode ? '#e0e0e0' : '#4a5568',
                                                        border: darkMode ? '1px solid #4a5568' : '1px solid #e2e8f0'
                                                    }}
                                                />
                                                <Legend
                                                    wrapperStyle={{
                                                        paddingTop: '10px',
                                                        color: darkMode ? '#e0e0e0' : '#4a5568'
                                                    }}
                                                />
                                                <Bar
                                                    dataKey="mae"
                                                    fill="#8884d8"
                                                    name="MAE"
                                                    radius={[4, 4, 0, 0]}
                                                    stroke={darkMode ? '#9b96e3' : '#7b78c4'}
                                                    strokeWidth={1}
                                                />
                                                <Bar
                                                    dataKey="rmse"
                                                    fill="#82ca9d"
                                                    name="RMSE"
                                                    radius={[4, 4, 0, 0]}
                                                    stroke={darkMode ? '#a3dbb6' : '#6db58b'}
                                                    strokeWidth={1}
                                                />
                                                <Bar
                                                    dataKey="r2"
                                                    fill="#ffc658"
                                                    name="R²"
                                                    radius={[4, 4, 0, 0]}
                                                    stroke={darkMode ? '#ffd47f' : '#e6b041'}
                                                    strokeWidth={1}
                                                />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}


                {activeTab === 'model-architecture' && (
                    <div className="model-architecture-section">
                        <h2>LSTM Model Architecture</h2>
                        <div>
                            <p>This model uses a hybrid CNN-LSTM architecture to predict air quality based on 10 days of historical data with 8 pollutant measurements.</p>
                            <p>The architecture is optimized for time-series forecasting with regularization techniques to prevent overfitting.</p>
                        </div>

                        <h3>Model Architecture Diagram</h3>
                        <div className="model-layers">
                            {modelLayers.map((layer, index) => (
                                <React.Fragment key={index}>
                                    <div className="layer-box" style={{ backgroundColor: layer.color }}>
                                        <h4>{layer.name}</h4>
                                        <p>{layer.description}</p>
                                    </div>
                                    {index < modelLayers.length - 1 && <div className="connector"></div>}
                                </React.Fragment>
                            ))}
                        </div>

                        <h3>Model Specifications</h3>
                        <div className="details-grid">
                            <div className="details-card">
                                <h4 className="details-card-title">Input Shape</h4>
                                <p className="details-card-text">10 days of historical data × 8 pollutants</p>
                                <p className="details-card-note">Ten days of historical measurements of all pollutants</p>
                            </div>
                            <div className="details-card">
                                <h4 className="details-card-title">Preprocessing</h4>
                                <p className="details-card-text">1D Convolutional Layer</p>
                                <p className="details-card-note">Captures local patterns in time series data</p>
                            </div>
                            <div className="details-card">
                                <h4 className="details-card-title">Learning Algorithm</h4>
                                <p className="details-card-text">Dual LSTM Layers</p>
                                <p className="details-card-note">Two LSTM layers with 128 units each for deep learning</p>
                            </div>
                            <div className="details-card">
                                <h4 className="details-card-title">Output Constraints</h4>
                                <p className="details-card-text">Softplus Activation</p>
                                <p className="details-card-note">Ensures all pollutant predictions are positive</p>
                            </div>
                        </div>

        

                        <h3>Training Parameters</h3>
                        <div className="training-params-grid">
                            <div className="param-card">
                                <h4 className="param-name">Learning Rate</h4>
                                <p className="param-value">0.001</p>
                            </div>
                            <div className="param-card">
                                <h4 className="param-name">LSTM Units</h4>
                                <p className="param-value">128</p>
                            </div>
                            <div className="param-card">
                                <h4 className="param-name">L2 Regularization</h4>
                                <p className="param-value">0.001</p>
                            </div>
                            <div className="param-card">
                                <h4 className="param-name">Dropout Rate</h4>
                                <p className="param-value">0.3</p>
                            </div>
                            <div className="param-card">
                                <h4 className="param-name">Batch Size</h4>
                                <p className="param-value">32</p>
                            </div>
                            <div className="param-card">
                                <h4 className="param-name">Epochs</h4>
                                <p className="param-value">200 (with early stopping)</p>
                            </div>
                        </div>

                        <h3>Key Model Features</h3>
                        <div className="features-grid">
                            <div className="feature-card">
                                <div className="icon-container">
                                    <BarChart2 size={24} />
                                </div>
                                <h4 className="feature-title">Multi-Pollutant Prediction</h4>
                                <p className="feature-text">Simultaneously predicts all 8 pollutants</p>
                            </div>
                            <div className="feature-card">
                                <div className="icon-container" style={{ color: '#ff9800' }}>
                                    <ThermometerSun size={24} />
                                </div>
                                <h4 className="feature-title">Real-time Data Integration</h4>
                                <p className="feature-text">Uses OpenWeatherMap API for latest historical data</p>
                            </div>
                            
                            <div className="feature-card">
                                <div className="icon-container" style={{ color: '#673ab7' }}>
                                    <Clock size={24} />
                                </div>
                                <h4 className="feature-title">Multi-Day Forecasting</h4>
                                <p className="feature-text">Predicts up to 5 days in advance using iterative prediction</p>
                            </div>
                        </div>

                        <h3>Model Limitations & Future Improvements</h3>
                        <ul className="limitations-list">
                            <li className="limitation-item">
                                <span className="limitation-title">Data Sparsity:</span> Some monitoring stations have incomplete historical data, affecting prediction accuracy
                            </li>
                            <li className="limitation-item">
                                <span className="limitation-title">API Dependency:</span> Relies on OpenWeatherMap API for real-time data, which may have rate limits or downtime
                            </li>
                            <li className="limitation-item">
                                <span className="limitation-title">State-Specific Models:</span> Each state has its own model, limiting cross-state predictions
                            </li>
                            <li className="limitation-item">
                                <span className="limitation-title">Prediction Horizon:</span> Maximum 5 days prediction due to iterative approach causing error accumulation
                            </li>
                        </ul>
                    </div>
                )}
            </div>

            <Footer />
        </div>
    );
};

export default AQIPrediction;