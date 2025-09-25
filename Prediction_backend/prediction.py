import pandas as pd
import numpy as np
import os
import json
import pickle
import requests
import base64
from datetime import datetime, timedelta
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import traceback
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Additional import for loading Keras models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting application")
    verify_files()
    test_api_key()
    yield
    # Shutdown
    logging.info("Shutting down application")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODELS_DIR = "trained_models"
RESULTS_DIR = "models_results"
METADATA_FILE = "../metadata.csv"  # Use metadata.csv from parent directory
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")  # Load from environment
MAX_PREDICTION_DAYS = 5

DEFAULT_POLLUTANTS = [
    "pm2_5",
    "pm10",
    "so2",
    "no",
    "no2",
    "nh3",
    "o3",
    "co"
]

DEFAULT_SEQUENCE_LENGTH = 10

# Cache for station data
station_cache = {}

# Load metadata with caching
metadata_cache = None

def load_metadata():
    global metadata_cache
    if metadata_cache is not None:
        return metadata_cache
    try:
        metadata_cache = pd.read_csv(METADATA_FILE)
        return metadata_cache
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return pd.DataFrame()

# Verify required files and directories
def verify_files():
    logging.info(f"Current working directory: {os.getcwd()}")
    
    # Check metadata file
    if not os.path.exists(METADATA_FILE):
        logging.error(f"Metadata file not found: {METADATA_FILE}")
    else:
        logging.info(f"Metadata file exists: {METADATA_FILE}")
    
    # Check model directories
    if not os.path.exists(MODELS_DIR):
        logging.error(f"Models directory not found: {MODELS_DIR}")
    else:
        states = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        logging.info(f"Found {len(states)} state directories: {states}")
        
        # Check model files for each state
        for state in states:
            model_path = os.path.join(MODELS_DIR, state, f"{state}_model.h5")
            scaler_path = os.path.join(MODELS_DIR, state, f"{state}_scaler.pkl")
            config_path = os.path.join(MODELS_DIR, state, f"{state}_config.json")
            
            if not os.path.exists(model_path):
                logging.error(f"ERROR: Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                logging.error(f"ERROR: Scaler file not found: {scaler_path}")
            if not os.path.exists(config_path):
                logging.error(f"ERROR: Config file not found: {config_path}")

# Test the OpenWeatherMap API key
def test_api_key():
    if not OPENWEATHERMAP_API_KEY:
        logging.error("OPENWEATHERMAP_API_KEY not set in environment variables")
        return
        
    lat, lon = 40.7128, -74.0060  # New York example coordinates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_timestamp}&end={end_timestamp}&appid={OPENWEATHERMAP_API_KEY}"
    try:
        response = requests.get(url)
        logging.info(f"API test response: {response.status_code}")
        if response.status_code == 200:
            logging.info("API key is valid")
        else:
            logging.error(f"API key test failed: {response.text}")
    except Exception as e:
        logging.error(f"API test error: {str(e)}")

# Endpoint: Get list of available states from model folders
@app.get("/api/states")
def get_states():
    try:
        states = [folder for folder in os.listdir(MODELS_DIR) 
                  if os.path.isdir(os.path.join(MODELS_DIR, folder))]
        logging.info(f"Retrieved {len(states)} states")
        return {"states": states}
    except Exception as e:
        logging.error(f"Error getting states: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting states: {str(e)}")

# Endpoint: Get stations for a specific state
@app.get("/api/stations")
def get_stations(state: str):
    global station_cache
    logging.info(f"Getting stations for state: {state}")
    
    if state in station_cache:
        logging.info(f"Returning {len(station_cache[state])} stations from cache")
        return {"stations": station_cache[state]}
    
    try:
        metadata = pd.read_csv(METADATA_FILE)
        state_stations = metadata[metadata['state'] == state].copy()
        
        stations = []
        for _, row in state_stations.iterrows():
            station = {
                "id": row['station_id'],
                "name": row['name'],
                "city": row['city'],
                "latitude": row['latitude'],
                "longitude": row['longitude']
            }
            stations.append(station)
        
        station_cache[state] = stations
        logging.info(f"Found {len(stations)} stations for state {state}")
        return {"stations": stations}
    except Exception as e:
        logging.error(f"Error getting stations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stations: {str(e)}")

# Endpoint: Get model metrics for a specific state
@app.get("/api/model_metrics")
def get_model_metrics(state: str):
    logging.info(f"Getting model metrics for state: {state}")
    try:
        metrics_path = os.path.join(RESULTS_DIR, state, f"{state}_metrics.csv")
        
        if not os.path.exists(metrics_path):
            logging.error(f"Metrics file not found: {metrics_path}")
            raise HTTPException(status_code=404, detail=f"Metrics file not found for state: {state}")

        metrics_df = pd.read_csv(metrics_path)
        
        # Calculate average metrics
        avg_mae = metrics_df['mae'].mean()
        avg_rmse = metrics_df['rmse'].mean()
        avg_r2 = metrics_df['r2'].mean()
        
        # Get metrics for each pollutant
        pollutants = {}
        for pollutant in ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']:
            pol_metrics = metrics_df[metrics_df['pollutant'] == pollutant]
            if not pol_metrics.empty:
                pollutants[pollutant] = {
                    'mae': pol_metrics['mae'].values[0],
                    'rmse': pol_metrics['rmse'].values[0],
                    'r2': pol_metrics['r2'].values[0]
                }
        
        return {
            "mae": avg_mae,
            "rmse": avg_rmse,
            "r2": avg_r2,
            "pollutants": pollutants
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model metrics: {str(e)}")

# Endpoint: Get model images for a specific state
@app.get("/api/model_images")
def get_model_images(state: str):
    logging.info(f"Getting model images for state: {state}")
    try:
        images = {}
        image_files = [
            f"{state}_all_predictions.png",
            f"{state}_pm2_5_prediction.png",
            f"{state}_pm10_prediction.png",
            f"{state}_o3_prediction.png",
            f"{state}_no2_prediction.png",
            f"{state}_so2_prediction.png",
            f"{state}_co_prediction.png",
            f"{state}_no_prediction.png",
            f"{state}_nh3_prediction.png"
        ]
        
        images_found = False
        for image_file in image_files:
            file_path = os.path.join(RESULTS_DIR, state, image_file)
            if os.path.exists(file_path):
                with open(file_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    if "all_predictions" in image_file:
                        key = "all_predictions"
                    else:
                        pollutant = image_file.replace(f"{state}_", "").replace(".png", "")
                        key = pollutant
                    images[key] = img_data
                    images_found = True
        
        if not images_found:
            logging.warning(f"No images found for state: {state}")
        
        return images
    except Exception as e:
        logging.error(f"Error getting model images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model images: {str(e)}")

# Endpoint: Get overall metrics
@app.get("/api/overall_metrics")
def get_overall_metrics():
    logging.info("Getting overall metrics")
    try:
        metrics_dir = os.path.join(RESULTS_DIR, "metrics")
        
        if not os.path.exists(metrics_dir):
            logging.error(f"Metrics directory not found: {metrics_dir}")
            raise HTTPException(status_code=404, detail="Overall metrics not found")
        
        avg_by_state_path = os.path.join(metrics_dir, "avg_metrics_by_state.csv")
        if not os.path.exists(avg_by_state_path):
            logging.error(f"State metrics file not found: {avg_by_state_path}")
            raise HTTPException(status_code=404, detail="State metrics not found")
            
        avg_by_state_df = pd.read_csv(avg_by_state_path)
        avg_by_state = avg_by_state_df.to_dict('records')
        
        avg_by_pollutant_path = os.path.join(metrics_dir, "avg_metrics_by_pollutant.csv")
        if not os.path.exists(avg_by_pollutant_path):
            logging.error(f"Pollutant metrics file not found: {avg_by_pollutant_path}")
            raise HTTPException(status_code=404, detail="Pollutant metrics not found")
            
        avg_by_pollutant_df = pd.read_csv(avg_by_pollutant_path)
        avg_by_pollutant = avg_by_pollutant_df.to_dict('records')
        
        all_metrics_path = os.path.join(metrics_dir, "all_metrics.csv")
        if not os.path.exists(all_metrics_path):
            logging.error(f"All metrics file not found: {all_metrics_path}")
            raise HTTPException(status_code=404, detail="All metrics not found")
            
        all_metrics_df = pd.read_csv(all_metrics_path)
        all_metrics = all_metrics_df.to_dict('records')
        
        return {
            "avg_by_state": avg_by_state,
            "avg_by_pollutant": avg_by_pollutant,
            "all_metrics": all_metrics
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error getting overall metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting overall metrics: {str(e)}")

# Function to get historical air quality data from OpenWeatherMap API
def get_historical_aqi_data(lat, lon, days=10):
    logging.info(f"Getting historical AQI data for lat={lat}, lon={lon}, days={days}")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_timestamp}&end={end_timestamp}&appid={OPENWEATHERMAP_API_KEY}"
    logging.info(f"Requesting data from OpenWeatherMap API: {url.replace(OPENWEATHERMAP_API_KEY, 'API_KEY')}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        logging.info(f"Received data from OpenWeatherMap API: {len(data.get('list', []))} entries")
        
        if 'list' not in data:
            logging.error(f"Invalid API response: {data}")
            raise Exception("Invalid response from OpenWeatherMap API")
        
        if len(data['list']) < 10:
            logging.warning(f"Very little data returned: {len(data['list'])} entries")
        
        daily_data = []
        current_date = None
        daily_pollutants = None
        count = 0
        
        for item in data['list']:
            timestamp = item['dt']
            date = datetime.fromtimestamp(timestamp).date()
            
            if current_date is None:
                current_date = date
                daily_pollutants = {p: 0 for p in DEFAULT_POLLUTANTS}
                count = 0
            
            if date != current_date:
                for key in daily_pollutants:
                    daily_pollutants[key] /= count if count > 0 else 1
                daily_data.append(daily_pollutants)
                current_date = date
                daily_pollutants = {p: 0 for p in DEFAULT_POLLUTANTS}
                count = 0
            
            components = item['components']
            for key in daily_pollutants:
                if key in components:
                    daily_pollutants[key] += components[key]
            count += 1
        
        if count > 0:
            for key in daily_pollutants:
                daily_pollutants[key] /= count
            daily_data.append(daily_pollutants)
        
        result = daily_data[-10:]
        logging.info(f"Processed {len(result)} days of historical data")
        return result
    
    except Exception as e:
        logging.error(f"Error fetching historical AQI data: {str(e)}")
        raise Exception(f"Error fetching historical AQI data: {str(e)}")

# Calculate AQI from pollutant concentrations
def calculate_aqi(pollutants):
    def map_aqi(concentration, c_low, c_high, i_low, i_high):
        return ((concentration - c_low) / (c_high - c_low)) * (i_high - i_low) + i_low
    
    def calculate_pm25_aqi(concentration):
        if concentration <= 12:
            return map_aqi(concentration, 0, 12, 0, 50)
        if concentration <= 35.4:
            return map_aqi(concentration, 12.1, 35.4, 51, 100)
        if concentration <= 55.4:
            return map_aqi(concentration, 35.5, 55.4, 101, 150)
        if concentration <= 150.4:
            return map_aqi(concentration, 55.5, 150.4, 151, 200)
        if concentration <= 250.4:
            return map_aqi(concentration, 150.5, 250.4, 201, 300)
        if concentration <= 500.4:
            return map_aqi(concentration, 250.5, 500.4, 301, 500)
        return 500
    
    def calculate_pm10_aqi(concentration):
        if concentration <= 54:
            return map_aqi(concentration, 0, 54, 0, 50)
        if concentration <= 154:
            return map_aqi(concentration, 55, 154, 51, 100)
        if concentration <= 254:
            return map_aqi(concentration, 155, 254, 101, 150)
        if concentration <= 354:
            return map_aqi(concentration, 255, 354, 151, 200)
        if concentration <= 424:
            return map_aqi(concentration, 355, 424, 201, 300)
        if concentration <= 604:
            return map_aqi(concentration, 425, 604, 301, 500)
        return 500
    
    def calculate_o3_aqi(concentration):
        ppb = concentration * 0.5
        if ppb <= 54:
            return map_aqi(ppb, 0, 54, 0, 50)
        if ppb <= 70:
            return map_aqi(ppb, 55, 70, 51, 100)
        if ppb <= 85:
            return map_aqi(ppb, 71, 85, 101, 150)
        if ppb <= 105:
            return map_aqi(ppb, 86, 105, 151, 200)
        if ppb <= 200:
            return map_aqi(ppb, 106, 200, 201, 300)
        return 500
    
    def calculate_no2_aqi(concentration):
        ppb = concentration * 0.53
        if ppb <= 53:
            return map_aqi(ppb, 0, 53, 0, 50)
        if ppb <= 100:
            return map_aqi(ppb, 54, 100, 51, 100)
        if ppb <= 360:
            return map_aqi(ppb, 101, 360, 101, 150)
        if ppb <= 649:
            return map_aqi(ppb, 361, 649, 151, 200)
        if ppb <= 1249:
            return map_aqi(ppb, 650, 1249, 201, 300)
        if ppb <= 2049:
            return map_aqi(ppb, 1250, 2049, 301, 500)
        return 500
    
    def calculate_so2_aqi(concentration):
        ppb = concentration * 0.38
        if ppb <= 35:
            return map_aqi(ppb, 0, 35, 0, 50)
        if ppb <= 75:
            return map_aqi(ppb, 36, 75, 51, 100)
        if ppb <= 185:
            return map_aqi(ppb, 76, 185, 101, 150)
        if ppb <= 304:
            return map_aqi(ppb, 186, 304, 151, 200)
        if ppb <= 604:
            return map_aqi(ppb, 305, 604, 201, 300)
        if ppb <= 1004:
            return map_aqi(ppb, 605, 1004, 301, 500)
        return 500
    
    def calculate_co_aqi(concentration):
        ppm = concentration * 0.000873
        if ppm <= 4.4:
            return map_aqi(ppm, 0, 4.4, 0, 50)
        if ppm <= 9.4:
            return map_aqi(ppm, 4.5, 9.4, 51, 100)
        if ppm <= 12.4:
            return map_aqi(ppm, 9.5, 12.4, 101, 150)
        if ppm <= 15.4:
            return map_aqi(ppm, 12.5, 15.4, 151, 200)
        if ppm <= 30.4:
            return map_aqi(ppm, 15.5, 30.4, 201, 300)
        if ppm <= 50.4:
            return map_aqi(ppm, 30.5, 50.4, 301, 500)
        return 500
    
    try:
        pm25 = pollutants.get('pm2_5', 0)
        pm10 = pollutants.get('pm10', 0)
        o3 = pollutants.get('o3', 0)
        no2 = pollutants.get('no2', 0)
        so2 = pollutants.get('so2', 0)
        co = pollutants.get('co', 0)
        
        pm25_aqi = calculate_pm25_aqi(pm25)
        pm10_aqi = calculate_pm10_aqi(pm10)
        o3_aqi = calculate_o3_aqi(o3)
        no2_aqi = calculate_no2_aqi(no2)
        so2_aqi = calculate_so2_aqi(so2)
        co_aqi = calculate_co_aqi(co)
        
        return max(pm25_aqi, pm10_aqi, o3_aqi, no2_aqi, so2_aqi, co_aqi)
    except Exception as e:
        logging.error(f"Error calculating AQI: {str(e)}")
        return 0

def load_model_and_predict(state, input_data, days_to_predict=5):
    try:
        state_dir = os.path.join(MODELS_DIR, state)
        
        # Load model configuration
        config_path = os.path.join(state_dir, f"{state}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            pollutants = config.get("pollutants", DEFAULT_POLLUTANTS)
            sequence_length = config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
            model_params = config.get("model_params", {
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "l2_reg": 0.001,
                "learning_rate": 0.001
            })
        else:
            pollutants = DEFAULT_POLLUTANTS
            sequence_length = DEFAULT_SEQUENCE_LENGTH
            model_params = {
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "l2_reg": 0.001,
                "learning_rate": 0.001
            }
            
        logging.info(f"Model config: pollutants={pollutants}, sequence_length={sequence_length}")
        
        # Force CPU usage with no GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Define input shape
        n_features = len(pollutants)
        input_shape = (sequence_length, n_features)
        
        # Create a model with the EXACT same architecture as during training
        model = Sequential([
            # 1D Convolutional layer to capture local patterns
            Conv1D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
            LeakyReLU(alpha=0.1),
            
            # First LSTM layer
            LSTM(model_params['lstm_units'], 
                return_sequences=True,
                kernel_regularizer=l2(model_params['l2_reg']),
                recurrent_regularizer=l2(model_params['l2_reg'])),
            BatchNormalization(),
            Dropout(model_params['dropout_rate']),
            
            # Second LSTM layer
            LSTM(model_params['lstm_units'],
                kernel_regularizer=l2(model_params['l2_reg']),
                recurrent_regularizer=l2(model_params['l2_reg'])),
            BatchNormalization(),
            Dropout(model_params['dropout_rate']),
            
            # Output layer - one neuron per pollutant
            Dense(n_features),
            # Applying activation to ensure positive values
            Activation('softplus')  # Softplus ensures outputs are always positive
        ])
        
        # Compile the model
        optimizer = Adam(learning_rate=model_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Load only the weights
        h5_model_path = os.path.join(state_dir, f"{state}_model.h5")
        logging.info(f"Loading weights from {h5_model_path}")
        model.load_weights(h5_model_path)
        
        # Load scaler
        scaler_path = os.path.join(state_dir, f"{state}_scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Loaded scaler from {scaler_path}")
        
        # Process input data
        input_df = pd.DataFrame(input_data)
        for col in pollutants:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        # Ensure the order of columns matches the expected pollutants order
        input_df = input_df[pollutants]
        
        # Scale input data
        scaled_input = scaler.transform(input_df)
        logging.info(f"Scaled input shape: {scaled_input.shape}")
        
        current_timesteps = scaled_input.shape[0]
        if current_timesteps < sequence_length:
            padding_needed = sequence_length - current_timesteps
            padding = np.zeros((padding_needed, len(pollutants)))
            scaled_input = np.vstack((padding, scaled_input))
            logging.info(f"Padded input from {current_timesteps} to {sequence_length} timesteps")
        elif current_timesteps > sequence_length:
            scaled_input = scaled_input[-sequence_length:]
            logging.info(f"Truncated input from {current_timesteps} to {sequence_length} timesteps")
        
        current_input = scaled_input.reshape(1, sequence_length, len(pollutants))
        logging.info(f"Reshaped input to {current_input.shape}")
        
        # Make predictions using a rolling window approach
        predictions = []
        current_sequence = current_input.copy()
        
        for _ in range(days_to_predict):
            # Use model.predict() directly with verbose=0 to reduce output
            pred = model.predict(current_sequence, verbose=0)
            
            predictions.append(pred[0])
            
            # Roll the window forward
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = pred[0]
        
        predictions_array = np.array(predictions)
        unscaled_predictions = scaler.inverse_transform(predictions_array)
        
        result = []
        for day in range(days_to_predict):
            day_result = {}
            for i, pollutant in enumerate(pollutants):
                day_result[pollutant] = float(unscaled_predictions[day, i])
            
            # Calculate AQI for each day's prediction
            day_result['aqi'] = calculate_aqi(day_result)
            day_result['date'] = (datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d')
            
            result.append(day_result)
            
        return result
        
    except Exception as e:
        logging.error(f"Error in model prediction: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        raise Exception(f"Error in prediction: {str(e)}")
# Function to rebuild model architecture
def rebuild_model_architecture(state):
    state_dir = os.path.join(MODELS_DIR, state)
    config_path = os.path.join(state_dir, f"{state}_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "pollutants": DEFAULT_POLLUTANTS,
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "model_params": {
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "l2_reg": 0.001,
                "learning_rate": 0.001
            }
        }
    
    pollutants = config.get("pollutants", DEFAULT_POLLUTANTS)
    sequence_length = config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)
    model_params = config.get("model_params", {})
    
    lstm_units = model_params.get("lstm_units", 128)
    dropout_rate = model_params.get("dropout_rate", 0.3)
    l2_reg = model_params.get("l2_reg", 0.001)
    learning_rate = model_params.get("learning_rate", 0.001)
    
    n_features = len(pollutants)
    input_shape = (sequence_length, n_features)
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(lstm_units, kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(n_features),
        Activation('softplus')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Prediction API endpoint
class PredictionRequest(BaseModel):
    state: str
    station_id: str
    days: int = 1

@app.post("/api/predict")
def predict(request: PredictionRequest = Body(...)):
    try:
        logging.info(f"Processing prediction request: state={request.state}, station_id={request.station_id}, days={request.days}")
        
        # Validate days
        if request.days < 1:
            raise HTTPException(status_code=400, detail="Days must be at least 1")
        
        state = request.state
        days = min(request.days, MAX_PREDICTION_DAYS)  # Limit to 5 days
        
        try:
            station_id = int(request.station_id)
            logging.info(f"Converted station_id to int: {station_id}")
        except (ValueError, TypeError):
            station_id = request.station_id
            logging.info(f"Using original station_id: {station_id}")
        
        try:
            metadata = pd.read_csv(METADATA_FILE)
            station_info = metadata[(metadata['station_id'] == station_id) | (metadata['station_id'] == str(station_id))]
            
            if station_info.empty:
                logging.error(f"No station found with ID: {station_id}")
                raise HTTPException(status_code=400, detail=f"Invalid station ID: {station_id}")
            
            latitude = station_info['latitude'].values[0]
            longitude = station_info['longitude'].values[0]
            
            # Validate coordinates
            if not (-90 <= latitude <= 90):
                raise HTTPException(status_code=400, detail=f"Invalid latitude: {latitude}")
            if not (-180 <= longitude <= 180):
                raise HTTPException(status_code=400, detail=f"Invalid longitude: {longitude}")
                
            logging.info(f"Using coordinates: lat={latitude}, lon={longitude}")
        except Exception as e:
            logging.error(f"Error getting station coordinates: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting station coordinates: {str(e)}")
        
        try:
            historical_data = get_historical_aqi_data(latitude, longitude)
            if len(historical_data) < 10:
                logging.warning(f"Insufficient historical data: {len(historical_data)} days, need 10")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Not enough historical data available (got {len(historical_data)}, need 10 days)"
                )
        except Exception as e:
            logging.error(f"Error getting historical data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting historical data: {str(e)}")
        
        try:
            predictions = load_model_and_predict(state, historical_data, days)
            logging.info(f"Successfully generated {len(predictions)} predictions")
            return {"predictions": predictions}
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"Unhandled error in predict endpoint: {str(e)}")
        logging.error(error_details)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)