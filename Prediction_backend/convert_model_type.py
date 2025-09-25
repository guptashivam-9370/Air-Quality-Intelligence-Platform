import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
import h5py
import numpy as np
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def rebuild_model_from_config(config_path):
    """
    Rebuild model architecture based on the config file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    pollutants = config["pollutants"]
    sequence_length = config["sequence_length"]
    lstm_units = config["model_params"]["lstm_units"]
    dropout_rate = config["model_params"]["dropout_rate"]
    l2_reg = config["model_params"]["l2_reg"]
    learning_rate = config["model_params"]["learning_rate"]
    
    # Number of features (pollutants)
    n_features = len(pollutants)
    input_shape = (10, 8)
    
    # Build model
    model = Sequential([
        # 1D Convolutional layer to capture local patterns
        Conv1D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        
        # First LSTM layer
        LSTM(lstm_units, 
            return_sequences=True,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(lstm_units,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer - one neuron per pollutant
        Dense(8),
        # Applying activation to ensure positive values
        Activation('softplus')  # Softplus ensures outputs are always positive
    ])
    
    # Using Adam optimizer with specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def convert_h5_to_tf(h5_model_path, tf_model_dir, config_path=None):
    """
    Convert a Keras .h5 model to TensorFlow SavedModel format (.tf)
    
    Args:
        h5_model_path: Path to the .h5 model file
        tf_model_dir: Directory where to save the converted model
        config_path: Optional path to the model configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(tf_model_dir, exist_ok=True)
    
    try:
        # Approach 1: Direct loading with custom objects
        print(f"Loading model from {h5_model_path}...")
        
        # Custom objects dictionary
        custom_objects = {
            'Policy': tf.keras.mixed_precision.Policy,
            'DTypePolicy': tf.keras.mixed_precision.Policy,
        }
        
        model = load_model(h5_model_path, compile=False, custom_objects=custom_objects)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Direct loading failed: {str(e)}")
        
        if config_path and os.path.exists(config_path):
            print(f"Using config file to rebuild model: {config_path}")
            try:
                # Approach 2: Rebuild model and load weights
                model = rebuild_model_from_config(config_path)
                
                # Extract weights from h5 file
                print("Extracting weights from H5 file...")
                model.load_weights(h5_model_path, by_name=True)
                print("Weights loaded successfully!")
                
            except Exception as nested_e:
                print(f"Rebuilding model failed: {str(nested_e)}")
                
                # Approach 3: Create a minimal model just to demonstrate the architecture
                print("Creating a minimal compatible model as placeholder...")
                try:
                    # Examine H5 file to determine input shape
                    with h5py.File(h5_model_path, 'r') as f:
                        # Try to find the input shape
                        print("Available keys in H5 file:", list(f.keys()))
                        
                    # Create a simple compatible model
                    model = Sequential()
                    model.add(InputLayer(input_shape=(7, 8)))  # 7 sequence length, 8 pollutants
                    model.add(LSTM(64))
                    model.add(Dropout(0.2))
                    model.add(Dense(8))  # 8 output features (one for each pollutant)
                    model.compile(optimizer='adam', loss='mse')
                    
                    print("Warning: Created placeholder model with default architecture.")
                    print("Note: This model does not contain the trained weights!")
                except Exception as final_e:
                    print(f"All approaches failed. Final error: {str(final_e)}")
                    raise
        else:
            # print("No config file provided or found. Using default model structure...")
            # # Create default model based on your provided config
            # model = Sequential()
            # model.add(InputLayer(input_shape=(7, 8)))  # 7 sequence length, 8 pollutants
            # model.add(LSTM(64))
            # model.add(Dropout(0.2))
            # model.add(Dense(8))  # 8 output features (one for each pollutant)
            # model.compile(optimizer='adam', loss='mse')
            
            try:
                # Try to load weights
                model.load_weights(h5_model_path)
                print("Loaded weights into default model structure")
            except Exception as w_e:
                print(f"Could not load weights: {str(w_e)}")
                print("Warning: Created empty model with default architecture!")
    
    # Save the model in TensorFlow SavedModel format
    print(f"Saving model to {tf_model_dir}...")
    tf.saved_model.save(model, tf_model_dir)
    print(f"Model successfully saved to {tf_model_dir}")
    
    return model

def process_all_state_models(base_dir="trained_models", config_file="model_config.json"):
    """
    Process all state models in the base directory
    
    Args:
        base_dir: Base directory containing state folders
        config_file: Name of the config file in each state folder
    """
    # Get all state folders
    state_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    if not state_folders:
        print(f"No state folders found in {base_dir}")
        return
    
    print(f"Found {len(state_folders)} state folders: {state_folders}")
    
    # Save the global config to disk for reference
    global_config = {
       "pollutants": [
        "pm2_5",
        "pm10",
        "so2",
        "no",
        "no2",
        "nh3",
        "o3",
        "co"
    ],
    "sequence_length": 10,
    "model_params": {
        "lstm_units": 128,
        "dropout_rate": 0.3,
        "l2_reg": 0.001,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 200,
        "patience": 20
    },
    "training_date": "2025-04-11 08:51:03.986845"
    }
    
    global_config_path = os.path.join(base_dir, "global_model_config.json")
    with open(global_config_path, 'w') as f:
        json.dump(global_config, f, indent=4)
    
    for state in state_folders:
        state_dir = os.path.join(base_dir, state)
        
        # Look for state-specific model naming pattern
        h5_model_path = os.path.join(state_dir, f"{state}_model.h5")
        
        # Check if the model file exists
        if not os.path.exists(h5_model_path):
            # Try the original naming pattern as fallback
            h5_model_path = os.path.join(state_dir, "_model.h5")
            if not os.path.exists(h5_model_path):
                print(f"No model .h5 file found in {state_dir}, skipping...")
                continue
        
        # Check for state-specific config file
        config_path = os.path.join(state_dir, config_file)
        if not os.path.exists(config_path):
            print(f"No config file found at {config_path}, using global config")
            config_path = global_config_path
        
        # Define output directory
        model_base_name = os.path.basename(h5_model_path).replace('.h5', '')
        tf_model_dir = os.path.join(state_dir, f"{model_base_name}.tf")
        
        print(f"\nProcessing state: {state}")
        try:
            model = convert_h5_to_tf(h5_model_path, tf_model_dir, config_path)
            print(f"Successfully processed model for {state}")
        except Exception as e:
            print(f"Error processing model for {state}: {str(e)}")

if __name__ == "__main__":
    # Display TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Process all state models
    process_all_state_models()
    print("\nConversion process completed.")