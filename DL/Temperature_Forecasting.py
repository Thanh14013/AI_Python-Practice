# Temperature Forecasting with Recurrent Neural Networks
# Implementation covering:
# - Data preparation
# - Baseline models (common-sense and ML)
# - Recurrent neural networks
# - Advanced techniques (dropout, stacking, bidirectional)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
import os

# ----- DATA PREPARATION -----

# Function to download and prepare the Jena climate dataset
def download_and_prepare_data(url):
    """
    Downloads the dataset if not available and returns it as a DataFrame
    """
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Path to the CSV file
    csv_path = os.path.join(data_dir, 'jena_climate.csv')
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(csv_path):
        print("Downloading dataset...")
        # Use pandas to download the CSV
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
        print("Dataset downloaded and saved.")
    else:
        print("Dataset already available.")
        df = pd.read_csv(csv_path)
    
    return df

# URL to the Jena Climate dataset
url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv'

# Download and prepare the data
try:
    df = download_and_prepare_data(url)
    print("Data shape:", df.shape)
    print("First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error downloading data: {e}")
    # Create synthetic data if download fails
    print("Creating synthetic climate data instead...")
    # Create a datetime index
    dates = pd.date_range(start='2009-01-01', end='2016-12-31', freq='10min')
    # Create synthetic temperature data with daily and yearly cycles plus noise
    n = len(dates)
    hourly_temp = 15 + 10 * np.sin(np.linspace(0, 2*n*np.pi/144, n))  # Daily cycle (144 10-min intervals per day)
    yearly_temp = 5 * np.sin(np.linspace(0, 2*np.pi, n))  # Yearly cycle
    noise = np.random.normal(0, 1, n)  # Random noise
    temp = hourly_temp + yearly_temp + noise
    
    # Create DataFrame with synthetic data
    df = pd.DataFrame({
        'Date Time': dates,
        'T (degC)': temp,
        'p (mbar)': 1000 + yearly_temp + np.random.normal(0, 5, n),
        'rh (%)': 70 + 20 * np.sin(np.linspace(0, 4*n*np.pi/144, n)) + np.random.normal(0, 5, n),
        'wv (m/s)': 3 + 2 * np.sin(np.linspace(0, 3*n*np.pi/144, n)) + np.random.normal(0, 1, n)
    })
    df['Date Time'] = df['Date Time'].astype(str)

# Data exploration
def explore_data(df):
    """
    Displays basic statistics and visualizations of the dataset
    """
    print("\nData exploration:")
    print(f"Dataset shape: {df.shape}")
    print(f"Time period: {df['Date Time'].iloc[0]} to {df['Date Time'].iloc[-1]}")
    print("\nBasic statistics for temperature:")
    temp_data = df['T (degC)']
    print(f"Mean: {temp_data.mean():.2f}°C")
    print(f"Max: {temp_data.max():.2f}°C")
    print(f"Min: {temp_data.min():.2f}°C")
    print(f"Standard deviation: {temp_data.std():.2f}°C")
    
    # Plot temperature data for a sample period (first week)
    plt.figure(figsize=(15, 5))
    sample_data = temp_data[:1008]  # First week (144 * 7 = 1008 measurements)
    plt.plot(range(len(sample_data)), sample_data)
    plt.title('Temperature data for the first week')
    plt.xlabel('Time steps (10-minute intervals)')
    plt.ylabel('Temperature (°C)')
    # plt.show()  # Uncomment if you want to display the plot

# Data preprocessing for time series forecasting
def preprocess_data(df, feature_cols=None, target_col='T (degC)', 
                   lookback=144, delay=144, batch_size=128, 
                   step=6, normalize=True):
    """
    Preprocesses the data for time series forecasting
    
    Args:
        df: DataFrame containing the time series data
        feature_cols: List of feature columns to use (None = use all except date)
        target_col: The target column to predict
        lookback: How many timesteps back to consider for each prediction
        delay: How many timesteps in the future to predict
        batch_size: Batch size for training
        step: Sampling rate (1 = use every timestep, 6 = use every hour)
        normalize: Whether to normalize the data
        
    Returns:
        Prepared datasets and scaler for denormalization
    """
    # Select features and target
    if feature_cols is None:
        # Use all columns except the date column
        feature_cols = [col for col in df.columns if col != 'Date Time']
    
    features = df[feature_cols].values
    target = df[target_col].values
    
    # Normalize the data if requested
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features)
        
        # Create a separate scaler for the target for denormalization later
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target = target_scaler.fit_transform(target.reshape(-1, 1))
    else:
        scaler = None
        target_scaler = None
        target = target.reshape(-1, 1)
    
    # Prepare the sequences for training
    def generate_sequences(data, target, lookback, delay, step):
        X, y = [], []
        max_idx = len(data) - delay - lookback
        
        for i in range(0, max_idx, step):
            X.append(data[i:i + lookback])
            y.append(target[i + lookback + delay - 1])
        
        return np.array(X), np.array(y)
    
    # Generate sequences for training, validation, and test
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    # Training set
    X_train, y_train = generate_sequences(
        features[:train_end], target[:train_end], lookback, delay, step)
    
    # Validation set
    X_val, y_val = generate_sequences(
        features[train_end:val_end], target[train_end:val_end], lookback, delay, step)
    
    # Test set
    X_test, y_test = generate_sequences(
        features[val_end:], target[val_end:], lookback, delay, step)
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

explore_data(df)

# Preprocess the data - here we're predicting temperature 24 hours ahead using the past 24 hours
# We're sampling at hourly intervals (step=6 because the data is recorded every 10 minutes)
X_train, y_train, X_val, y_val, X_test, y_test, temp_scaler = preprocess_data(
    df, 
    feature_cols=['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)'],
    target_col='T (degC)',
    lookback=144,  # 24 hours lookback
    delay=144,     # Predict 24 hours ahead
    step=6,        # Sample hourly
    normalize=True
)

# ----- COMMON-SENSE, NON-MACHINE-LEARNING BASELINE -----

def evaluate_model(y_true, y_pred, model_name, scaler=None):
    """
    Evaluates the model performance
    """
    # Denormalize if a scaler is provided
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

# Persistence model (naive baseline) - predict that tomorrow's temperature will be the same as today's
def persistence_baseline(X_test, y_test, scaler=None):
    """
    Implements a persistence model that predicts the last observed value
    """
    # For each test sample, predict the last value in the input sequence
    y_pred = X_test[:, -1, 0].reshape(-1, 1)  # Temperature is the first feature
    
    # Evaluate the model
    return evaluate_model(y_test, y_pred, "Persistence Baseline", scaler)

# Seasonal baseline - predict based on the temperature at the same time the previous day
def seasonal_baseline(X_test, y_test, lookback, scaler=None):
    """
    Implements a seasonal model that predicts based on the value from the same time in the previous day
    """
    # Assuming the data is sampled hourly and lookback is 24 hours,
    # we're predicting that temperature will be the same as it was 24 hours ago
    steps_per_day = 24
    if lookback >= steps_per_day:
        y_pred = X_test[:, -steps_per_day, 0].reshape(-1, 1)
    else:
        y_pred = X_test[:, 0, 0].reshape(-1, 1)  # Fallback to first value if lookback is too small
        
    # Evaluate the model
    return evaluate_model(y_test, y_pred, "Seasonal Baseline", scaler)

# Evaluate the baseline models
persistence_mae, persistence_rmse = persistence_baseline(X_test, y_test, temp_scaler)
seasonal_mae, seasonal_rmse = seasonal_baseline(X_test, y_test, 144, temp_scaler)

# ----- BASIC MACHINE LEARNING APPROACH -----

# Dense neural network as a simple machine learning approach
def dense_model(input_shape):
    """
    Creates a simple dense neural network model
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the dense model
dense_nn = dense_model((X_train.shape[1], X_train.shape[2]))
print(dense_nn.summary())

# Train the model
dense_history = dense_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the dense model
dense_pred = dense_nn.predict(X_test)
dense_mae, dense_rmse = evaluate_model(y_test, dense_pred, "Dense Neural Network", temp_scaler)

# ----- FIRST RECURRENT BASELINE -----

# Simple RNN model
def simple_rnn_model(input_shape):
    """
    Creates a simple RNN model
    """
    model = keras.Sequential([
        layers.SimpleRNN(32, input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# LSTM model
def lstm_model(input_shape):
    """
    Creates an LSTM model
    """
    model = keras.Sequential([
        layers.LSTM(32, input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# GRU model
def gru_model(input_shape):
    """
    Creates a GRU model
    """
    model = keras.Sequential([
        layers.GRU(32, input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the RNN models
input_shape = (X_train.shape[1], X_train.shape[2])

# Simple RNN
simple_rnn = simple_rnn_model(input_shape)
print(simple_rnn.summary())

# Train Simple RNN
simple_rnn_history = simple_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# LSTM
lstm = lstm_model(input_shape)
print(lstm.summary())

# Train LSTM
lstm_history = lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# GRU
gru = gru_model(input_shape)
print(gru.summary())

# Train GRU
gru_history = gru.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the RNN models
simple_rnn_pred = simple_rnn.predict(X_test)
simple_rnn_mae, simple_rnn_rmse = evaluate_model(
    y_test, simple_rnn_pred, "Simple RNN", temp_scaler)

lstm_pred = lstm.predict(X_test)
lstm_mae, lstm_rmse = evaluate_model(
    y_test, lstm_pred, "LSTM", temp_scaler)

gru_pred = gru.predict(X_test)
gru_mae, gru_rmse = evaluate_model(
    y_test, gru_pred, "GRU", temp_scaler)

# ----- USING RECURRENT DROPOUT TO FIGHT OVERFITTING -----

def lstm_with_dropout(input_shape, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    """
    Creates an LSTM model with dropout for regularization
    """
    model = keras.Sequential([
        layers.LSTM(32, 
                   dropout=dropout_rate,
                   recurrent_dropout=recurrent_dropout_rate,
                   input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the LSTM with dropout
lstm_dropout = lstm_with_dropout(input_shape, 0.2, 0.2)
print(lstm_dropout.summary())

# Train LSTM with dropout
lstm_dropout_history = lstm_dropout.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the LSTM with dropout
lstm_dropout_pred = lstm_dropout.predict(X_test)
lstm_dropout_mae, lstm_dropout_rmse = evaluate_model(
    y_test, lstm_dropout_pred, "LSTM with Dropout", temp_scaler)

# ----- STACKING RECURRENT LAYERS -----

def stacked_lstm_model(input_shape, dropout_rate=0.2):
    """
    Creates a stacked LSTM model
    """
    model = keras.Sequential([
        layers.LSTM(32, 
                   dropout=dropout_rate,
                   return_sequences=True,
                   input_shape=input_shape),
        layers.LSTM(16, dropout=dropout_rate),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the stacked LSTM
stacked_lstm = stacked_lstm_model(input_shape, 0.2)
print(stacked_lstm.summary())

# Train stacked LSTM
stacked_lstm_history = stacked_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the stacked LSTM
stacked_lstm_pred = stacked_lstm.predict(X_test)
stacked_lstm_mae, stacked_lstm_rmse = evaluate_model(
    y_test, stacked_lstm_pred, "Stacked LSTM", temp_scaler)

# ----- USING BIDIRECTIONAL RNNs -----

def bidirectional_lstm_model(input_shape, dropout_rate=0.2):
    """
    Creates a bidirectional LSTM model
    """
    model = keras.Sequential([
        layers.Bidirectional(
            layers.LSTM(32, dropout=dropout_rate), 
            input_shape=input_shape
        ),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create and train the bidirectional LSTM
bidirectional_lstm = bidirectional_lstm_model(input_shape, 0.2)
print(bidirectional_lstm.summary())

# Train bidirectional LSTM
bidirectional_lstm_history = bidirectional_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the bidirectional LSTM
bidirectional_lstm_pred = bidirectional_lstm.predict(X_test)
bidirectional_lstm_mae, bidirectional_lstm_rmse = evaluate_model(
    y_test, bidirectional_lstm_pred, "Bidirectional LSTM", temp_scaler)

# ----- GOING EVEN FURTHER -----

def advanced_lstm_model(input_shape):
    """
    Creates an advanced LSTM model with multiple techniques
    - Stacked bidirectional layers
    - Dropout for regularization
    - Batch normalization
    - Residual connections
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # First bidirectional LSTM layer
    x = layers.Bidirectional(layers.LSTM(64, 
                                        dropout=0.2, 
                                        return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    
    # Second bidirectional LSTM layer with residual connection
    residual = x
    x = layers.Bidirectional(layers.LSTM(32, 
                                        dropout=0.2, 
                                        return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    
    # Add a 1x1 conv to match dimensions for the residual connection
    residual = layers.TimeDistributed(layers.Dense(64))(residual)
    x = layers.add([x, residual])
    
    # Final LSTM layer
    x = layers.LSTM(32, dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    # Output
    outputs = layers.Dense(1)(x)
    
    # Build the model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Create and train the advanced LSTM
advanced_lstm = advanced_lstm_model(input_shape)
print(advanced_lstm.summary())

# Train advanced LSTM
advanced_lstm_history = advanced_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the advanced LSTM
advanced_lstm_pred = advanced_lstm.predict(X_test)
advanced_lstm_mae, advanced_lstm_rmse = evaluate_model(
    y_test, advanced_lstm_pred, "Advanced LSTM", temp_scaler)

# ----- WRAPPING UP -----

# Collect all results
models = [
    "Persistence Baseline",
    "Seasonal Baseline",
    "Dense Neural Network",
    "Simple RNN",
    "LSTM",
    "GRU",
    "LSTM with Dropout",
    "Stacked LSTM",
    "Bidirectional LSTM",
    "Advanced LSTM"
]

mae_values = [
    persistence_mae,
    seasonal_mae,
    dense_mae,
    simple_rnn_mae,
    lstm_mae,
    gru_mae,
    lstm_dropout_mae,
    stacked_lstm_mae,
    bidirectional_lstm_mae,
    advanced_lstm_mae
]

rmse_values = [
    persistence_rmse,
    seasonal_rmse,
    dense_rmse,
    simple_rnn_rmse,
    lstm_rmse,
    gru_rmse,
    lstm_dropout_rmse,
    stacked_lstm_rmse,
    bidirectional_lstm_rmse,
    advanced_lstm_rmse
]

# Create a summary dataframe
results_df = pd.DataFrame({
    'Model': models,
    'MAE': mae_values,
    'RMSE': rmse_values
})

print("\nModel Performance Summary:")
print(results_df.sort_values('MAE'))

# Plot the results
plt.figure(figsize=(12, 8))

# Plot MAE values
plt.subplot(2, 1, 1)
plt.barh(models, mae_values)
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('MAE')
plt.tight_layout()

# Plot RMSE values
plt.subplot(2, 1, 2)
plt.barh(models, rmse_values)
plt.title('Root Mean Squared Error (RMSE)')
plt.xlabel('RMSE')
plt.tight_layout()

# plt.show()  # Uncomment to display the plot

# Visualize predictions for the best model (assuming it's the advanced LSTM)
plt.figure(figsize=(15, 6))

# Get a subset of the data for visualization
time_steps = range(100)
truth = temp_scaler.inverse_transform(y_test[:100])
predictions = temp_scaler.inverse_transform(advanced_lstm_pred[:100])

plt.plot(time_steps, truth, 'b-', label='Actual Temperature')
plt.plot(time_steps, predictions, 'r-', label='Predicted Temperature')
plt.title('Temperature Forecast (24 hours ahead)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

# plt.show()  # Uncomment to display the plot

print("\nTemperature Forecasting Project Completed!")
print("We've implemented:")
print("1. Data preparation and preprocessing")
print("2. Common-sense baselines (persistence and seasonal)")
print("3. A basic machine learning approach (dense neural network)")
print("4. RNN baselines (Simple RNN, LSTM, GRU)")
print("5. Advanced techniques for RNNs:")
print("   - Recurrent dropout to fight overfitting")
print("   - Stacking recurrent layers")
print("   - Using bidirectional RNNs")
print("   - Advanced architectures with batch normalization and residual connections")
print("\nBased on the results, we can see the progression in model performance as we add more sophisticated techniques.")