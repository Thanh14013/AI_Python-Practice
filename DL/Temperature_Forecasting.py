# Temperature Forecasting with Recurrent Neural Networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import os

# ----- DATA PREPARATION -----
def download_and_prepare_data(url):
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'jena_climate.csv')
    
    if not os.path.exists(csv_path):
        print("Downloading dataset...")
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
except Exception as e:
    print(f"Error downloading data: {e}")
    # Create synthetic data if download fails
    dates = pd.date_range(start='2009-01-01', end='2016-12-31', freq='10min')
    n = len(dates)
    hourly_temp = 15 + 10 * np.sin(np.linspace(0, 2*n*np.pi/144, n))
    yearly_temp = 5 * np.sin(np.linspace(0, 2*np.pi, n))
    noise = np.random.normal(0, 1, n)
    temp = hourly_temp + yearly_temp + noise
    
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
    print("\nData exploration:")
    print(f"Dataset shape: {df.shape}")
    print(f"Time period: {df['Date Time'].iloc[0]} to {df['Date Time'].iloc[-1]}")
    print("\nBasic statistics for temperature:")
    temp_data = df['T (degC)']
    print(f"Mean: {temp_data.mean():.2f}°C")
    print(f"Max: {temp_data.max():.2f}°C")
    print(f"Min: {temp_data.min():.2f}°C")
    print(f"Standard deviation: {temp_data.std():.2f}°C")
    
    plt.figure(figsize=(15, 5))
    sample_data = temp_data[:1008]
    plt.plot(range(len(sample_data)), sample_data)
    plt.title('Temperature data for the first week')
    plt.xlabel('Time steps (10-minute intervals)')
    plt.ylabel('Temperature (°C)')

# Data preprocessing for time series forecasting
def preprocess_data(df, feature_cols=None, target_col='T (degC)', 
                   lookback=144, delay=144, batch_size=128, 
                   step=6, normalize=True):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'Date Time']
    
    features = df[feature_cols].values
    target = df[target_col].values
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features)
        
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target = target_scaler.fit_transform(target.reshape(-1, 1))
    else:
        scaler = None
        target_scaler = None
        target = target.reshape(-1, 1)
    
    def generate_sequences(data, target, lookback, delay, step):
        X, y = [], []
        max_idx = len(data) - delay - lookback
        
        for i in range(0, max_idx, step):
            X.append(data[i:i + lookback])
            y.append(target[i + lookback + delay - 1])
        
        return np.array(X), np.array(y)
    
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    X_train, y_train = generate_sequences(
        features[:train_end], target[:train_end], lookback, delay, step)
    
    X_val, y_val = generate_sequences(
        features[train_end:val_end], target[train_end:val_end], lookback, delay, step)
    
    X_test, y_test = generate_sequences(
        features[val_end:], target[val_end:], lookback, delay, step)
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

explore_data(df)

X_train, y_train, X_val, y_val, X_test, y_test, temp_scaler = preprocess_data(
    df, 
    feature_cols=['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)'],
    target_col='T (degC)',
    lookback=144,
    delay=144,
    step=6,
    normalize=True
)

def evaluate_model(y_true, y_pred, model_name, scaler=None):
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

def persistence_baseline(X_test, y_test, scaler=None):
    y_pred = X_test[:, -1, 0].reshape(-1, 1)
    return evaluate_model(y_test, y_pred, "Persistence Baseline", scaler)

def seasonal_baseline(X_test, y_test, lookback, scaler=None):
    steps_per_day = 24
    if lookback >= steps_per_day:
        y_pred = X_test[:, -steps_per_day, 0].reshape(-1, 1)
    else:
        y_pred = X_test[:, 0, 0].reshape(-1, 1)
        
    return evaluate_model(y_test, y_pred, "Seasonal Baseline", scaler)

persistence_mae, persistence_rmse = persistence_baseline(X_test, y_test, temp_scaler)
seasonal_mae, seasonal_rmse = seasonal_baseline(X_test, y_test, 144, temp_scaler)

def dense_model(input_shape):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

dense_nn = dense_model((X_train.shape[1], X_train.shape[2]))
print(dense_nn.summary())

dense_history = dense_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

dense_pred = dense_nn.predict(X_test)
dense_mae, dense_rmse = evaluate_model(y_test, dense_pred, "Dense Neural Network", temp_scaler)

# Simple RNN model definitions
def simple_rnn_model(input_shape):
    model = keras.Sequential([
        layers.SimpleRNN(32, input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def rnn_with_dropout(input_shape, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    model = keras.Sequential([
        layers.SimpleRNN(32, 
                         dropout=dropout_rate,
                         recurrent_dropout=recurrent_dropout_rate,
                         input_shape=input_shape),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def stacked_rnn_model(input_shape, dropout_rate=0.2):
    model = keras.Sequential([
        layers.SimpleRNN(32, 
                        dropout=dropout_rate,
                        return_sequences=True,
                        input_shape=input_shape),
        layers.SimpleRNN(16, dropout=dropout_rate),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def bidirectional_rnn_model(input_shape, dropout_rate=0.2):
    model = keras.Sequential([
        layers.Bidirectional(
            layers.SimpleRNN(32, dropout=dropout_rate), 
            input_shape=input_shape
        ),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def advanced_rnn_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Bidirectional(layers.SimpleRNN(64, 
                                            dropout=0.2, 
                                            return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    
    residual = x
    x = layers.Bidirectional(layers.SimpleRNN(32, 
                                            dropout=0.2, 
                                            return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    
    residual = layers.TimeDistributed(layers.Dense(64))(residual)
    x = layers.add([x, residual])
    
    x = layers.SimpleRNN(32, dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

input_shape = (X_train.shape[1], X_train.shape[2])

# Train basic RNN
simple_rnn = simple_rnn_model(input_shape)
print(simple_rnn.summary())

simple_rnn_history = simple_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Train RNN with dropout
rnn_dropout = rnn_with_dropout(input_shape, 0.2, 0.2)
print(rnn_dropout.summary())

rnn_dropout_history = rnn_dropout.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Train stacked RNN
stacked_rnn = stacked_rnn_model(input_shape, 0.2)
print(stacked_rnn.summary())

stacked_rnn_history = stacked_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Train bidirectional RNN
bidirectional_rnn = bidirectional_rnn_model(input_shape, 0.2)
print(bidirectional_rnn.summary())

bidirectional_rnn_history = bidirectional_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Train advanced RNN
advanced_rnn = advanced_rnn_model(input_shape)
print(advanced_rnn.summary())

advanced_rnn_history = advanced_rnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate models
simple_rnn_pred = simple_rnn.predict(X_test)
simple_rnn_mae, simple_rnn_rmse = evaluate_model(
    y_test, simple_rnn_pred, "Simple RNN", temp_scaler)

rnn_dropout_pred = rnn_dropout.predict(X_test)
rnn_dropout_mae, rnn_dropout_rmse = evaluate_model(
    y_test, rnn_dropout_pred, "RNN with Dropout", temp_scaler)

stacked_rnn_pred = stacked_rnn.predict(X_test)
stacked_rnn_mae, stacked_rnn_rmse = evaluate_model(
    y_test, stacked_rnn_pred, "Stacked RNN", temp_scaler)

bidirectional_rnn_pred = bidirectional_rnn.predict(X_test)
bidirectional_rnn_mae, bidirectional_rnn_rmse = evaluate_model(
    y_test, bidirectional_rnn_pred, "Bidirectional RNN", temp_scaler)

advanced_rnn_pred = advanced_rnn.predict(X_test)
advanced_rnn_mae, advanced_rnn_rmse = evaluate_model(
    y_test, advanced_rnn_pred, "Advanced RNN", temp_scaler)

# Compare model performance
models = [
    "Persistence Baseline",
    "Seasonal Baseline",
    "Dense Neural Network",
    "Simple RNN",
    "RNN with Dropout",
    "Stacked RNN",
    "Bidirectional RNN",
    "Advanced RNN"
]

mae_values = [
    persistence_mae,
    seasonal_mae,
    dense_mae,
    simple_rnn_mae,
    rnn_dropout_mae,
    stacked_rnn_mae,
    bidirectional_rnn_mae,
    advanced_rnn_mae
]

rmse_values = [
    persistence_rmse,
    seasonal_rmse,
    dense_rmse,
    simple_rnn_rmse,
    rnn_dropout_rmse,
    stacked_rnn_rmse,
    bidirectional_rnn_rmse,
    advanced_rnn_rmse
]

results_df = pd.DataFrame({
    'Model': models,
    'MAE': mae_values,
    'RMSE': rmse_values
})

print("\nModel Performance Summary:")
print(results_df.sort_values('MAE'))

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.barh(models, mae_values)
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('MAE')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.barh(models, rmse_values)
plt.title('Root Mean Squared Error (RMSE)')
plt.xlabel('RMSE')
plt.tight_layout()

# Create single combined plot for actual vs predicted temperature
plt.figure(figsize=(15, 6))
time_steps = range(100)
truth = temp_scaler.inverse_transform(y_test[:100])
predictions = temp_scaler.inverse_transform(advanced_rnn_pred[:100])
plt.plot(time_steps, truth, 'b-', label='Actual Temperature')
plt.plot(time_steps, predictions, 'r-', label='Predicted Temperature')
plt.title('Temperature Forecast Comparison (24 hours ahead)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.savefig('temperature_forecast.png')

print("\nTemperature Forecasting completed successfully!")
print(f"Best model: {results_df.sort_values('MAE').iloc[0]['Model']} (MAE: {results_df.sort_values('MAE').iloc[0]['MAE']:.2f})")