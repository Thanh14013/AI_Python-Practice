import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate sine wave data
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, 10 * np.pi, num_samples + seq_length)
    y = np.sin(x)
    
    X = np.zeros((num_samples, seq_length))
    y_target = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        X[i] = y[i:i+seq_length]
        y_target[i] = y[i+seq_length]
    
    return X, y_target

# Parameters
seq_length = 50
num_samples = 1000
inputs_size = 1
hidden_size = 32
output_size = 1
num_epochs = 100
batch_size = 64
learning_rate = 0.01

# Generate data
X, y = generate_sine_wave(seq_length, num_samples)

# Split into train and test sets
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape inputs for Keras [samples, time steps, features]
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Create the GRU model using Keras Sequential API
def create_gru_model():
    model = keras.Sequential([
        # GRU layer
        layers.GRU(hidden_size, 
                  input_shape=(seq_length, inputs_size),
                  return_sequences=False,
                  name='gru_layer'),  # Added name for easier reference
        
        # Output layer
        layers.Dense(output_size, name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

# Create a more complex GRU model with multiple layers and dropout
def create_deep_gru_model():
    model = keras.Sequential([
        # First GRU layer with return sequences for stacking
        layers.GRU(hidden_size, 
                  input_shape=(seq_length, inputs_size),
                  return_sequences=True,
                  name='gru_layer_1'),
        
        # Dropout for regularization
        layers.Dropout(0.2),
        
        # Second GRU layer
        layers.GRU(hidden_size // 2, return_sequences=False, name='gru_layer_2'),
        
        # Dropout for regularization
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(output_size, name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

# Create the model (choose between basic and deep model)
model = create_gru_model()  # or create_deep_gru_model()

# The model needs to be built before we can access its outputs
# Let's build it by calling it with a dummy input
dummy_input = np.zeros((1, seq_length, inputs_size))
_ = model(dummy_input)

# Display the model architecture
model.summary()

# Callbacks for training
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MSE: {test_loss:.6f}')

# Make predictions on test data
test_predictions = model.predict(X_test)

# Function to generate future predictions
def predict_future(model, initial_sequence, steps=200):
    current_sequence = initial_sequence.copy()
    predictions = []
    
    for _ in range(steps):
        # Reshape for prediction
        current_inputs = current_sequence.reshape(1, seq_length, 1)
        
        # Get next predicted value
        next_pred = model.predict(current_inputs, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence for next prediction (remove oldest, add new prediction)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return np.array(predictions)

# Get initial sequence from last test example
initial_sequence = X_test[-1].flatten()

# Generate future predictions
future_predictions = predict_future(model, initial_sequence)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))

# Plot test data
actual_indices = np.arange(len(y_test))
plt.plot(actual_indices, y_test, label='Ground Truth')

# Plot predictions
pred_indices = np.arange(len(y_test), len(y_test) + len(future_predictions))
plt.plot(pred_indices, future_predictions, label='GRU Predictions', color='red')

# Add separator line
plt.axvline(x=len(y_test), color='k', linestyle='--')
plt.title('GRU: Sine Wave Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Analyze and visualize GRU internal representations
# Find the GRU layer by name instead of type
gru_layer = model.get_layer('gru_layer')  # Use the name we gave to the GRU layer

# Create a model that outputs the GRU layer activations
activation_model = keras.Model(
    inputs=model.input,
    outputs=gru_layer.output
)

# Get hidden states for a few test samples
test_samples = X_test[:10]
hidden_states = activation_model.predict(test_samples)

# Visualize hidden states using PCA (for complex high-dimensional states)
from sklearn.decomposition import PCA

# Depending on return_sequences, the shape will be different
if len(hidden_states.shape) == 3:  # If return_sequences=True
    # Get the last time step for each sequence
    hidden_states = hidden_states[:, -1, :]

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_states = pca.fit_transform(hidden_states)

# Plot the reduced representations
plt.figure(figsize=(8, 6))
plt.scatter(reduced_states[:, 0], reduced_states[:, 1])
for i in range(len(reduced_states)):
    plt.annotate(str(i), (reduced_states[i, 0], reduced_states[i, 1]))
plt.title('PCA of GRU Hidden States')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Create a custom prediction function that returns both the prediction and hidden state
# This helps us understand how the model processes sequential information
def predict_with_attention(model, sequence):
    """Make a prediction and return the hidden states to analyze attention"""
    # Get the GRU layer by name
    gru_layer = model.get_layer('gru_layer')
    
    # Create a model that returns both the prediction and the hidden state
    attention_model = keras.Model(
        inputs=model.input,
        outputs=[model.output, gru_layer.output]
    )
    
    # Make prediction
    sequence_reshaped = sequence.reshape(1, seq_length, 1)
    prediction, hidden_states = attention_model.predict(sequence_reshaped, verbose=0)
    
    return prediction[0, 0], hidden_states

# Visualize how the GRU attends to different parts of the inputs sequence
sample_sequence = X_test[0]
pred, hidden_states = predict_with_attention(model, sample_sequence)

# If return_sequences=True, we have hidden states for each time step
if len(hidden_states.shape) == 3:
    # Calculate the L2 norm of hidden states at each time step as a proxy for "importance"
    importance = np.linalg.norm(hidden_states[0], axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sample_sequence)
    plt.title('inputs Sequence')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(importance)), importance)
    plt.title('Importance of Each Time Step (L2 Norm of Hidden States)')
    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.tight_layout()

plt.show()