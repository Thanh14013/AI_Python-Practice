import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
# Load the IMDB dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Print dataset information
print(f"Training data: {len(x_train)} examples")
print(f"Testing data: {len(x_test)} examples")
print(f"Example review as word indices: {x_train[0][:20]}...")

# Preprocess the data
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    # Set specific indices of results[i] to 1s
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

# Vectorize the training and test data
print("Vectorizing sequences...")
x_train_vec = vectorize_sequences(x_train)
x_test_vec = vectorize_sequences(x_test)

# Convert labels to float
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print(f"Training data shape after vectorization: {x_train_vec.shape}")
print(f"Example vectorized review: {x_train_vec[0][:20]}...")

# Build the neural network model as described in the text
def build_model():
    model = keras.Sequential([
        # Two intermediate layers with 16 hidden units each, using ReLU activation
        layers.Dense(16, activation='relu', input_shape=(10000,)),
        layers.Dense(16, activation='relu'),
        # Final layer with sigmoid activation for binary classification
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create the model
model = build_model()
model.summary()

# Set aside a validation set
x_val = x_train_vec[:10000]
partial_x_train = x_train_vec[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train the model
print("Training the model...")
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# Plot the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test data
results = model.evaluate(x_test_vec, y_test)
print(f"Test loss: {results[0]:.3f}")
print(f"Test accuracy: {results[1]:.3f}")

# Function to decode a review from integer indices back to words
def decode_review(encoded_review):
    # Load the word index mapping
    word_index = keras.datasets.imdb.get_word_index()
    # Reverse the mapping
    reverse_word_index = {value: key for key, value in word_index.items()}
    # Add special tokens
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    reverse_word_index[3] = '<UNUSED>'
    # Decode the review
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Make predictions on new data
def predict_sentiment(review_text, model, word_index=None):
    if word_index is None:
        word_index = keras.datasets.imdb.get_word_index()
    
    # Convert review text to integers
    words = review_text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words if word_index.get(word, 2) < 10000]
    
    # Vectorize the review
    vectorized_review = vectorize_sequences([encoded_review])
    
    # Make prediction
    prediction = model.predict(vectorized_review)[0][0]
    
    return prediction, "Positive" if prediction > 0.5 else "Negative"

# Example usage of the prediction function
word_index = keras.datasets.imdb.get_word_index()
sample_review = "This movie was great! I really enjoyed it and would recommend it to anyone."
prediction, sentiment = predict_sentiment(sample_review, model, word_index)
print(f"Sample review: {sample_review}")
print(f"Sentiment prediction: {sentiment} (score: {prediction:.3f})")

# Load and print a few examples from the test set
print("\nExamples from test set:")
for i in range(3):
    decoded_review = decode_review(x_test[i])
    prediction, sentiment = predict_sentiment(decoded_review, model, word_index)
    print(f"Review: {decoded_review[:100]}...")
    print(f"True sentiment: {'Positive' if y_test[i] == 1 else 'Negative'}")
    print(f"Predicted sentiment: {sentiment} (score: {prediction:.3f})")
    print("-" * 80)