import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt

# 1. Preparing the data
print("Loading Reuters dataset...")
max_words = 10000  # Consider only the top 10,000 words in the dataset
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)
print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples")

# Check the number of classes in the dataset
num_classes = max(y_train) + 1
print(f"Number of categories: {num_classes}")

# Convert data to vectorized format
def vectorize_sequences(sequences, dimension=max_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for word_index in sequence:
            results[i, word_index] = 1.
    return results

# Vectorize the data
x_train_vectorized = vectorize_sequences(x_train)
x_test_vectorized = vectorize_sequences(x_test)

# Convert labels to categorical format (one-hot encoding)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# 2. Building your network
print("\nBuilding the network...")
# Create a model with sufficiently large intermediate layers
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(max_words,)))
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# 3. Validating your approach
print("\nValidating approach with a validation set...")
# Set aside a validation set
x_val = x_train_vectorized[:1000]
y_val = y_train_categorical[:1000]
x_train_partial = x_train_vectorized[1000:]
y_train_partial = y_train_categorical[1000:]

# Train with validation
history = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=2
)

# Visualize training results
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Uncomment to display plots in a notebook environment
# plot_training_history(history)

# 4. Generating predictions on new data
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(x_test_vectorized, y_test_categorical)
print(f"Test accuracy: {test_acc:.4f}")

# Generate predictions
predictions = model.predict(x_test_vectorized)
print("Example prediction (probability distribution):", predictions[0][:5], "...")
print("Predicted class:", np.argmax(predictions[0]))
print("Actual class:", y_test[0])

# 5. A different way to handle the labels and loss
# Instead of categorical_crossentropy with one-hot encoding, 
# we can use sparse_categorical_crossentropy
print("\nAlternative approach: using sparse categorical crossentropy...")
model_sparse = Sequential()
model_sparse.add(Dense(512, activation='relu', input_shape=(max_words,)))
model_sparse.add(Dropout(0.5))
model_sparse.add(Dense(256, activation='relu'))
model_sparse.add(Dropout(0.5))
model_sparse.add(Dense(num_classes, activation='softmax'))

model_sparse.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Different loss function
    metrics=['accuracy']
)

# Train with original (non-one-hot) labels
x_val = x_train_vectorized[:1000]
y_val_sparse = y_train[:1000]  # Original integer labels
x_train_partial = x_train_vectorized[1000:]
y_train_partial_sparse = y_train[1000:]  # Original integer labels

# Just train for a few epochs to demonstrate
history_sparse = model_sparse.fit(
    x_train_partial,
    y_train_partial_sparse,
    epochs=5,  # Reduced epochs for demonstration
    batch_size=512,
    validation_data=(x_val, y_val_sparse),
    verbose=2
)

# 6. Visualize a few examples and their predicted labels
def decode_newswire(index, word_index):
    # Reverse the word index dictionary to get words from indices
    reverse_word_index = {value: key for key, value in word_index.items()}
    # Decode the sequence of indices
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in index if i > 3])
    return decoded_newswire

# Load the word index
word_index = reuters.get_word_index()

print("\nExample documents and their classifications:")
for i in range(3):  # Show first 3 examples
    # Get the predicted and actual class
    pred_class = np.argmax(predictions[i])
    actual_class = y_test[i]
    
    # Decode the newswire text (just the beginning for brevity)
    text = decode_newswire(x_test[i][:20], word_index)
    
    print(f"Document {i+1}: '{text}...'")
    print(f"  Predicted class: {pred_class}, Actual class: {actual_class}")
    print()

# 7. Further experiments
print("\nFurther experiments:")
print("1. We could try different architectures with more or fewer layers")
print("2. We could experiment with different dropout rates")
print("3. We could use different word embedding techniques like Word2Vec or GloVe")
print("4. We could experiment with recurrent networks (LSTM, GRU) for this task")
print("5. We could use transfer learning with pre-trained language models")

# 8. Wrapping up
print("\nWrapping up:")
print("- Successfully built a multiclass classification model for Reuters newswires")
print(f"- Achieved {test_acc:.4f} accuracy on the test set")
print("- Demonstrated different approaches to handling labels and loss functions")
print("- Suggested further experiments for improving performance")