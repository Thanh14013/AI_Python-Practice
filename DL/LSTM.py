import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# PART 1: Load and preprocess text data
# For demonstration, let's create a simple sentiment analysis dataset
# In a real scenario, you would load your own dataset

def create_sample_data(n_samples=1000):
    """Create a sample sentiment analysis dataset"""
    # Create positive examples
    positive_texts = [
        "I loved this movie, it was fantastic!",
        "The service was excellent and staff were friendly",
        "This is the best product I've ever bought",
        "I had an amazing experience, would recommend",
        "The quality exceeded my expectations",
    ]
    
    # Create negative examples
    negative_texts = [
        "This was a terrible experience, very disappointed",
        "The product broke after two days, waste of money",
        "Worst customer service I've ever encountered",
        "I would not recommend this to anyone",
        "The quality was poor and not worth the price",
    ]
    
    # Generate more examples by adding random variations
    texts = []
    labels = []
    
    for _ in range(n_samples // 2):
        # Positive examples
        base = np.random.choice(positive_texts)
        texts.append(base + " " + np.random.choice(["Highly recommend.", "Great experience.", "Very satisfied.", "Will buy again.", ""]))
        labels.append(1)  # Positive label
        
        # Negative examples
        base = np.random.choice(negative_texts)
        texts.append(base + " " + np.random.choice(["Avoid at all costs.", "Very unhappy.", "Would not buy again.", "Complete waste of time.", ""]))
        labels.append(0)  # Negative label
    
    # Create DataFrame
    data = pd.DataFrame({
        'text': texts,
        'sentiment': labels
    })
    
    # Shuffle the data
    return data.sample(frac=1).reset_index(drop=True)

# Create or load your dataset
data = create_sample_data(1000)
print(f"Dataset shape: {data.shape}")
print(data.head())

# PART 2: Text preprocessing and tokenization
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42
)

# Tokenize the text
max_features = 5000  # Maximum number of words to keep
max_len = 100  # Maximum sequence length

tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Padded training data shape: {X_train_pad.shape}")

# PART 3: Build the LSTM model
embedding_dim = 100  # Dimension of the embedding layer

model = Sequential([
    # Embedding layer converts integers to dense vectors of fixed size
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len),
    
    # LSTM layer
    LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    
    # Second LSTM layer for deeper representation
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
    
    # Dense hidden layer
    Dense(units=32, activation='relu'),
    Dropout(0.4),
    
    # Output layer - binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# PART 4: Train the model
# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_pad, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# PART 5: Evaluate the model
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# PART 6: Make predictions on new text
def predict_sentiment(text, model, tokenizer, max_len):
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded)[0][0]
    
    # Return prediction
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': float(confidence),
        'raw_score': float(prediction)
    }

# Example usage
if __name__ == "__main__":
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Example predictions
    sample_texts = [
        "This product exceeded all my expectations, I'm very satisfied!",
        "The customer service was terrible and the product didn't work.",
        "It was okay, not great but not terrible either."
    ]
    
    print("\nSample Predictions:")
    for text in sample_texts:
        result = predict_sentiment(text, model, tokenizer, max_len)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")