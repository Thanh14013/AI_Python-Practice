import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(dataset_path):
    """Load and prepare the dataset"""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    texts = df['text'].astype(str).values
    labels = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Total samples: {len(texts)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return texts, encoded_labels, label_encoder

def preprocess_text_lstm(texts, tokenizer=None, max_words=10000, max_len=150):
    """Tokenize and preprocess text data for LSTM"""
    if tokenizer is None:
         tokenizer = Tokenizer(num_words=max_words)
         tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return padded_sequences, tokenizer, max_words, max_len

def build_lstm_model(max_words, max_len, num_classes):
    """Build the LSTM model"""
    print("Building LSTM model...")
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    
    print(model.summary())
    return model

def train_lstm_model(model, X_train_pad, y_train, X_val_pad, y_val, epochs=10, batch_size=32):
    """Train the LSTM model"""
    print("\n" + "="*50)
    print("Training LSTM model...")
    print("="*50)
    
    # Add Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=3, # Stop after 3 epochs with no improvement
        restore_best_weights=True # Restore best weights
    )
    
    start_time = time.time()
    history = model.fit(
        X_train_pad, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_pad, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"LSTM model training completed in {training_time:.2f} seconds")
    
    return history, training_time

def evaluate_lstm_model(model, X_test_pad, y_test):
    """Evaluate the LSTM model"""
    print("\n" + "="*50)
    print("Evaluating LSTM model...")
    print("="*50)
    
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    return accuracy, loss, y_pred, y_pred_probs

def save_results(save_dir, accuracy, loss, training_time, y_true, y_pred, class_names):
    """Save model, tokenizer, label encoder, and results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results to pickle file
    results = {
        'accuracy': accuracy,
        'loss': loss,
        'training_time': training_time,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names
    }
    
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {os.path.join(save_dir, 'results.pkl')}")

def main():
    """Main function to run the LSTM model on CAPEC dataset"""
    print("=" * 80)
    print("Running LSTM model on CAPEC dataset")
    print("=" * 80)
    
    # Define dataset path and result directory
    dataset_path = 'Dataset/dataset_capec_transfer.csv' # Change to _test.csv for test dataset
    model_result_dir = 'Results/lstm_capec_transfer'      # Change directory name accordingly
    
    # Load data
    texts, encoded_labels, label_encoder = load_data(dataset_path)
    
    # Split data into train and test sets (stratified)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Split training data further for validation (stratified)
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Preprocess text (tokenize and pad) using tokenizer fitted on training data
    X_train_pad, tokenizer, max_words, max_len = preprocess_text_lstm(X_train_text)
    X_val_pad, _, _, _ = preprocess_text_lstm(X_val_text, tokenizer=tokenizer, max_words=max_words, max_len=max_len)
    X_test_pad, _, _, _ = preprocess_text_lstm(X_test_text, tokenizer=tokenizer, max_words=max_words, max_len=max_len)
    
    # Build LSTM model
    num_classes = len(label_encoder.classes_)
    lstm_model = build_lstm_model(max_words, max_len, num_classes)
    
    # Train LSTM model
    history, training_time = train_lstm_model(lstm_model, X_train_pad, y_train, X_val_pad, y_val, epochs=10)
    
    # Evaluate LSTM model
    accuracy, loss, y_pred, y_pred_probs = evaluate_lstm_model(lstm_model, X_test_pad, y_test)
    
    # Save results
    save_results(model_result_dir, accuracy, loss, training_time, y_test, y_pred, label_encoder.classes_)
    
    print("=" * 80)
    print(f"LSTM model results saved to: {model_result_dir}")
    print("=" * 80)

if __name__ == "__main__":
    # Fix TensorFlow memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main() 