import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
DATASET_PATH = "Dataset/dataset_capec_transfer.csv"
RESULTS_DIR = "Results/lstm_transfer"
MAX_WORDS = 10000 # Adjust based on vocabulary size
MAX_SEQUENCE_LENGTH = 128 # Adjust based on typical sequence length
EMBEDDING_DIM = 100 # Adjust based on desired embedding size

class LSTMTransferModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.num_classes = 0
        self.results_dir = RESULTS_DIR

    def load_data(self):
        print(f"Loading data from {DATASET_PATH}")
        
        chunks = []
        chunksize = 10000  # Define chunk size
        try:
            for chunk in pd.read_csv(DATASET_PATH, chunksize=chunksize, engine='python'):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"Error reading CSV file in chunks: {e}")
            # Fallback or raise error
            raise e # Re-raise the exception after printing

        texts = df['text'].values
        labels = df['label'].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Found {self.num_classes} unique labels.")

        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

        # Convert labels to one-hot encoding (necessary for multi-class with categorical crossentropy)
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)

        return X_train, X_test, y_train_one_hot, y_test_one_hot

    def build_model(self, existing_model=None):
        # --- Transfer Learning Logic Placeholder ---
        # If existing_model is provided, load its weights and potentially modify the top layers.
        # Otherwise, build a new model from scratch.
        # For this example, we build a model from scratch for multi-class classification.
        # You would typically load a pre-trained model here and adapt it.
        print("Building LSTM model for transfer learning (multi-class classification)")

        model = Sequential([
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            LSTM(64, return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            # Output layer changed for multi-class classification
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        print("Model built successfully.")
        self.model.summary()

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        print("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1 # Using a split of training data for validation during training
        )
        print("Training finished.")
        return history

    def evaluate(self, X_test, y_test_one_hot):
        print("Evaluating LSTM model...")
        # Use the raw encoded labels for confusion matrix calculation
        y_test_encoded = np.argmax(y_test_one_hot, axis=1)
        
        loss, accuracy = self.model.evaluate(X_test, y_test_one_hot, verbose=0)
        
        # Get predictions
        y_pred_one_hot = self.model.predict(X_test)
        y_pred_encoded = np.argmax(y_pred_one_hot, axis=1)

        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Return true labels and predictions for confusion matrix
        return loss, accuracy, y_test_encoded, y_pred_encoded

    def save_results(self, loss, accuracy, true_labels, predictions):
        os.makedirs(self.results_dir, exist_ok=True)
        results_path = os.path.join(self.results_dir, "evaluation_results.pkl")
        results = {'loss': loss, 'accuracy': accuracy}
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_path}")

        # Save label encoder and tokenizer
        tokenizer_path = os.path.join(self.results_dir, "tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        label_encoder_path = os.path.join(self.results_dir, "label_encoder.pkl")
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Generate and save confusion matrix plot
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_plot_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path)
        print(f"Confusion matrix plot saved to {cm_plot_path}")
        plt.close()

    def run(self):
        X_train, X_test, y_train, y_test_one_hot = self.load_data()
        self.build_model()
        self.train(X_train, y_train)
        
        # Get true labels and predictions from evaluate
        loss, accuracy, y_test_encoded, y_pred_encoded = self.evaluate(X_test, y_test_one_hot)
        
        # Pass them to save_results
        self.save_results(loss, accuracy, y_test_encoded, y_pred_encoded)

if __name__ == "__main__":
    # Example of how to run this script directly
    # In a real transfer learning scenario, you might load a pre-trained model here
    # For simplicity, this example builds and trains from scratch on the transfer dataset
    lstm_transfer = LSTMTransferModel()
    lstm_transfer.run() 