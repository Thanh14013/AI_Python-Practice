import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# You will need to install transformers library for BERT
# pip install transformers tensorflow
from transformers import TFBertForSequenceClassification, BertTokenizer
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

def preprocess_text_bert(texts, tokenizer, max_len=128):
    """Tokenize and preprocess text data for BERT"""
    input_ids = []
    attention_masks = []

    print(f"Tokenizing and padding text data with max_len={max_len}...")
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                        # Sentence to encode.
            add_special_tokens = True,   # Add '[CLS]' and '[SEP]'
            max_length = max_len,        # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,  # Construct attention masks.
            return_tensors = 'tf',     # Return tensorflow tensors.
        )
        
        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert lists to tensors
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    
    return input_ids, attention_masks

def build_bert_model(num_classes):
    """Build the BERT model for sequence classification"""
    print("Building BERT model...")
    # Load pre-trained BERT base model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    
    # Compile the model (using AdamW optimizer recommended for BERT)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Note: BERT models from transformers typically output logits, not softmax
    # Use from_logits=True in the loss function.
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    print(model.summary())
    return model

def train_bert_model(model, train_inputs, train_labels, val_inputs, val_labels, epochs=3, batch_size=32):
    """Train the BERT model"""
    print("\n" + "="*50)
    print("Training BERT model...")
    print("="*50)
    
    # Create tf.data.Dataset for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices(train_inputs).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_inputs).batch(batch_size)
    
    # BERT models from transformers require inputs as dictionaries/tuples
    # Adjusting input format for model.fit
    train_inputs_dict = {'input_ids': train_inputs[0], 'attention_mask': train_inputs[1]}
    val_inputs_dict = {'input_ids': val_inputs[0], 'attention_mask': val_inputs[1]}

    start_time = time.time()
    history = model.fit(
        [train_inputs[0], train_inputs[1]], train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([val_inputs[0], val_inputs[1]], val_labels) # Correct validation data format
    )
    training_time = time.time() - start_time
    print(f"BERT model training completed in {training_time:.2f} seconds")
    
    return history, training_time

def evaluate_bert_model(model, test_inputs, test_labels):
    """Evaluate the BERT model"""
    print("\n" + "="*50)
    print("Evaluating BERT model...")
    print("="*50)
    
    loss, accuracy = model.evaluate(
        [test_inputs[0], test_inputs[1]], test_labels, verbose=1 # Correct evaluation data format
    )
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions
    predictions = model.predict([test_inputs[0], test_inputs[1]]) # Correct prediction input format
    # BERT model outputs logits, apply softmax to get probabilities
    predicted_probs = tf.nn.softmax(predictions.logits, axis=-1).numpy()
    y_pred = np.argmax(predicted_probs, axis=1)
    
    return accuracy, loss, y_pred, predicted_probs

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
    """Main function to run the BERT model on CAPEC dataset"""
    print("=" * 80)
    print("Running BERT model on CAPEC dataset")
    print("=" * 80)
    
    # Define dataset path and result directory
    dataset_path = 'Dataset/dataset_capec_transfer.csv' # Change to _test.csv for test dataset
    model_result_dir = 'Results/bert_capec_transfer'      # Change directory name accordingly
    
    # Load data
    texts, encoded_labels, label_encoder = load_data(dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels # Use stratify
    )
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Preprocess text for BERT
    max_len = 128 # Max length for BERT input
    train_input_ids, train_attention_masks = preprocess_text_bert(X_train, tokenizer, max_len)
    test_input_ids, test_attention_masks = preprocess_text_bert(X_test, tokenizer, max_len)
    
    # Combine inputs into tuples/lists for model
    train_inputs = (train_input_ids, train_attention_masks)
    test_inputs = (test_input_ids, test_attention_masks)
    
    # Build BERT model
    num_classes = len(label_encoder.classes_)
    bert_model = build_bert_model(num_classes)
    
    # Train BERT model
    # Create a validation set from the training data for monitoring
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train # Use a split of X_train
    )
    val_input_ids, val_attention_masks = preprocess_text_bert(val_texts, tokenizer, max_len)
    val_inputs = (val_input_ids, val_attention_masks)

    # Ensure correct input format for train_bert_model
    train_inputs_fit = (train_input_ids, train_attention_masks)
    val_inputs_fit = (val_input_ids, val_attention_masks)

    history, training_time = train_bert_model(bert_model, train_inputs_fit, train_labels, val_inputs_fit, val_labels, epochs=3)
    
    # Evaluate BERT model
    accuracy, loss, y_pred, predicted_probs = evaluate_bert_model(bert_model, test_inputs, y_test)
    
    # Save results
    save_results(model_result_dir, accuracy, loss, training_time, y_test, y_pred, label_encoder.classes_)
    
    print("=" * 80)
    print(f"BERT model results saved to: {model_result_dir}")
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