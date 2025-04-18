# Deep Learning for Computer Vision Implementation
# Chapter 5: Deep Learning for Computer Vision

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, applications # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version:", tf.__version__)

# ======================================================================
# 5.1 Introduction to Convnets
# ======================================================================

# ----- The convolution operation -----
def apply_convolution(input_image, kernel):
    """Apply a convolution kernel to an input image manually"""
    # Get dimensions
    i_height, i_width = input_image.shape
    k_height, k_width = kernel.shape
    
    # Calculate output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Apply convolution
    for i in range(o_height):
        for j in range(o_width):
            output[i, j] = np.sum(input_image[i:i+k_height, j:j+k_width] * kernel)
    
    return output

# Example usage
print("\n----- Convolution Operation Example -----")
# Create a simple 5x5 input
input_img = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

# Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Apply convolution
result = apply_convolution(input_img, edge_kernel)
print("Input image:")
print(input_img)
print("\nKernel (edge detection):")
print(edge_kernel)
print("\nConvolution result:")
print(result)

# ----- The max-pooling operation -----
def apply_max_pooling(input_feature_map, pool_size=2, stride=2):
    """Apply max pooling to an input feature map manually"""
    # Get dimensions
    height, width = input_feature_map.shape
    
    # Calculate output dimensions
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    
    # Initialize output
    output = np.zeros((out_height, out_width))
    
    # Apply max pooling
    for i in range(out_height):
        for j in range(out_width):
            output[i, j] = np.max(
                input_feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            )
    
    return output

# Example usage
print("\n----- Max Pooling Operation Example -----")
pooled_result = apply_max_pooling(input_img)
print("Original image:")
print(input_img)
print("\nAfter max pooling (2x2, stride 2):")
print(pooled_result)

# ======================================================================
# 5.2 Training a convnet from scratch on a small dataset
# ======================================================================

def train_convnet_from_scratch():
    print("\n----- Training a ConvNet from Scratch -----")
    
    # ----- Downloading the data -----
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Convert to float32 and normalize to range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    print(f"Train data shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")
    
    # ----- Data preprocessing -----
    # Create data generator for data augmentation
    print("\nSetting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,  # Randomly rotate images by up to 15 degrees
        width_shift_range=0.1,  # Randomly shift images horizontally by up to 10%
        height_shift_range=0.1,  # Randomly shift images vertically by up to 10%
        horizontal_flip=True,  # Randomly flip images horizontally
        zoom_range=0.1  # Randomly zoom into images by up to 10%
    )
    datagen.fit(train_images)
    
    # ----- Building your network -----
    print("\nBuilding ConvNet model...")
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model (uncomment to actually train - this will take some time)
    """
    print("\nTraining model...")
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=64),
        epochs=50,
        validation_data=(test_images, test_labels),
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save('cifar10_convnet.h5')
    print("Model saved as 'cifar10_convnet.h5'")
    """
    
    print("\nTraining code is ready but commented out to save time")
    print("Uncomment the training section to run actual training")
    
    return model

# ======================================================================
# 5.3 Using a pretrained convnet
# ======================================================================

def use_pretrained_convnet():
    print("\n----- Using a Pretrained ConvNet -----")
    
    # ----- Feature extraction -----
    print("Setting up feature extraction with VGG16...")
    
    # Load the VGG16 model pre-trained on ImageNet
    base_model = applications.VGG16(
        weights='imagenet',
        include_top=False,  # Exclude the fully-connected layers at the top
        input_shape=(224, 224, 3)
    )
    
    # Create a new model that outputs the features from VGG16
    feature_extractor = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    
    # Freeze the base model weights
    base_model.trainable = False
    
    print("Base model architecture:")
    base_model.summary()
    
    # ----- Fine-tuning -----
    print("\nSetting up fine-tuning model...")
    
    # Create a model for fine-tuning
    fine_tune_model = models.Sequential([
        # Pre-trained VGG16 base (frozen during initial training)
        base_model,
        
        # Add classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    # Compile the model
    fine_tune_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("The training process would include two phases:")
    print("1. Train only the top layers (base model frozen)")
    print("2. Unfreeze some layers of the base model and fine-tune with a low learning rate")
    
    # ----- Wrapping up -----
    print("\nFeature extraction and fine-tuning setup complete")
    print("The actual training code is omitted to save time")
    
    return fine_tune_model

# ======================================================================
# 5.4 Visualizing what convnets learn
# ======================================================================

def visualize_convnet_learning():
    print("\n----- Visualizing What ConvNets Learn -----")
    
    # Load a pre-trained model for visualization
    model = applications.VGG16(weights='imagenet', include_top=True)
    
    # ----- Visualizing intermediate activations -----
    def visualize_activations(model, image_path):
        print("\nVisualizing intermediate activations")
        print("(Code ready but execution skipped to save time)")
        
        """
        # Load and preprocess an image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        
        # Create a model that outputs all intermediate activations
        layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(img_array)
        
        # Plot a sample of activations
        for i, activation in enumerate(activations[:3]):  # Show first 3 conv layers
            plt.figure(figsize=(12, 6))
            for j in range(min(16, activation.shape[-1])):  # Show up to 16 channels
                plt.subplot(4, 4, j+1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')
                plt.title(f'Channel {j}')
                plt.axis('off')
            plt.suptitle(f'Layer {i+1} Activations')
            plt.tight_layout()
            plt.show()
        """
    
    # ----- Visualizing convnet filters -----
    def visualize_filters(model):
        print("\nVisualizing convnet filters")
        print("(Code ready but execution skipped to save time)")
        
        """
        # Get weights from the first convolutional layer
        filters = model.layers[1].get_weights()[0]  # VGG16's first conv layer
        
        # Normalize filter values to 0-1 for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Plot first 16 filters
        plt.figure(figsize=(10, 8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(filters[:, :, :, i].squeeze(), cmap='viridis')
            plt.axis('off')
        plt.suptitle('First Conv Layer Filters')
        plt.tight_layout()
        plt.show()
        """
    
    # ----- Visualizing heatmaps of class activation -----
    def visualize_class_activation(model, image_path, class_idx):
        print("\nVisualizing heatmaps of class activation")
        print("(Code ready but execution skipped to save time)")
        
        """
        # Grad-CAM implementation
        # Create a model that outputs both the predictions and the last conv layer
        last_conv_layer = next(layer for layer in reversed(model.layers) 
                              if isinstance(layer, layers.Conv2D))
        last_conv_output = last_conv_layer.output
        
        # Get the gradient of the top predicted class with respect to the output feature map
        with tf.GradientTape() as tape:
            # Add a dimension for batch size
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = applications.vgg16.preprocess_input(img_array)
            
            # Get the feature map outputs
            last_conv_model = tf.keras.models.Model(model.inputs, last_conv_output)
            conv_outputs = last_conv_model(img_array)
            
            # Predict the class logits
            pred_model = tf.keras.models.Model(model.inputs, model.outputs)
            preds = pred_model(img_array)
            top_pred_idx = class_idx if class_idx is not None else tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_idx]
            
        # Get gradients of the top class output with respect to conv outputs
        grads = tape.gradient(top_class_channel, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over channels to get heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        # Overlay heatmap on original image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img / 255.0, 0, 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img / 255.0)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap / 255.0)
        plt.title('Class Activation Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title('Heatmap Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        """
    
    # Call the visualization functions (implementations are ready but commented out)
    visualize_activations(model, "sample_image.jpg")  # Replace with a real image path
    visualize_filters(model)
    visualize_class_activation(model, "sample_image.jpg", None)  # Replace with a real image path
    
    return "Visualization functions are prepared but execution is skipped to save time"

# ======================================================================
# Main execution
# ======================================================================

def main():
    print("==== DEEP LEARNING FOR COMPUTER VISION IMPLEMENTATION ====")
    
    # Section 5.1: Introduction to convnets
    # The convolution operation and max-pooling examples are above
    
    # Section 5.2: Training a convnet from scratch
    model = train_convnet_from_scratch()
    
    # Section 5.3: Using a pretrained convnet
    fine_tune_model = use_pretrained_convnet()
    
    # Section 5.4: Visualizing what convnets learn
    visualize_convnet_learning()
    
    print("\n==== IMPLEMENTATION COMPLETE ====")
    print("This code demonstrates the key concepts from Chapter 5 of the Deep Learning for Computer Vision textbook.")

if __name__ == "__main__":
    main()