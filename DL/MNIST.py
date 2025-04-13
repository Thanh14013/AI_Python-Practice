# MNIST Digit Classification using Keras
# This implementation follows the tutorial shown in the images

# Import necessary libraries
import numpy as np
import tensorflow as tf
from keras import mnist
from keras import models
from keras import layers
from keras import to_categorical

# Step 1: Load the MNIST dataset
# The dataset comes preloaded in Keras and returns training and testing data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Let's print the shapes to understand our data
print("Training data shapes:")
print(f"train_images shape: {train_images.shape}")
print(f"train_labels shape: {train_labels.shape}")
print("Test data shapes:")
print(f"test_images shape: {test_images.shape}")
print(f"test_labels shape: {test_labels.shape}")

# Let's look at a few examples of the labels
print("Sample of training labels:", train_labels[:10])

# Step 2: Prepare the image data
# We need to reshape the data to the format the network expects and scale the values
# Reshape: keep dimensions (28, 28) as is, but make sure values are float
train_images = train_images.reshape((60000, 28, 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28)).astype('float32') / 255

# Step 3: Prepare the labels
# Convert labels to categorical one-hot encoding
# This transforms the digit (0-9) into a 10-element vector with a 1 at the index of the digit
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Original label:", train_labels[0])
print("Shape after categorical encoding:", train_labels.shape)

# Step 4: Build the neural network architecture
# We'll use a Sequential model with two Dense layers
network = models.Sequential()
# First layer: 512 neurons with ReLU activation
# The input shape is (28, 28) which is flattened to 784 inputs
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# Second layer: 10 neurons with softmax activation (for 10 digit classes)
network.add(layers.Dense(10, activation='softmax'))

# Step 5: Compile the network
# We specify the optimizer, loss function, and metrics to monitor
network.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Step 6: Train the network
# Note: There's a discrepancy in the images between first reshaping to (60000, 28, 28)
# and then feeding a flattened input to the network. Let's fix this:
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# Now we fit the model to our training data
history = network.fit(train_images, train_labels, 
                      epochs=5, 
                      batch_size=128)

# Step 7: Evaluate the model on the test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Step 8: Let's make a prediction on a single test image
# Get a single test image
image_index = 0
single_image = test_images[image_index].reshape(1, 28 * 28)

# Make a prediction
prediction = network.predict(single_image)

# The prediction is a 10-element array of probabilities
# We need to find the index with the highest probability
predicted_digit = np.argmax(prediction[0])
actual_digit = np.argmax(test_labels[image_index])

print(f"Predicted digit: {predicted_digit}")
print(f"Actual digit: {actual_digit}")
print(f"Confidence: {prediction[0][predicted_digit]:.4f}")

# Optional: Plot sample images and predictions for visual verification
try:
    import matplotlib.pyplot as plt
    
    # Function to display sample images with their predictions
    def display_sample(num_samples=5):
        plt.figure(figsize=(12, 6))
        for i in range(num_samples):
            # Get a random index
            idx = np.random.randint(0, len(test_images))
            
            # Get image and reshape for display
            img = test_images[idx].reshape(28, 28)
            
            # Make prediction
            pred = np.argmax(network.predict(test_images[idx].reshape(1, 28 * 28))[0])
            actual = np.argmax(test_labels[idx])
            
            # Display image with prediction
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Pred: {pred}, Act: {actual}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Display some samples
    display_sample()
    
except ImportError:
    print("Matplotlib not installed. Skipping visualization.")