import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Understanding 1D convolution for sequence data
# Creating a sample sequence
def generate_sample_sequence(length=128, num_features=1):
    """Generate a simple sine wave sequence for demonstration"""
    x = np.linspace(0, 4 * np.pi, length)
    if num_features == 1:
        return np.sin(x).reshape(-1, 1).astype(np.float32)
    else:
        # Multiple features (channels)
        sequences = []
        for i in range(num_features):
            sequences.append(np.sin(x + i*np.pi/4))
        return np.column_stack(sequences).astype(np.float32)

# Demonstrating 1D convolution
def demo_1d_convolution():
    # Create a sample sequence (batch_size=1, channels=1, seq_length=128)
    sequence = generate_sample_sequence(length=128)
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)  # Add batch dimension
    
    # Define a 1D convolution layer (1 input channel, 16 output channels, kernel size=3)
    conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    
    # Apply convolution (need to reshape for PyTorch's Conv1d which expects [batch, channels, length])
    sequence_tensor = sequence_tensor.permute(0, 2, 1)  # Reshape to [batch, channels, length]
    output = conv1d(sequence_tensor)
    
    print(f"Input shape: {sequence_tensor.shape}")
    print(f"Output shape after Conv1D: {output.shape}")
    return output

# 2. 1D pooling for sequence data
def demo_1d_pooling(conv_output):
    # Max pooling
    max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    max_pool_output = max_pool(conv_output)
    print(f"Output shape after MaxPool1D: {max_pool_output.shape}")
    
    # Average pooling
    avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    avg_pool_output = avg_pool(conv_output)
    print(f"Output shape after AvgPool1D: {avg_pool_output.shape}")
    
    return max_pool_output, avg_pool_output

# 3. Implementing a 1D ConvNet
class Conv1DNet(nn.Module):
    def __init__(self, input_channels=1, seq_length=128, num_classes=2):
        super(Conv1DNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after all conv and pooling layers
        # After 3 pooling layers with stride 2, the length is reduced to seq_length/8
        flattened_size = 128 * (seq_length // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Apply convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 4. Combining CNNs and RNNs for processing long sequences
class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=1, seq_length=128, hidden_size=64, num_classes=2):
        super(CNNRNNModel, self).__init__()
        
        # CNN part for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate sequence length after pooling
        self.cnn_output_length = seq_length // 4
        
        # RNN part for sequential processing
        self.rnn = nn.GRU(
            input_size=64,  # Output channels from CNN
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Fully connected part for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Reshape for RNN: from [batch, channels, seq_len] to [batch, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # RNN processing
        x, _ = self.rnn(x)
        
        # Use the final output for classification (or implement attention here)
        # Either take the last time step
        out = x[:, -1, :]
        # Or use global average pooling
        # out = torch.mean(x, dim=1)
        
        # Classification layer
        out = self.fc(out)
        
        return out

# 5. Example of training a sequence model
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=128, num_features=1, num_classes=2):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Generate data
        self.sequences = []
        self.labels = []
        
        for i in range(num_samples):
            # Generate a sequence with random phase shift
            phase = np.random.uniform(0, 2 * np.pi)
            x = np.linspace(0, 4 * np.pi, seq_length)
            
            if num_features == 1:
                seq = np.sin(x + phase).reshape(-1, 1).astype(np.float32)
            else:
                seq_data = []
                for j in range(num_features):
                    seq_data.append(np.sin(x + phase + j * np.pi / 4))
                seq = np.column_stack(seq_data).astype(np.float32)
            
            self.sequences.append(seq)
            
            # For simplicity, assign class based on the phase
            label = 0 if phase < np.pi else 1
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        sequence_tensor = torch.from_numpy(sequence)
        
        # For Conv1D, we need [channels, length]
        sequence_tensor = sequence_tensor.permute(1, 0)
        
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

def train_sequence_model(model_type='cnn', epochs=5):
    # Create a dataset
    dataset = SequenceDataset(num_samples=1000, seq_length=128, num_features=1, num_classes=2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Select the model
    if model_type == 'cnn':
        model = Conv1DNet(input_channels=1, seq_length=128, num_classes=2)
    elif model_type == 'cnn_rnn':
        model = CNNRNNModel(input_channels=1, seq_length=128, hidden_size=64, num_classes=2)
    else:
        raise ValueError("Model type not supported")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return model

# Additional: Wrapping up with a function to demonstrate all parts
def run_sequence_processing_demo():
    print("1. Demonstrating 1D Convolution")
    conv_output = demo_1d_convolution()
    
    print("\n2. Demonstrating 1D Pooling")
    demo_1d_pooling(conv_output)
    
    print("\n3. Training a 1D ConvNet")
    train_sequence_model(model_type='cnn', epochs=2)
    
    print("\n4. Training a CNN-RNN Hybrid Model")
    train_sequence_model(model_type='cnn_rnn', epochs=2)
    
    print("\nDemo complete!")

# Uncomment to run the demo
# run_sequence_processing_demo()