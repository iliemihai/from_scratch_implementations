import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from read_mnist import train_images, train_labels, test_images, test_labels
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        # Define convolutional layers and pooling layers
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        # Calculate the number of features after the convolutional layers
        self.num_features = self._get_num_features()

        # Define fully-connected layers
        self.fc1 = nn.Linear(self.num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through the convolutional layers and pooling
        #print("SHAPE", x, x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the features for fully-connected layers
        x = x.view(-1, self.num_features)

        # Pass through the fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_num_features(self):
        """
        Calculates the number of features after the convolutional layers and pooling.
        """
        with torch.no_grad():
            # Create a dummy input
            dummy_input = torch.randn(1, 1, 28, 28)  # Assuming input size 28x28 (modify if different)
            # Pass the dummy input through the layers
            x = self.conv1(dummy_input)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            # Return the number of elements in the flattened output
            return x.view(-1).size(0)

class MNISTDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        # Convert grayscale image to a single-channel tensor
        if len(image.shape) == 3:  # Check if image has 3 dimensions (RGB)
            image = image[:, :, 0]  # Select the first channel (assuming grayscale is first)
        image = image.reshape(1, image.shape[0], image.shape[1])  # Add a channel dimension
        image = image.astype(np.float32)  # Convert to float32 for PyTorch compatibility

        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, train_loader, optimizer, criterion, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass, calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}]\tLoss: {loss.item():.4f}")


def evaluate(model, eval_loader, criterion):
    """
    Evaluates the model on the given data loader and returns accuracy and loss.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        eval_loader (torch.utils.data.DataLoader): The data loader for the evaluation data.
        criterion (torch.nn.Module): The loss function used for evaluation.

    Returns:
        tuple: A tuple containing the average accuracy and loss over the entire evaluation dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize variables to track metrics
    total_loss = 0
    correct = 0
    total = 0

    # Set the model to evaluation mode (optional for specific layers)
    model.eval()

    # No need to iterate through multiple epochs for evaluation
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update total loss
            total_loss += loss.item()

    # Calculate average accuracy and loss
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(eval_loader)

    return accuracy, avg_loss

# Load MNIST data
# Create datasets and data loaders
train_dataset = MNISTDataset(train_images, train_labels)
test_dataset = MNISTDataset(test_images, test_labels)

# Create datasets and data loaders
train_dataset = MNISTDataset(train_images.astype(np.float32), train_labels)
test_dataset = MNISTDataset(test_images.astype(np.float32), test_labels)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Example usage
input_channels = 1  # Grayscale images
num_classes = 10

model = CNN(input_channels, num_classes)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train(model, train_loader, optimizer, criterion, epochs=10)

# Example usage:
accuracy, loss = evaluate(model, eval_loader, criterion)
print(f"Accuracy: {accuracy:.2f}%, Average Loss: {loss:.4f}")
