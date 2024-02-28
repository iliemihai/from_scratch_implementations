import numpy as np
from read_mnist import train_images, train_labels, test_images, test_labels


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    # use cross entropy loss
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

def init_params(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

def forward_prop(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    one_hot_Y = np.eye(A2.shape[1])[Y]

    dZ2 = A2 - one_hot_Y
    dW2 = A1.T.dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T.dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Initialize parameters
input_dim = 784 # 28*28
hidden_dim = 128
output_dim = 10
W1, b1, W2, b2 = init_params(input_dim, hidden_dim, output_dim)

train_images_flat = train_images.reshape(train_images.shape[0], -1) / 255.0 # Normalize
test_images_flat = test_images.reshape(test_images.shape[0], -1) / 255.0 # Normalize


num_epochs = 10
learning_rate = 0.01

for epoch in range(num_epochs):
    # forward propagate
    Z1, A1, Z2, A2 = forward_prop(train_images_flat, W1, b1, W2, b2)
    # Compute loss
    loss = compute_loss(train_labels, A2)
    #update params
    dW1, db1, dW2, db2 = backward_prop(train_images_flat, train_labels, Z1, A1, Z2, A2, W2)
    # Update parameters
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    if epoch % 1 == 0:
        train_preds = predict(train_images_flat, W1, b1, W2, b2)
        train_acc = accuracy(train_labels, train_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}")


test_preds = predict(test_images_flat, W1, b1, W2, b2)
test_acc = accuracy(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")


