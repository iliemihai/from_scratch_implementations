import numpy as np
from read_mnist import train_images, train_labels, test_images, test_labels
import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_backward(dA, Z):
    """Backward propagation for ReLU."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(x):
    """Softmax activation function for multi-class classification."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    """Cross-entropy loss function."""
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

def init_cnn_params(filter_size, num_filters, output_dim):
    """Initializes CNN parameters."""
    W_conv = np.random.randn(filter_size, filter_size, 1, num_filters) / np.sqrt(filter_size * filter_size)
    b_conv = np.zeros((1, num_filters))
    num_flattened = (28 // 2) * (28 // 2) * num_filters
    W_fc = np.random.randn(num_flattened, output_dim) / np.sqrt(num_flattened)
    b_fc = np.zeros((1, output_dim))
    return W_conv, b_conv, W_fc, b_fc

def conv2d(X, W, b, stride=1, padding=0):
    """Convolution operation."""
    n_h, n_w = X.shape
    f_h, f_w, _, n_c = W.shape
    out_h = (n_h - f_h + 2 * padding) // stride + 1
    out_w = (n_w - f_w + 2 * padding) // stride + 1
    Z = np.zeros((out_h, out_w, n_c))
    X_padded = np.pad(X, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
    for h in range(out_h):
        for w in range(out_w):
            for c in range(n_c):
                vert_start = h * stride
                vert_end = vert_start + f_h
                horiz_start = w * stride
                horiz_end = horiz_start + f_w
                X_slice = X_padded[vert_start:vert_end, horiz_start:horiz_end]
                Z[h, w, c] = np.sum(X_slice * W[:, :, :, c]) + b[0, c]
    return Z

def max_pool(X, f=2, stride=2):
    """Max pooling operation."""
    n_h, n_w, n_c = X.shape
    out_h = int(1 + (n_h - f) / stride)
    out_w = int(1 + (n_w - f) / stride)
    A = np.zeros((out_h, out_w, n_c))
    for h in range(out_h):
        for w in range(out_w):
            for c in range(n_c):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
                A[h, w, c] = np.max(X[vert_start:vert_end, horiz_start:horiz_end, c])
    return A

def max_pool_backward(dmax_pool, X):
    """Backward propagation through max pooling."""
    n_h, n_w, n_c = dmax_pool.shape
    dA = np.zeros_like(X)
    for h in range(n_h):
        for w in range(n_w):
            for c in range(n_c):
                c = np.where(X[h * stride:h * stride + f, w * stride:w * stride + f, c] == dmax_pool[h, w, c])[0][0]
                dA[h * stride + c, w * stride + c, c] = dmax_pool[h, w, c]
    return dA

def conv2d_backward(dZ, X, W):
    """Backward propagation through convolution."""
    n_h, n_w, n_c = X.shape
    f_h, f_w, _, n_c_out = W.shape
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(W[:, :, 0, :])
    X_padded = np.pad(X, ((f_h - 1, f_h - 1), (f_w - 1, f_w - 1)), 'constant', constant_values=0)
    for h in range(n_h):
        for w in range(n_w):
            for c in range(n_c_out):
                for kh in range(f_h):
                    for kw in range(f_w):
                        dX[h, w, :] += np.sum(dZ[h, w, :] * W[kh, kw, :, c] * X_padded[h + kh, w + kw, :])
                db[0, c] = np.sum(dZ[:, :, c])
                for i in range(n_c):
                    dW[kh, kw, i, c] = np.sum(X_padded[h + kh:h + kh + 1, w + kw:w + kw + 1, i] * dZ[:, :, c])
    return dX, dW, db

def cnn_forward_prop(X, W_conv, b_conv, W_fc, b_fc):
    """Forward propagation through the CNN."""
    Z_conv = conv2d(X, W_conv, b_conv, stride=1, padding=0)
    A_conv = relu(Z_conv)
    A_pool = max_pool(A_conv, f=2, stride=2)
    A_flat = A_pool.reshape(A_pool.shape[0]*A_pool.shape[1]*A_pool.shape[2], -1).T
    Z_fc = np.dot(A_flat, W_fc) + b_fc
    A_fc = softmax(Z_fc)
    cache = (Z_conv, A_conv, A_pool, A_flat, Z_fc, A_fc)
    return A_fc, cache

def cnn_backward_prop(y_true, cache, W_conv, b_conv, W_fc, b_fc, learning_rate):
    """Backward propagation through the CNN."""
    (Z_conv, A_conv, A_pool, A_flat, Z_fc, A_fc) = cache
    m = y_true.shape[0]
    y_true = np.eye(W_fc.shape[1])[y_true]

    dZ_fc = A_fc - y_true
    dW_fc = np.dot(A_flat.T, dZ_fc) / m
    db_fc = np.sum(dZ_fc, axis=0, keepdims=True) / m
    dA_flat = np.dot(dZ_fc, W_fc.T)
    dA_pool = dA_flat.reshape(A_pool.shape)

    # Backpropagate through pooling and convolution layers
    dmax_pool = max_pool_backward(dA_pool, A_conv)
    dX_conv, dW_conv, db_conv = conv2d_backward(dmax_pool, X, W_conv)

    # Update parameters
    W_conv -= learning_rate * dW_conv
    b_conv -= learning_rate * db_conv
    W_fc -= learning_rate * dW_fc
    b_fc -= learning_rate * db_fc

    return W_conv, b_conv, W_fc, b_fc

def train(X_train, y_train, epochs, learning_rate):
    """Trains the CNN model."""
    filter_size = 5
    num_filters = 8
    output_dim = 10
    W_conv, b_conv, W_fc, b_fc = init_cnn_params(filter_size, num_filters, output_dim)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for X, y_true in zip(X_train, y_train):
            X = X.reshape(28, 28)  # Assuming grayscale images of size 28x28
            A_fc, cache = cnn_forward_prop(X, W_conv, b_conv, W_fc, b_fc)
            W_conv, b_conv, W_fc, b_fc = cnn_backward_prop(y_true, cache, W_conv, b_conv, W_fc, b_fc, learning_rate)

        # Optionally, evaluate the model on a validation set here

    return W_conv, b_conv, W_fc, b_fc

# Example usage
# X_train, y_train = load_data()  # Assume this function loads your training data
trained_params = train(train_images, train_labels, epochs=10, learning_rate=0.01)

