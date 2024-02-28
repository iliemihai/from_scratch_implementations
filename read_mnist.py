import os
import struct
import numpy as np

"""
reads the first 8 bytes of the MNIST file, unpacks them as two unsigned integers in big-endian format, and assigns these integers to the magic and num_images 
"""
def read_mnist_images(file_path):
    with open(file_path, "rb") as file:
        magic, num_images = struct.unpack(">II", file.read(8))
        rows, cols = struct.unpack(">II", file.read(8))
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def read_mnist_labels(file_path):
    with open(file_path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)

    return labels

train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

train_images = read_mnist_images(train_images_path)
train_labels = read_mnist_labels(train_labels_path)
test_images = read_mnist_images(test_images_path)
test_labels = read_mnist_labels(test_labels_path)

print(train_images.shape)
print(train_labels.shape)
