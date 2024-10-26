import os
import numpy as np
import struct
from PIL import Image

def read_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_labels(filename):
    with open(filename, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
    return labels

def save_images(images, labels, folder):
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(folder, f"label{label}")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image_path = os.path.join(label_dir, f"{i:05d}.jpeg")
        print(image_path)
        img = Image.fromarray(image)
        img.save(image_path)


# Paths to your MNIST data files
train_images_path = '/proj/common-datasets/MNIST/train-images-idx3-ubyte'
train_labels_path = '/proj/common-datasets/MNIST/train-labels-idx1-ubyte'
test_images_path = '/proj/common-datasets/MNIST/t10k-images-idx3-ubyte'
test_labels_path = '/proj/common-datasets/MNIST/t10k-labels-idx1-ubyte'

# Read the datasets
train_images = read_images(train_images_path)
train_labels = read_labels(train_labels_path)
test_images = read_images(test_images_path)
test_labels = read_labels(test_labels_path)

# Save the images in structured directories
save_images(train_images, train_labels, '/proj/sciml/users/x_stear/playground/train')
save_images(test_images, test_labels, '/proj/sciml/users/x_stear/playground/test')
