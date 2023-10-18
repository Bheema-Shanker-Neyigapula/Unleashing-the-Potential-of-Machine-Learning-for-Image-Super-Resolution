import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load the CIFAR-10 dataset
(train_images, _), (test_images, _) = cifar10.load_data()

# Preprocess the images
def preprocess_images(images):
    # Convert pixel values to the range of [-1, 1]
    images = (images.astype(np.float32) - 127.5) / 127.5
    return images

# Preprocess the training and testing images
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# Split the dataset into high-resolution and low-resolution images
low_res_images = tf.image.resize(train_images, [32, 32], method='bicubic').numpy()

# Print the shapes of the low-resolution and high-resolution images
print("Low-resolution images shape:", low_res_images.shape)
print("High-resolution images shape:", train_images.shape)
