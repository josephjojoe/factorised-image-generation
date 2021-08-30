import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import tensorflow_docs.vis.embed as embed

# Loads in mnist data.
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshaping and normalisation.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalises the image data to [-1, 1] rabe

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batches and shuffles the data.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
