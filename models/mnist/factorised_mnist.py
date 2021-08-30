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

# Factorised generator model.
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (3, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(128, (1, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(128, (3, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(128, (1, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 1), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(64, (1, 3), strides=(2, 2), padding='same', use_bias=False))
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 1), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Conv2DTranspose(1, (1, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model
