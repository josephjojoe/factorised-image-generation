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
from tensorflow.keras.preprocessing import image

# Important image variables.
latent_dim = 32
height = 32
width = 32
channels = 3

# Generator model (factorised).
generator_input = tf.keras.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, (5, 1), padding='same')(x)
x = layers.Conv2D(256, (1, 5), padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, (5, 1), padding='same')(x)
x = layers.Conv2D(256, (1, 5), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, (5, 1), padding='same')(x)
x = layers.Conv2D(256, (1, 5), padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = tf.keras.models.Model(generator_input, x)
generator.summary()

# Discriminator model.
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = tf.keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0008,
                                                     clipvalue=1.0,
                                                     decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# GAN
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)

gan_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# Reshapes data and uses only frog images.
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20

# Training loop
start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    generated_images = generator.predict(random_latent_vectors)
    
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    misleading_targets = np.zeros((batch_size, 1))
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
        
    if step % 100 == 0:
        gan.save_weights('./cifar_training_checkpoints/gan.h5')
        
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join('./cifar_images/generated/generated_frog' + str(step) + '.png'))
        
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join('./cifar_images/real/real_frog' + str(step) + '.png'))
