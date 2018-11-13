# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# and to https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
# which I've used as a reference for this implementation
import argparse
import os
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import load_model
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
    def __init__(self, param):
        self.img_rows = param.image_rows
        self.img_cols = param.image_columns
        self.channels = param.image_channels  # MNIST only have 1 channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = param.noise_dimension
        r, c = 5, 5
        np.random.seed(1693214)
        self.image_noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        if not param.load_model:
            # Following parameter and optimizer set as recommended in paper
            self.n_critic = 5
            optimizer = RMSprop(lr=0.00005)

            # Build the generator and critic
            self.generator = self.build_generator()
            self.critic = self.build_critic()

            #-------------------------------
            # Construct Computational Graph
            #       for the Critic
            #-------------------------------

            # Freeze generator's layers while training critic
            self.generator.trainable = False

            # Image input (real sample)
            real_img = Input(shape=self.img_shape)

            # Noise input
            z_disc = Input(shape=(self.latent_dim,))
            # Generate image based of noise (fake sample)
            fake_img = self.generator(z_disc)

            # Discriminator determines validity of the real and fake images
            fake = self.critic(fake_img)
            valid = self.critic(real_img)

            # Construct weighted average between real and fake images
            interpolated_img = RandomWeightedAverage()([real_img, fake_img])
            # Determine validity of weighted sample
            validity_interpolated = self.critic(interpolated_img)

            # Use Python partial to provide loss function with additional
            # 'averaged_samples' argument
            partial_gp_loss = partial(self.gradient_penalty_loss,
                              averaged_samples=interpolated_img)
            partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

            self.critic_model = Model(inputs=[real_img, z_disc],
                                outputs=[valid, fake, validity_interpolated])
            self.critic_model.compile(loss=[self.wasserstein_loss,
                                                  self.wasserstein_loss,
                                                  partial_gp_loss],
                                            optimizer=optimizer,
                                            loss_weights=[1, 1, 10])
            #-------------------------------
            # Construct Computational Graph
            #         for Generator
            #-------------------------------

            # For the generator we freeze the critic's layers
            self.critic.trainable = False
            self.generator.trainable = True

            # Sampled noise for input to generator
            z_gen = Input(shape=(self.latent_dim,))
            # Generate images based of noise
            img = self.generator(z_gen)
            # Discriminator determines validity
            valid = self.critic(img)
            # Defines generator model
            self.generator_model = Model(z_gen, valid)
            self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        else:
            working_directory_path = os.getcwd()
            self.critic = load_model(working_directory_path + param.model_path + "/critic" + ".h5")
            self.generator = load_model(working_directory_path + param.model_path + "/generator" + ".h5")

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        # https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7 why we use leaky relu
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add((LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.1))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.1))
        model.add(Dense(1, kernel_initializer='he_normal'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

        return Model(img, validity)

    def train(self, param):
        epochs = param.epochs
        batch_size = param.batch_size
        sample_interval = param.save_sample_interval
        # first thing first... we need to get our dataset!
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        img_datagen = ImageDataGenerator()
        img_generator = img_datagen.flow_from_directory(directory=param.dataset_path,
                                                        target_size=(param.image_rows, param.image_columns),
                                                        color_mode='rgb', classes=None,
                                                        class_mode="binary", batch_size=10000000, shuffle=True)
        # test = img_generator.next()
        # X_train = np.zeros((img_generator.n, param.image_rows, param.image_columns, param.image_channels))
        # Rescale -1 to 1
        X_train = img_generator.next()[0]
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # define the checkpoint
                directory_path = os.getcwd() + "/models"
                if not os.path.exists(directory_path):
                    os.mkdir(directory_path)
                prefix = "models/"
                suffix = ".h5"
                self.generator.save(prefix + "generator" + suffix)  # save the generator model
                self.critic.save(prefix + "critic" + suffix)  # save the discriminator model
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = self.image_noise
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        directory_path = os.getcwd() + "/images"
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    # default mninst parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_rows", type=int, default=64)  # rows of the image
    parser.add_argument("--image_columns", type=int, default=64)  # columns of the images
    parser.add_argument("--image_channels", type=int,
                        default=3)  # number of channels, 1 for BW. 3 for color images
    parser.add_argument("--noise_dimension", type=int, default=100)  # noise dimension for the generator
    parser.add_argument("--batch_size", type=int, default=32)  # size of the batch
    parser.add_argument("--epochs", type=int, default=30000)  # size of the batch
    parser.add_argument("--save_sample_interval", type=int, default=50)  # interval between saving sample imgs
    parser.add_argument("--dataset_path", type=str, default="/home/glytz/Datasets/cats_64x64/")
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="/models")
    parser.add_argument("--model_version", type=str, default="")
    param = parser.parse_args()
    wgan = WGANGP(param)
    wgan.train(param)