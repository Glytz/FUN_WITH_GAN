from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os



# implementation of the original GAN from https://arxiv.org/pdf/1406.2661.pdf
# inpiration from https://towardsdatascience.com/understanding-and-optimizing-gans-going-back-to-first-principles-e5df8835ae18
# The code mainly come from : https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
# with bits here and there added and modified by me to fit my needs
class GAN():
    def __init__(self, param):
        self.image_rows = param.image_rows
        self.image_columns = param.image_columns
        self.image_channels = param.image_channels  # MNIST only have 1 channel
        self.image_shape = (self.image_rows, self.image_columns, self.image_channels)
        self.noise_dimension = param.noise_dimension
        if not param.load_model:
            optimizer = Adam(0.0002, 0.5)

            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])

            # we build the generator
            self.generator = self.build_generator()

            # we build the combined model to be able to train the generator model!
            # build the noise input
            z = Input(shape=(self.noise_dimension,))
            # generate the images
            img = self.generator(z)

            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # generate the validity of the images
            valid = self.discriminator(img)

            # train the generator to fool the discriminator!
            self.combined = Model(z, valid)
            # we compile the model with binary cossentropy as suggested
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        else:
            working_directory_path = os.getcwd()
            self.discriminator = load_model(working_directory_path + param.model_path + "/discriminator" + ".h5")
            self.generator = load_model(working_directory_path + param.model_path + "/generator" + ".h5")
            self.combined = load_model(working_directory_path + param.model_path + "/combined" + ".h5")

        r, c = 5, 5
        np.random.seed(310)
        self.image_noise = np.random.normal(0, 1, (r * c, self.noise_dimension))

    def build_generator(self):

        model = Sequential()

        model.add(Dense(1024 * 8 * 8, activation="relu", input_dim=self.noise_dimension))
        model.add(Reshape((8, 8, 1024)))
        model.add(UpSampling2D())


        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())


        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())


        model.add(Conv2D(filters=3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.noise_dimension,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(filters=3, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())

        model.add(Dense(1, activation="sigmoid"))


        model.summary()

        img = Input(shape=self.image_shape)
        validity = model(img)

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
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.noise_dimension))

            # generate a batch of new images
            generated_images = self.generator.predict(noise)

            # train the discriminator on true images and false images
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.noise_dimension))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # define the checkpoint
                directory_path = os.getcwd() + "/models"
                if not os.path.exists(directory_path):
                    os.mkdir(directory_path)
                prefix = "models/"
                suffix = ".h5"
                self.generator.save(prefix + "generator" + suffix)  # save the generator model
                self.discriminator.save(prefix + "discriminator" + suffix)  # save the discriminator model
                self.combined.save(prefix + "combined" + suffix)  # save the combined model used to train the generator
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
            parser.add_argument("--batch_size", type=int, default=128)  # size of the batch
            parser.add_argument("--epochs", type=int, default=30000)  # size of the batch
            parser.add_argument("--save_sample_interval", type=int, default=50)  # interval between saving sample imgs
            parser.add_argument("--dataset_path", type=str, default="/home/glytz/Datasets/cats_64x64/")
            parser.add_argument("--load_model", type=bool, default=False)
            parser.add_argument("--model_path", type=str, default="/models")
            parser.add_argument("--model_version", type=str, default="")
            param = parser.parse_args()
            gan = GAN(param)
            gan.train(param=param)