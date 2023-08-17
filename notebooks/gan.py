import os

from skimage.transform import rescale, resize, downscale_local_mean
from IPython import display
import tensorflow as tf

import astroCAST.analysis
from astroCAST import denoising
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import logging
from pathlib import Path

print(f"tf.version: {tf.__version__}")

class ImgGenerator():
    
    def __init__(self, files, loc="data", BATCH_SIZE=32, IMG_SIZE=(28, 28), max_per_file=1, scale=1):
        
        xy = (int(IMG_SIZE[0]*scale), int(IMG_SIZE[1]*scale))
        
        self.sfg = denoising.SubFrameGenerator(files, loc=loc,
            batch_size = BATCH_SIZE, input_size=xy, max_per_file=max_per_file*BATCH_SIZE,
            pre_post_frame=(1, 0), gap_frames=0, allowed_rotation=(1, 2, 3), allowed_flip=(0, 1), padding=None,
            random_offset=True, normalize=None, logging_level=logging.INFO)
        
        self.scale = scale
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def __getitem__(self, index):
        data = self.sfg[index][1]
        
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        data = 2 * (data - min_) / (max_ - min_) - 1
        
        if self.scale != 1:
            
            new_data = np.zeros((data.shape[0], self.IMG_SIZE[0], self.IMG_SIZE[1]), dtype=data.dtype)
            for z in range(data.shape[0]):
            
                new_data[z, :, :] = resize(data[z, :, :], self.IMG_SIZE, anti_aliasing=True)
        
            data = new_data
        
        return data
    
    def __len__(self):
        return len(self.sfg)
    
    def __iter__(self):
        self.current = 0
        return self

    def __next__(self): # Python 2: def next(self)
        
        if self.current < len(self) - 1:
            self.current += 1
            return self[self.current]
        
        self.sfg.on_epoch_end()
        raise StopIteration

    def next(self):
        for t in self:
            return t

    def sample(self, index=0, max_n=16):

        i0 = self[index][:max_n]

        logging.info(f"img shape: {i0.shape}")
        logging.info(f"values: {np.min(i0):.2f} - {np.max(i0):.2f}")

        N = i0.shape[0]
        n = int(np.sqrt(N)) + int(N % int(np.sqrt(N)) > 0)

        fig, axx = plt.subplots(n, n)
        axx = list(axx.flatten())

        for i in range(N):
            axx[i].imshow(i0[i, :, :])
            axx[i].axis("off")

        for i in range(i, n**2):
            axx[i].remove()

        plt.tight_layout()

        return i0

class TemplateGenerator():

    def __init__(self, files, loc="data", BATCH_SIZE=32, IMG_SIZE=(28, 28), max_per_file=1, scale=1):

        xy = (int(IMG_SIZE[0]*scale), int(IMG_SIZE[1]*scale))

        self.tg = denoising.SubFrameGenerator(files, loc=loc,
            batch_size = BATCH_SIZE, input_size=xy, max_per_file=max_per_file*BATCH_SIZE,
            pre_post_frame=(1, 0), gap_frames=0, allowed_rotation=(1, 2, 3), allowed_flip=(0, 1), padding=None,
            random_offset=True, normalize=None, logging_level=logging.INFO)

        self.scale = scale
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def __getitem__(self, index):
        data = self.tg[index][1]

        mask = data > 1
        background = np.random.normal(size=data.shape)
        signal = np.random.normal(loc=1, scale=0.5, size=data.shape) * mask

        data = background + signal

        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        data = 2 * (data - min_) / (max_ - min_) - 1

        if self.scale != 1:

            new_data = np.zeros((data.shape[0], self.IMG_SIZE[0], self.IMG_SIZE[1]), dtype=data.dtype)
            for z in range(data.shape[0]):

                new_data[z, :, :] = resize(data[z, :, :], self.IMG_SIZE, anti_aliasing=True)

            data = new_data

        return data

    def __len__(self):
        return len(self.tg)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self): # Python 2: def next(self)

        if self.current < len(self) - 1:
            self.current += 1
            return self[self.current]

        self.tg.on_epoch_end()
        raise StopIteration

    def next(self):
        for t in self:
            return t

    def sample(self, index=0, max_n=16):

        i0 = self[index][:max_n]

        logging.info(f"img shape: {i0.shape}")
        logging.info(f"values: {np.min(i0):.2f} - {np.max(i0):.2f}")

        N = i0.shape[0]
        n = int(np.sqrt(N)) + int(N % int(np.sqrt(N)) > 0)

        fig, axx = plt.subplots(n, n)
        axx = list(axx.flatten())

        for i in range(N):
            axx[i].imshow(i0[i, :, :])
            axx[i].axis("off")

        for i in range(i, n**2):
            axx[i].remove()

        plt.tight_layout()

        return i0

class GAN:

    def __init__(self, img_generator, tmp_generator, checkpoint_dir='./training_checkpoints', checkpoint_frequency=10,
                 modifier_param={},
                 optimizer_lambda=1e-4, logging_level=logging.INFO):

        logging.basicConfig(level=logging_level)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.img_generator = img_generator
        self.tmp_generator = tmp_generator

        self.batch_size = self.img_generator.BATCH_SIZE
        self.input_noise_shape = tuple((*self.img_generator.IMG_SIZE, 1))
        logging.info(f"batch size: {self.batch_size}, noise shape: {self.input_noise_shape}")

        self.generator = self.make_modifier_model_up_down(**modifier_param)
        self.discriminator = self.make_discriminator_model()

        # Optimizers
        if isinstance(optimizer_lambda, float):
            optimizer_lambda_1 = optimizer_lambda_2 = optimizer_lambda
        elif isinstance(optimizer_lambda, tuple):
            optimizer_lambda_1, optimizer_lambda_2 = optimizer_lambda
        else:
            raise ValueError(f"optimizer_lambda must be float or tuple instead of: {optimizer_lambda}")

        self.generator_optimizer = tf.keras.optimizers.Adam(optimizer_lambda_1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(optimizer_lambda_2)

        # Checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        self.checkpoint_frequency = checkpoint_frequency

    def make_modifier_model(self, num_layers=2, kernel_size=5, kernel_exponent=7, line_length=100):

        layer_container = []
        for i in range(num_layers):

            if i == 0:
                layer_container += [layers.Conv2DTranspose(int(2**kernel_exponent), (kernel_size, kernel_size),
                                                      strides=(1, 1), padding="same", use_bias=False,
                                                      input_shape=self.input_noise_shape)]
            else:
                layer_container += [layers.Conv2DTranspose(int(2**kernel_exponent), (kernel_size, kernel_size),
                                                      strides=(1, 1), padding="same", use_bias=False)]

            layer_container += [layers.BatchNormalization()]
            layer_container += [layers.LeakyReLU()]

            kernel_exponent -= 1

            if kernel_exponent < 1:
                logging.warning(f"kernel_exponent too small ({kernel_exponent}). Increase exponent or decrease num_layers")
                break

        layer_container += [layers.Conv2DTranspose(1, (kernel_size, kernel_size), strides=(1, 1),
                                                  padding='same', use_bias=False, activation="tanh")]

        model = tf.keras.Sequential(layer_container)
        logging.info(model.summary(line_length=line_length))

        return model

    def make_modifier_model_up_down(self, kernel_exponents=(6, 5), kernel_size=5, line_length=100):

        layer_container = []
        for i, k in enumerate(kernel_exponents):

            if i == 0:
                layer_container += [layers.Conv2DTranspose(int(2**k), (kernel_size, kernel_size),
                                                                      strides=(2, 2), padding="same", use_bias=False,
                                                                      input_shape=self.input_noise_shape)]
            else:
                layer_container += [layers.Conv2DTranspose(int(2**k), (kernel_size, kernel_size),
                                                                      strides=(2, 2), padding="same", use_bias=False)]

            layer_container += [layers.BatchNormalization()]
            layer_container += [layers.LeakyReLU()]

        for i, k in enumerate(kernel_exponents[::-1]):

            if i == 0:
                layer_container += [layers.Conv2D(int(2**k), (kernel_size, kernel_size),
                                                                      strides=(2, 2), padding="same", use_bias=False,
                                                                      input_shape=self.input_noise_shape)]
            else:
                layer_container += [layers.Conv2D(int(2**k), (kernel_size, kernel_size),
                                                                      strides=(2, 2), padding="same", use_bias=False)]

            layer_container += [layers.BatchNormalization()]
            layer_container += [layers.LeakyReLU()]

        layer_container += [layers.Conv2DTranspose(1, (kernel_size, kernel_size), strides=(1, 1),
                                                          padding='same', use_bias=False, activation="tanh")]

        model = tf.keras.Sequential(layer_container)
        logging.info(model.summary(line_length=line_length))

        return model

    def make_discriminator_model(self, line_length=100):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=self.input_noise_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        logging.info(model.summary(line_length=line_length))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def load_last_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def generate_seed(self, num_seeds=1):
        return tf.random.normal([num_seeds, *self.input_noise_shape])

    def train(self, epochs, num_examples_to_generate=12, load_pretrained=False,
              plot_param={}):

        if load_pretrained:
            self.load_last_checkpoint()

        seed = self.tmp_generator.next()[:num_examples_to_generate, :, :]
        truth = self.img_generator.next()[:num_examples_to_generate, :, :]

        for epoch in range(epochs):
            start = time.time()

            for image_batch in self.img_generator:
                self.train_step(image_batch)

            self.generate_and_save_images(epoch=epoch+1, test_input=seed, reference=truth, **plot_param)

            # Save the model every 15 epochs
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            logging.info(f'Time for epoch {epoch+1} is {time.time()-start:.2f}s')

        # Generate after the final epoch
        self.generate_and_save_images(epoch=epochs, test_input=seed, **plot_param)

    @tf.function
    def train_step(self, images):
        noise = self.tmp_generator.next()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generate_and_save_images(self, epoch, test_input, reference, max_n=5, figsize=(6,6), figsize_multiply=4, save_dir=None):

        predictions = self.generator(test_input, training=False)

        class_p = self.discriminator(predictions, training=False)
        class_r = self.discriminator(reference, training=False)

        predictions = np.concatenate([np.squeeze(predictions), np.squeeze(reference)], axis=0)
        class_ = np.concatenate([class_p, class_r], axis=0)

        display.clear_output(wait=True)

        N=len(class_)
        plotting = astroCAST.analysis.Plotting(None)
        fig, axx = plotting._get_square_grid(N=N, figsize=figsize, figsize_multiply=figsize_multiply, max_n=max_n)
        fig.suptitle(f"Epoch {epoch}")

        for i in range(N):
            axx[i].imshow(predictions[i, :, :],
                          vmin=-1, vmax=1
                          # cmap="Greys"
                          )
            axx[i].axis('off')
            axx[i].text(0.05, 0.95, f"{float(class_[i]):.2f}", transform=axx[i].transAxes, fontsize=7,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                                                       facecolor="green" if class_[i]>0 else "red",
                                                       alpha=0.5))

        plt.tight_layout()

        if save_dir is not None:

            save_dir = Path(save_dir)
            if not save_dir.is_dir():
                save_dir.mkdir()

            plt.savefig(save_dir.joinpath('image_at_epoch_{:04d}.png'.format(epoch)))

        plt.show()

    @staticmethod
    def display_image(epoch, save_dir):
        return PIL.Image.open(Path(save_dir).joinpath('image_at_epoch_{:04d}.png'.format(epoch)))

    def sample(self, plot_param={}):

        noise = self.generate_seed()
        gen_img = self.generator(noise, training=False)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(gen_img[0, :, :, 0], cmap='gray', **plot_param)

