import logging
import os
import glob
import random
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import keras
from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import mixed_precision

import numpy as np
import h5py as h5
import tifffile as tiff
from skimage.transform import resize
import pandas as pd

# TODO combine generators into one class with combined interface

class FullFrameGenerator(keras.utils.Sequence):

    """ Takes a single .h5 or tiff file and generates preprocessed training batches.

    """

    def __init__(self, pre_post_frame, batch_size, steps_per_epoch,
                 file_path, loc=None,
                 max_frame_summary_stats=1000, # TODO superseded
                 start_frame=0, end_frame=-1,
                 gap_frames=0, total_samples=-1, randomize=True):

        """

        :param pre_post_frame:
        :param batch_size:
        :param steps_per_epoch:
        :param start_frame:
        :param end_frame: compatlible with negative frames. -1 is the last
        :param gap_frames:
        :param total_samples:
        :param randomize:
        """

        if type(pre_post_frame) == int:
            self.pre_frame, self.post_frame = pre_post_frame, pre_post_frame
        else:
            self.pre_frame, self.post_frame = pre_post_frame

        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.pre_post_omission = gap_frames
        self.total_samples = total_samples
        self.randomize = randomize

        self.file_path = file_path
        self.loc = loc

        # We initialize the epoch counter
        self.epoch_index = 0

        # read file dimensions
        if file_path.endswith(".h5"):

            assert loc is not None, "When using a .h5 file the 'loc' parameter needs to be provided"

            with h5.File(file_path, "r") as file:

                data = file[loc]
                self.total_frame_per_movie = int(data.shape[0])

                self._update_end_frame(self.total_frame_per_movie)
                self._calculate_list_samples(self.total_frame_per_movie)

                average_nb_samples = np.min([int(self.total_frame_per_movie), max_frame_summary_stats])
                local_data = data[0:average_nb_samples, :, :].flatten()
                local_data = local_data.astype("float32")

                self.local_mean = np.mean(local_data)
                self.local_std = np.std(local_data)

        if file_path.endswith(".tiff") or file_path.endswith(".tif"):

            tif = tiff.TiffFile(file_path)
            self.total_frame_per_movie = len(tif.pages)
            tif.close()

            self._update_end_frame(self.total_frame_per_movie)
            self._calculate_list_samples(self.total_frame_per_movie)

            average_nb_samples = np.min([int(self.total_frame_per_movie), max_frame_summary_stats])
            local_data = tiff.imread(file_path, key=range(self.start_frame, average_nb_samples)).flatten()
            local_data = local_data.astype("float32")

            self.local_mean = np.mean(local_data)
            self.local_std = np.std(local_data)

    def _get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape[1:]

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape[1:]

    # TODO change 512, 512 >> dynamic
    def __getitem__(self, index):
        shuffle_indexes = self.generate_batch_indexes(index)

        input_full = np.zeros(
            [self.batch_size, 512, 512, self.pre_frame + self.post_frame],
            dtype="float32",
        )

        output_full = np.zeros([self.batch_size, 512, 512, 1], dtype="float32")

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    # TODO change 512, 512 >> dynamic
    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        input_full = np.zeros([1, 512, 512, self.pre_frame + self.post_frame])
        output_full = np.zeros([1, 512, 512, 1])

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index !=
                                      index_frame - index_padding]
            input_index = input_index[input_index !=
                                      index_frame + index_padding]

        if self.file_path.endswith(".h5"):

            with h5.File(self.file_path, "r") as movie_obj:

                data_img_input = movie_obj[self.loc][input_index, :, :]
                data_img_input = np.swapaxes(data_img_input, 1, 2)
                data_img_input = np.swapaxes(data_img_input, 0, 2)

                data_img_output = movie_obj[self.loc][index_frame, :, :]

        elif self.file_path.endswith(".tiff") or self.file_path.endswith(".tif"):

            data_img_input = tiff.imread(self.file_path, key=input_index)
            # TODO this might not be necessary for TIFFs
            data_img_input = np.swapaxes(data_img_input, 1, 2)
            data_img_input = np.swapaxes(data_img_input, 0, 2)

            data_img_output = tiff.imread(self.file_path, key=index_frame)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float") - self.local_mean
        ) / self.local_std

        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full

    def __get_norm_parameters__(self, idx):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample

        Parameters:
        idx index of the sample

        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std

    def _update_end_frame(self, total_frame_per_movie):
        """Update end_frame based on the total number of frames available.
        This allows for truncating the end of the movie when end_frame is
        negative."""

        # This is to handle selecting the end of the movie
        if self.end_frame < 0:
            self.end_frame = total_frame_per_movie+self.end_frame
        elif total_frame_per_movie <= self.end_frame:
            self.end_frame = total_frame_per_movie-1

    def _calculate_list_samples(self, total_frame_per_movie):

        # We first cut if start and end frames are too close to the edges.
        self.start_sample = np.max([self.pre_frame
                                    + self.pre_post_omission,
                                    self.start_frame])
        self.end_sample = np.min([self.end_frame, total_frame_per_movie - 1 -
                                  self.post_frame - self.pre_post_omission])

        if (self.end_sample - self.start_sample+1) < self.batch_size:
            raise Exception("Not enough frames to construct one " +
                            str(self.batch_size) + " frame(s) batch between " +
                            str(self.start_sample) +
                            " and "+str(self.end_sample) +
                            " frame number.")

        # +1 to make sure end_samples is included
        self.list_samples = np.arange(self.start_sample, self.end_sample+1)

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def on_epoch_end(self):
        """We only increase index if steps_per_epoch is set to positive value.
        -1 will force the generator to not iterate at the end of each epoch."""
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def __len__(self):
        "Denotes the total number of batches"
        return int(len(self.list_samples) / self.batch_size)

    def generate_batch_indexes(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

        return shuffle_indexes

class SubFrameGenerator(tf.keras.utils.Sequence):

    """ Takes a single or multiple paths to a .h5 file containing video data in (Z, X, Y) format and generates
        batches of preprocessed data of 'input_size'.
    """

    def __init__(self, paths,
                 batch_size,
                 input_size=(100, 100),
                 pre_post_frame=5, gap_frames=0, z_steps=0.1, z_select=None,
                 allowed_rotation=[0], allowed_flip=[-1],
                 random_offset=False, add_noise=False, drop_frame_probability=None,
                 extend_z=-1, max_per_file=None,
                 overlap=None,
                 shuffle=True, normalize=None,
                 loc="data/",
                 output_size=None, cache_results=False):

        if type(paths) != list:
            paths = [paths]
        self.paths = paths
        self.loc = loc

        logging.debug(f"data files: {self.paths}")
        logging.debug(f"data loc: {self.loc}")

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

        if type(pre_post_frame) == int:
            pre_post_frame = (pre_post_frame, pre_post_frame)
        self.signal_frames = pre_post_frame

        if type(gap_frames) == int:
            gap_frames = (gap_frames, gap_frames)
        self.gap_frames = gap_frames

        self.z_steps = z_steps
        self.z_select = z_select
        self.max_per_file = max_per_file

        self.allowed_rotation = allowed_rotation
        self.allowed_flip = allowed_flip

        self.overlap = overlap  # float
        self.random_offset = random_offset
        self.extend_z = extend_z
        self.add_noise = add_noise
        self.drop_frame_probability = drop_frame_probability

        assert normalize in [None, "normalize", "center", "standardize"], "normalize argument needs be one of: [None, 'normalize', 'center', 'standardize']"
        self.normalize = normalize

        self.shuffle = shuffle
        self.n = None

        # get items
        self.items = self.generate_items()

        # cache
        self.cache_results = cache_results
        self.cache = {}

    def generate_items(self):

        # define size of each predictive field of view (X, Y)
        iw, ih = self.input_size

        if self.overlap is not None:
            dw = int(iw * self.overlap)
            dh = int(ih * self.overlap)
        else:
            dw = iw
            dh = ih

        # define prediction length (Z)
        if type(self.signal_frames) == int:
            signal_frames = (self.signal_frames, self.signal_frames)
        else:
            signal_frames = self.signal_frames

        if type(self.gap_frames) == int:
            gap_frames = (self.gap_frames, self.gap_frames)
        else:
            gap_frames = self.gap_frames

        stack_len = signal_frames[0] + gap_frames[0] + 1 + gap_frames[1] + signal_frames[1]
        z_steps = max(1, int(self.z_steps * stack_len))

        # randomize input
        if self.random_offset:
            x_start = np.random.randint(0, dw)
            y_start = np.random.randint(0, dh)
            z_start = np.random.randint(0, stack_len)
        else:
            x_start, y_start, z_start = 0, 0, 0

        allowed_rotation = self.allowed_rotation if self.allowed_rotation is not None else [None]
        allowed_flip = self.allowed_flip if self.allowed_flip is not None else [None]

        # iterate over possible items
        idx = 0
        container = []
        for file in self.paths:

            if type(file) == str:
                file = Path(file)

            assert file.is_file(), "can't find: {}".format(file)

            if file.suffix == ".h5":
                with h5.File(file.as_posix(), "r") as f:
                    data = f[self.loc]
                    Z, X, Y = data.shape

            elif file.suffix in (".tiff", ".tif"):

                tif = tiff.TiffFile(file.as_posix())
                Z = len(tif.pages)
                X, Y = tif.pages[0].shape
                tif.close()
            else:
                raise NotImplementedError(f"filetype is recognized - please provide .h5, .tif or .tiff instead of: {file}")

            if self.z_select is not None:
                Z0 = max(0, self.z_select[0])
                Z1 = min(Z, self.z_select[1])
            else:
                Z0, Z1 = 0, Z

            zRange =list(range(Z0 + z_start, Z1 - stack_len - z_start, z_steps))
            xRange = list(range(x_start, X - iw - x_start, dw))
            yRange = list(range(y_start, Y - ih - y_start, dh))

            if self.shuffle:
                random.shuffle(zRange)
                random.shuffle(xRange)
                random.shuffle(yRange)

            logging.debug(f"file ZXY ranges:\nzrange: {zRange}\nxrange: {xRange}\nyrange: {yRange}")

            per_file_counter = 0
            for z0 in zRange:
                z1 = z0 + stack_len

                for x0 in xRange:
                    x1 = x0 + iw

                    for y0 in yRange:
                        y1 = y0 + ih

                        if (self.max_per_file is not None) and (per_file_counter > self.max_per_file):
                            continue

                        rot = random.choice(allowed_rotation)
                        flip = random.choice(allowed_flip)

                        if (self.drop_frame_probability is not None) and (np.random.random() <= self.drop_frame_probability):
                            drop_frame = np.random.randint(0, np.sum(signal_frames))
                        else:
                            drop_frame = -1

                        container.append(
                            {"idx": idx, "path": file, "z0": z0, "z1": z1, "x0": x0, "x1": x1, "y0": y0,
                             "y1": y1, "rot": rot, "flip": flip,
                             "noise": self.add_noise, "drop_frame": drop_frame, "extend_z": self.extend_z})
                        idx += 1
                        per_file_counter += 1

        items = pd.DataFrame(container)
        logging.debug(f"items: {items}")

        if self.shuffle:
            items = items.sample(frac=1).reset_index(drop=True)

        items["batch"] = items.idx / self.batch_size
        items.batch = items.batch.astype(int)

        self.n = len(items)

        return items

    def on_epoch_end(self):

        # called after each epoch
        if self.shuffle:
            if self.random_offset:
                self.items = self.generate_items()
            else:
                self.items = self.items.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):

        if index in self.cache.keys():
            return self.cache[index]

        X = []
        y = []
        for _, row in self.items[self.items.batch == index].iterrows():

            if row.path.suffix == ".h5":
                with h5.File(row.path.as_posix(), "r") as f:
                    data = f[self.loc][row.z0:row.z1, row.x0:row.x1, row.y0:row.y1]

            elif row.path.suffix in (".tif", ".tiff"):
                data = tiff.imread(row.path.as_posix(), key=range(row.z0, row.z1))
                data = data[:, row.x0:row.x1, row.y0:row.y1]

            if row.rot != 0:

                data = np.rollaxis(data, 0, 3)
                data = np.rot90(data, k=row.rot)
                data = np.rollaxis(data, 2, 0)

            if row.flip != -1:
                data = np.flip(data, row.flip)

            if row.noise is not None:
                data = data + np.random.random(data.shape)*row.noise

            sub = 0
            div = 1
            if self.normalize is not None:

                if self.normalize == "normalize":
                    sub = np.min(data)
                    data = data - sub

                    div = np.max(data)
                    data = data / div

                elif self.normalize == "center":
                    sub = np.mean(data)
                    data = data - sub

                elif self.normalize == "standardize":
                    sub = np.mean(data)
                    data = data - sub

                    div = np.std(data)
                    data = data / div

            if row.drop_frame != -1:
                data[row.drop_frame, :, :] = np.zeros(data[0, :, :].shape)

            x_indices = list(range(0, self.signal_frames[0])) + list(range(-self.signal_frames[1], 0))
            X_ = data[x_indices, :, :]
            X.append(X_)

            y_idx = self.signal_frames[0] + self.gap_frames[0]
            Y_ = data[y_idx, :, :]

            if (self.output_size is not None) and (self.output_size != Y_.shape):
                Y_ = resize(Y_, self.output_size)

            y.append(Y_)

        X = np.stack(X)
        y = np.stack(y)

        X = np.rollaxis(X, 1, 4)

        if self.cache_results:
            self.cache[index] = (X, y)

        return (X, y)

        # return (X, y)
        # X: [batch_size, frames, input_height, input_width]
        # y: [input_height, input_width]

    def __len__(self):
        return self.n // self.batch_size

class Network:

    def __init__(self, train_generator, val_generator=None, learning_rate=0.0001,
                 n_stacks=3, kernel=64, batchNormalize=False,
                 use_cpu=False):

        if use_cpu:
            tf.config.set_visible_devices([], 'GPU')

        self.train_gen = train_generator
        self.val_gen = val_generator

        self.model = self.create_unet(n_stacks=n_stacks, kernel=kernel, batchNormalize=batchNormalize)

        opt = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss=self.mean_squareroot_error,
                           # metrics=["val_loss", "loss", self.mean_squareroot_error]
                           )

    def run(self,
            batch_size=10, num_epochs = 25,
            patience=3, min_delta=0.005, monitor="val_loss",
            save_model=None, load_weights=False,
            verbose=1):

        if save_model is not None and not save_model.is_dir():
            print("created save dir at: ", save_model)
            save_model.mkdir()

        callbacks = [
            EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, verbose=verbose),
        ]

        if save_model is not None:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_model.joinpath("model-{epoch:02d}-{val_loss:.2f}.hdf5").as_posix(),
                    # filepath=save_model.joinpath("model-{epoch:02d}.hdf5").as_posix(),
                save_weights_only=False,
                monitor=monitor,
                mode='min',
                save_best_only=True,))

            if load_weights:
                print("loading previous weights!")
                latest_weights = tf.train.latest_checkpoint(save_model)
                self.model.load_weights(latest_weights)

        history = self.model.fit(self.train_gen,
                            batch_size=batch_size,
                            validation_data=self.val_gen,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            shuffle=False,
                            verbose=verbose)

        # save model
        if save_model is not None:
            self.model.save(save_model.joinpath("model.h5").as_posix())


        return history

    def get_vanilla_architecture(self, verbose=1):

        input_img = self.train_gen.__getitem__(0)
        input_window = Input((input_img[0].shape[1:]))

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_window)  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv3
        )  # 128 x 128 x 128
        up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128

        conc_up_1 = Concatenate()([up1, conv2])
        conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64

        conc_up_2 = Concatenate()([up2, conv1])
        decoded = Conv2D(1, (3, 3), activation=None, padding="same")(
            conc_up_2
        )  # 512 x 512 x 1

        decoder = Model(input_window, decoded)

        if verbose > 0:
            decoder.summary(line_length=100)

        return decoder

    def create_unet(self, n_stacks=3, kernel=64, batchNormalize=False, verbose=1):

        input_img = self.train_gen.__getitem__(0)
        input_window = Input((input_img[0].shape[1:]))

        last_layer = input_window

        if batchNormalize:
            last_layer = BatchNormalization()(last_layer)

        # enocder
        enc_conv = []
        for i in range(n_stacks):

            conv = Conv2D(kernel, (3, 3), activation="relu", padding="same")(last_layer)
            enc_conv.append(conv)

            if i != n_stacks-1:
                pool = MaxPooling2D(pool_size=(2, 2))(conv)

                kernel = kernel * 2
                last_layer = pool
            else:

                last_layer = conv

        # decoder
        for i in range(n_stacks):

            if i != n_stacks-1:

                conv = Conv2D(kernel, (3, 3), activation="relu", padding="same")(last_layer)
                up = UpSampling2D((2, 2))(conv)
                conc = Concatenate()([up, enc_conv[-(i+2)]])

                last_layer = conc
                kernel = kernel / 2
            else:

                decoded = Conv2D(1, (3, 3), activation=None, padding="same")(last_layer)

                decoder = Model(input_window, decoded)

        if verbose > 0:
            decoder.summary(line_length=100)

        return decoder

    @staticmethod
    def annealed_loss(y_true, y_pred):
        if not tf.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        local_power = 4
        final_loss = K.pow(K.abs(y_pred - y_true) + 0.00000001, local_power)
        return K.mean(final_loss, axis=-1)

    @staticmethod
    def mean_squareroot_error(y_true, y_pred):
        if not tf.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.mean(K.sqrt(K.abs(y_pred - y_true) + 0.00000001), axis=-1)


class DeepInterpolate:

    def __init__(self):
        print("hello")

    def infer(self, input_file, model, loc=None,
              output=None, dtype=np.float32, out_loc=None, chunk_size=(1, 1, 1), rescale=True,
              frames=None, batch_size=1, pre_post_frame=5):

        """
        :param input_file: tiff file or .h5 file
        :param model: exported deep neural network for export or if folder newest file is chosen
        :param loc: optional, if .h5 file is provided
        :param frames: infer all frames if None, else tuple expected (frame_start, frame_stop)
        :param batch_size: batch_size for inference; decrease if RAM is limited
        :param pre_post_frame: number of ommitted frames before and after inference frame
        :return:
        """

        # quality control
        assert os.path.isfile(input_file), "input doesn't exist: "+ input_file
        assert os.path.isdir(model) or os.path.isfile(model), "model doesn't exist: "+ model

        # create data generator
        # TODO add arguments
        data_generator = FullFrameGenerator(pre_post_frame, batch_size, steps_per_epoch=1,
                                            file_path=input_file, loc=loc, max_frame_summary_stats=1000,
                                            start_frame=0, end_frame=-1,
                                            gap_frames=0, total_samples=-1, randomize=False)

        # load model
        if os.path.isdir(model):
            models = list(filter(os.path.isfile, glob.glob(model + "/*.h5")))
            models.sort(key=lambda x: os.path.getmtime(x))
            model = models[0]

        model = load_model(model, custom_objects={"annealed_loss": self.annealed_loss})

        # infer
        num_datasets = len(data_generator)
        indiv_shape = data_generator.get_output_size()

        final_shape = [num_datasets * batch_size]
        first_sample = 0

        final_shape.extend(indiv_shape[:-1])

        if (output is None) or (output.endswith(".tiff")) or (output.endswith(".tiff")):
            dset_out = np.zeros(tuple(final_shape), dtype=dtype)

        elif output.endswith(".h5"):

            assert out_loc is not None, "when exporting results to .h5 file please provide 'out_loc' flag"

            f = h5.File(output, "a")
            dset_out = f.create_dataset(out_loc, shape=final_shape, chunks=chunk_size, dtype=dtype)

        for index_dataset in np.arange(0, num_datasets, 1):

            local_data = data_generator[index_dataset]
            predictions_data = model.predict(local_data[0])

            local_mean, local_std = \
                data_generator.__get_norm_parameters__(index_dataset)
            local_size = predictions_data.shape[0]

            corrected_data = predictions_data * local_std + local_mean if rescale else predictions_data

            start = first_sample + index_dataset * batch_size
            end = first_sample + index_dataset * batch_size \
                + local_size

            # We squeeze to remove the feature dimension from tensorflow
            dset_out[start:end, :] = np.squeeze(corrected_data, -1)

        if output is None:
            return dset_out

        elif output.endswith(".tiff") or output.endswith(".tif"):
            tiff.imwrite(output, data=dset_out)

    def annealed_loss(y_true, y_pred):
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        local_power = 4
        final_loss = K.pow(K.abs(y_pred - y_true) + 0.00000001, local_power)
        return K.mean(final_loss, axis=-1)

