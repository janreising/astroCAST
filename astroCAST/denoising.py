import logging
import os
import glob

import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import numpy as np
import h5py as h5
import tifffile as tiff

class Generator(keras.utils.Sequence):

    def __init__(self, pre_post_frame, batch_size, steps_per_epoch,
                 file_path, loc=None, max_frame_summary_stats=1000,
                 start_frame=0, end_frame=-1,
                 pre_post_omission=0, total_samples=-1, randomize=True):

        """

        :param pre_post_frame:
        :param batch_size:
        :param steps_per_epoch:
        :param start_frame:
        :param end_frame: compatlible with negative frames. -1 is the last
        :param pre_post_omission:
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
        self.pre_post_omission = pre_post_omission
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
        data_generator = Generator(pre_post_frame, batch_size, steps_per_epoch=1,
                 file_path=input_file, loc=loc, max_frame_summary_stats=1000,
                 start_frame=0, end_frame=-1,
                 pre_post_omission=0, total_samples=-1, randomize=False)

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

