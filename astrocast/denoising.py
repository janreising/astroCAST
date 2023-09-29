import logging
import os
import glob
import pathlib
import random
from pathlib import Path
from typing import Union

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import keras
from tensorflow.keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam

import numpy as np
import h5py as h5
import tifffile as tiff
from skimage.transform import resize
import pandas as pd

from scipy.stats import bootstrap

# TODO write plotting for generators
# TODO write plotting for network history

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
                 max_per_file=None,
                 overlap=None, padding=None,
                 shuffle=True, normalize=None,
                 loc="data/",
                 output_size=None, cache_results=False, in_memory=False, save_global_descriptive=True,
                 logging_level=logging.INFO):

        logging.basicConfig(level=logging_level)

        if type(paths) != list:
            paths = [paths]
        self.paths = paths
        self.loc = loc

        logging.debug(f"data files: {self.paths}")
        logging.debug(f"data loc: {self.loc}")

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.save_global_descriptive = save_global_descriptive

        if type(pre_post_frame) == int:
            pre_post_frame = (pre_post_frame, pre_post_frame)
        self.signal_frames = pre_post_frame

        if type(gap_frames) == int:
            gap_frames = (gap_frames, gap_frames)
        self.gap_frames = gap_frames

        self.z_steps = z_steps
        self.z_select = z_select
        self.max_per_file = max_per_file

        if (1 in allowed_rotation) or (3 in allowed_rotation):
            assert input_size[0] == input_size[1], f"when using 90 or 270 degree rotation (allowed rotation: 1 or 3) the 'input_size' needs to be square. However input size is: {input_size}"
        self.allowed_rotation = allowed_rotation

        self.allowed_flip = allowed_flip

        self.overlap = overlap  # float

        assert padding in [None, "symmetric", "edge"]
        assert not (random_offset and (padding is not None)), "cannot use 'padding' and 'random_offset' flag. Please choose one or the other!"
        self.padding = padding

        self.random_offset = random_offset
        self.add_noise = add_noise
        self.drop_frame_probability = drop_frame_probability

        # assert normalize in [None, "normalize", "center", "standardize"], "normalize argument needs be one of: [None, 'normalize', 'center', 'standardize']"
        assert normalize in [None, "local", "global"], "normalize argument needs be one of: [None, local, global]"
        self.normalize = normalize
        if self.normalize == "global":
                self.descr = {}

        self.shuffle = shuffle
        self.n = None

        # in memory
        self.mem_data = {} if in_memory else -1

        # get items
        self.fov_size = None
        self.items = self.generate_items()

        # cache
        self.cache_results = cache_results
        if cache_results:
            logging.warning("using caching may lead to memory leaks. Please set to false if you experience Out-Of-Memory errors.")

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

        # enforce tuples for signal and gap frames
        if isinstance(self.signal_frames, int):
            signal_frames = (self.signal_frames, self.signal_frames)
        else:
            signal_frames = self.signal_frames

        if isinstance(self.gap_frames, int):
            gap_frames = (self.gap_frames, self.gap_frames)
        else:
            gap_frames = self.gap_frames

        # define prediction length (Z)
        stack_len = signal_frames[0] + gap_frames[0] + 1 + gap_frames[1] + signal_frames[1]
        z_steps = max(1, int(self.z_steps * stack_len))

        self.fov_size = (stack_len, dw, dh)

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
        for file in tqdm(self.paths, desc="file preprocessing"):

            file_container = []
            file = Path(file)
            assert file.is_file(), "can't find: {}".format(file)

            if file.suffix == ".h5":
                with h5.File(file.as_posix(), "r") as f:

                    if self.loc not in f:
                        logging.warning(f"cannot find {self.loc} in {file}")
                        continue

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

            # Calculate padding (if applicable)
            if self.padding is not None:
                pad_z0 = signal_frames[0] + gap_frames[0]
                pad_z1 = signal_frames[1] + gap_frames[1] + 1
                pad_x1 = iw % X
                pad_y1 = ih % Y
            else:
                pad_z0 = pad_z1 = pad_x1 = pad_y1 = 0

            zRange =list(range(Z0 + z_start - pad_z0, Z1 - stack_len - z_start + pad_z1, z_steps))
            xRange = list(range(x_start, X - x_start + pad_x1, dw))
            yRange = list(range(y_start, Y - y_start + pad_y1, dh))

            logging.debug(f"\nz_range: {zRange}")
            logging.debug(f"\nx_range: {xRange}")
            logging.debug(f"\nx_range param > x_start:{x_start}, X:{X} iw:{iw} pad_x1:{pad_x1}, dw:{dw}")
            logging.debug(f"\ny_range: {yRange}")

            if self.shuffle:
                random.shuffle(zRange)
                random.shuffle(xRange)
                random.shuffle(yRange)

            for z0 in zRange:
                z1 = z0 + stack_len

                for x0 in xRange:
                    x1 = x0 + iw

                    for y0 in yRange:
                        y1 = y0 + ih

                        # choose modification
                        rot = random.choice(allowed_rotation)
                        flip = random.choice(allowed_flip)

                        # mark dropped frame (if applicable)
                        if (self.drop_frame_probability is not None) and (np.random.random() <= self.drop_frame_probability):
                            drop_frame = np.random.randint(0, np.sum(signal_frames))
                        else:
                            drop_frame = -1

                        # calculate necessary padding
                        padding = np.zeros(4, dtype=int)

                        if not z0 > 0:
                            padding[0] = 0 - z0

                        if not z1 < Z1:
                            padding[1] = z1 - Z1

                        if not x1 < X:
                            padding[2] = x1 - X

                        if not y1 < Y:
                            padding[3] = y1 - Y

                        # cannot pad on empty axis
                        if (padding[0] >= stack_len) or (padding[1] >= stack_len) or (padding[2] >= dw) or (padding[3] >= dh):
                            continue

                        # create item
                        item = {
                            "idx": idx, "path": file, "z0": z0, "z1": z1, "x0": x0, "x1": x1, "y0": y0,
                            "y1": y1, "rot": rot, "flip": flip,
                            "noise": self.add_noise, "drop_frame": drop_frame,
                            "padding": padding}

                        file_container.append(item)

                        idx += 1

            file_container = pd.DataFrame(file_container)

            if len(file_container) < 1:
                raise ValueError("cannot find suitable data chunks.")

            if self.normalize == "global":
                if len(file_container) > 1:

                    local_save = self.get_local_descriptive(file, h5_loc=self.loc)

                    if local_save is None and not self.save_global_descriptive:
                        self.descr[file] = self._bootstrap_descriptive(file_container)

                    elif local_save is None:

                        mean, std = self._bootstrap_descriptive(file_container)
                        self.descr[file] = (mean, std)

                        if self.save_global_descriptive:
                            self.set_local_descriptive(file, h5_loc=self.loc, mean=mean, std=std)

                    else:
                        logging.info("loading descriptive statistics from file")
                        self.descr[file] = local_save

                else:
                    logging.warning(f"found file without eligible items: {file}")

            if self.max_per_file is not None:
                file_container = file_container.sample(self.max_per_file)

            container.append(file_container)

        items = pd.concat(container)
        logging.debug(f"items: {items}")

        if self.shuffle:
            items = items.sample(frac=1).reset_index(drop=True)

        items["batch"] = (np.array(range(len(items))) / self.batch_size)
        items.batch = items.batch.astype(int) # round down

        self.n = len(items)

        return items

    def on_epoch_end(self):

        # called after each epoch
        if self.shuffle:
            if self.random_offset:
                self.items = self.generate_items()
                self.cache = {}
            else:
                self.items = self.items.sample(frac=1).reset_index(drop=True)

    def _bootstrap_descriptive(self, items, frac=0.01, confidence_level=0.75, n_resamples=5000,
                               max_confidence_span=0.2, iteration_frac_increase=1.5):

        means = []
        raws = []

        confidence_span = mean_conf = std_conf = max_confidence_span + 1

        def custom_bootstrap(data):

            try:
                res = bootstrap((np.array(data),), np.median, n_resamples=n_resamples, axis=0, vectorized=False, confidence_level=confidence_level, method="bca")
            except:
                logging.warning("all values are the same. Using np.mean instead of bootstrapping")
                return np.mean(np.array(data)), 0

            # calculate confidence span
            ci = res.confidence_interval
            mean = np.mean((ci.low, ci.high))
            confidence_span = (ci.high-ci.low)/mean

            return mean, confidence_span

        it = 0
        while (frac < 1) and confidence_span > max_confidence_span:

            sel_items = items.sample(frac=frac)
            if len(sel_items) < 1:
                logging.debug(f"items: {items}")
                logging.debug(f"sel items: {sel_items}")
            for _, row in sel_items.iterrows():

                raw = self._load_row(row.path, row.z0, row.z1, row.x0, row.x1, row.y0, row.y1).flatten()
                raws.append(raw)

                means += [np.nanmedian(raw)]

            mean_, mean_conf = custom_bootstrap(means)

            std_, std_conf = custom_bootstrap([np.std(r-mean_) for r in raws])

            confidence_span = max(mean_conf, std_conf)

            # increase frac
            frac *= iteration_frac_increase
            it += 1
            logging.debug(f"iteration {it} {mean_:.2f}+-{mean_conf:.2f} / {std_:.2f}+-{std_conf:.2f} ")

        if mean_ is None or np.isnan(mean_):
            logging.warning(f"unable to calculate mean")
            mean_ = np.nanmean(means)

        if std_ is None or np.isnan(std_):
            logging.warning(f"unable to calculate std")
            std_ = np.nanmean([np.std(r-mean_) for r in raws])

        return mean_, std_

    def _load_row(self, path, z0, z1, x0, x1, y0, y1, padding=None):

        if padding is not None:
            pad_z0, pad_z1, pad_x1, pad_y1 = padding

            z0 = z0 + pad_z0
            z1 = z1 - pad_z1
            x1 = x1 - pad_x1
            y1 = y1 - pad_y1

        if type(self.mem_data) == dict:

            # load to memory if necessary
            if path not in self.mem_data.keys():

                if path.suffix == ".h5":
                    with h5.File(path.as_posix(), "r") as f:
                        self.mem_data[path] = f[self.loc][:]

                elif path.suffix in (".tif", ".tiff"):
                    self.mem_data[path] = tiff.imread(path.as_posix())

            data = self.mem_data[path][z0:z1, x0:x1, y0:y1]

        elif path.suffix == ".h5":
            with h5.File(path.as_posix(), "r") as f:
                data = f[self.loc][z0:z1, x0:x1, y0:y1]

        elif path.suffix in (".tif", ".tiff"):
            data = tiff.imread(path.as_posix(), key=range(z0, z1))
            data = data[:, x0:x1, y0:y1]

        if (padding is not None) and np.sum((pad_z0, pad_z1, pad_x1, pad_y1)) > 0:
            data = np.pad(data, ((pad_z0, pad_z1), (0, pad_x1), (0, pad_y1)),
                          mode=self.padding)

        return data

    def __getitem__(self, index):

        if index in self.cache.keys():
            return self.cache[index]

        X = []
        y = []
        for _, row in self.items[self.items.batch == index].iterrows():

            data = self._load_row(row.path, row.z0, row.z1, row.x0, row.x1, row.y0, row.y1,
                                  padding=None if self.padding is None else row.padding)

            assert data.shape == self.fov_size, f"loaded data does not match expected FOV size " \
                                                f"(fov: {self.fov_size}) vs. (load: {data.shape}"

            if row.rot != 0:

                data = np.rollaxis(data, 0, 3)
                data = np.rot90(data, k=row.rot)
                data = np.rollaxis(data, 2, 0)

            if row.flip != -1:
                data = np.flip(data, row.flip)

            if row.noise is not None:
                data = data + np.random.random(data.shape)*row.noise

            if self.normalize == "local":
                sub = np.mean(data)
                data = data - sub

                div = np.std(data)
                data = data / div

            elif self.normalize == "global":
                sub, div = self.descr[row.path]

                data = data - sub
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

    def __len__(self):
        # return self.n // self.batch_size
        return len(self.items.batch.unique())

    def __get_norm_parameters__(self, index):

        files = self.items[self.items.batch == index].path.unique()

        if len(files) == 1:
            return self.descr[files[0]] if files[0] in self.descr.keys() else (1, 1)
        elif len(files) > 1:
            return {f:self.descr[f] if f in self.descr.keys() else (1, 1) for f in files}
        else:
            return None

    def infer(self, model, output=None,
              out_loc=None, dtype="same", chunk_size=None, rescale=True):

        # load model if not provided
        if type(model) in [str, pathlib.PosixPath]:
            model = Path(model)

            if os.path.isdir(model):

                models = list(model.glob("*.h5"))

                if len(models) < 1:
                    raise FileNotFoundError(f"cannot find model in provided directory: {model}")

                models.sort(key=lambda x: os.path.getmtime(x))
                model_path = models[0]
                logging.info(f"directory provided. Selected most recent model: {model_path}")

            elif os.path.isfile(model):
                model_path = model

            else:
                raise FileNotFoundError(f"cannot find model: {model}")

            model = load_model(model_path,
                    custom_objects={"annealed_loss": Network.annealed_loss, "mean_squareroot_error":Network.mean_squareroot_error})

        else:
            logging.warning(f"providing model via parameter. Model type: {type(model)}")

            # deals with keras.__version__ > 2.10.0
            try:
                expected_model_type = keras.engine.functional.Functional
            except AttributeError:
                expected_model_type = keras.src.engine.functional.Functional

            assert type(model) == expected_model_type, f"Please provide keras model, " \
                                                       f"file_path or dir_path instead of {type(model)}"

        # enforce pathlib.Path
        if output is not None:
            output = Path(output)

        # create output arrays
        assert len(self.items.path.unique()) < 2, f"inference from multiple files is currently not implemented: {self.items.path.unique()}"
        items = self.items

        if "padding" in items.columns:
            pad_z_max = items.padding.apply(lambda x: x[1]).max()
            pad_x_max = items.padding.apply(lambda x: x[2]).max()
            pad_y_max = items.padding.apply(lambda x: x[3]).max()
        else:
            pad_z_max, pad_x_max, pad_y_max = 0, 0, 0

        output_shape = (items.z1.max()-pad_z_max, items.x1.max()-pad_x_max, items.y1.max()-pad_y_max)

        if dtype == "same":
            x, _ = self[self.items.batch.tolist()[0]] # raw data
            dtype = x.dtype
            logging.warning(f"choosing dtype: {dtype}")

        if output is not None and output.suffix in (".h5", ".hdf5"):

            assert out_loc is not None, "when exporting results to .h5 file please provide 'out_loc' flag"

            f = h5.File(output, "a")
            rec = f.create_dataset(out_loc, shape=output_shape, chunks=chunk_size, dtype=dtype)
        else:
            rec = np.zeros(output_shape, dtype=dtype)

        # infer frames
        for batch in tqdm(self.items.batch.unique()):

            x, _ = self[batch] # raw data
            y = model.predict(x) # denoised data

            if dtype != y.dtype:
                y = y.astype(dtype)

            x_items = items[items.batch == batch] # meta data

            assert len(x) == len(x_items), f"raw and meta data must have the same length: raw ({len(x)}) vs. meta ({len(x_items)})"

            c=0
            for _, row in x_items.iterrows():

                im = y[c, :, :, 0]

                pad_z0, pad_z1, pad_x1, pad_y1 = row.padding

                if pad_x1 > 0:
                    im = im[:-pad_x1, :]

                if pad_y1 > 0:
                    im = im[:, :-pad_y1]

                if rescale:
                    mean, std = self.descr[self.items.iloc[0].path]
                    im = (im * std) + mean

                rec[row.z0+5, row.x0:row.x1, row.y0:row.y1] = im

                c+=1

        if output is None:
            return rec

        elif output.suffix in (".tiff", ".tif"):
            tiff.imwrite(output, data=rec)
            return output

        elif output.suffix in (".h5", ".hdf5"):
            f.close()
            return output

    @staticmethod
    def get_local_descriptive(path, h5_loc):

        if path.suffix != ".h5":
            # local save only implemented for hdf5 files
            return None

        mean_loc = "/descr/" + h5_loc + "/mean"
        std_loc = "/descr/" + h5_loc + "/std"

        with h5.File(path.as_posix(), "r") as f:

            if mean_loc in f:
                mean = f[mean_loc][0]
            else:
                return None

            if std_loc in f:
                std = f[std_loc][0]
            else:
                return None

        return mean, std

    @staticmethod
    def set_local_descriptive(path, h5_loc, mean, std):

        logging.warning("saving results of descriptive")

        if path.suffix != ".h5" or mean is None or std is None:
            # local save only implemented for hdf5 files
            return False

        mean_loc = "/descr/" + h5_loc + "/mean"
        std_loc = "/descr/" + h5_loc + "/std"

        with h5.File(path.as_posix(), "a") as f:

            if mean_loc not in f:
                f.create_dataset(mean_loc, shape=(1), dtype=float, data=mean)
            else:
                f[mean_loc] = mean

            if std_loc not in f:
                f.create_dataset(std_loc, shape=(1), dtype=float, data=std)
            else:
                f[std_loc] = std

        return True


class Network:
    def __init__(self, train_generator, val_generator=None, learning_rate=0.0001,
                 n_stacks=3, kernel=64, batchNormalize=False, loss=None,
                 use_cpu=False):
        """
        Initializes the Network class.

        Args:
            train_generator: The generator for training data.
            val_generator: The generator for validation data.
            learning_rate (float): The learning rate for the optimizer.
            n_stacks (int): The number of stacks in the U-Net model.
            kernel (int): The number of filters in the first layer of the U-Net model.
            batchNormalize (bool): Whether to apply batch normalization in the U-Net model.
            loss: The loss function to use. If None, the annealed_loss function will be used.
            use_cpu (bool): Whether to use the CPU for training (disable GPU).

        Returns:
            None
        """

        if use_cpu:
            # Set the visible GPU devices to an empty list to use CPU
            tf.config.set_visible_devices([], 'GPU')

        # Assign the train and validation generators
        self.train_gen = train_generator
        self.val_gen = val_generator

        # Create the U-Net model
        self.n_stacks = n_stacks
        self.kernel = kernel
        self.model = self.create_unet(n_stacks=n_stacks, kernel=kernel, batchNormalize=batchNormalize)

        # Set the optimizer and compile the model
        opt = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt,
                           loss=self.annealed_loss if loss is None else loss,
                           # metrics=["val_loss", "loss", self.mean_squareroot_error]
                           )

    def run(self,
            batch_size=10, num_epochs=25,
            patience=3, min_delta=0.005, monitor="val_loss",
            save_model=None, load_weights=False,
            verbose=1):
        """
        Trains the model.

        Args:
            batch_size (int): Number of samples per gradient update.
            num_epochs (int): Number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Quantity to be monitored during training.
            save_model (str or pathlib.Path): Directory to save the model and checkpoints.
            load_weights (bool): Whether to load the previous weights if available.
            verbose (int): Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch).

        Returns:
            tf.keras.History: Object containing the training history.
        """

        if save_model is not None and not save_model.is_dir():
            logging.info("Created save dir at: %s", save_model)
            save_model.mkdir()

        callbacks = [
            # Early stopping callback to stop training if no improvement
            EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, verbose=verbose),
        ]

        if save_model is not None:
            if type(save_model) == str:
                save_model = Path(save_model)

            # Model checkpoint callback to save the best model during training
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_model.joinpath("model-{epoch:02d}-{val_loss:.2f}.hdf5").as_posix(),
                    save_weights_only=False,
                    monitor=monitor,
                    mode='min',
                    save_best_only=True,
                )
            )

            if load_weights:
                logging.info("Loading previous weights!")
                latest_weights = tf.train.latest_checkpoint(save_model)

                if latest_weights is not None:
                    self.model.load_weights(latest_weights)

        # Start model training
        history = self.model.fit(
            self.train_gen,
            batch_size=batch_size,
            validation_data=self.val_gen,
            epochs=num_epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=verbose,  # Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch)
        )

        # Save the final model
        if save_model is not None:
            # Create a filename with parameters
            input_shape_str = "x".join(map(str, self.train_gen.__getitem__(0)[0].shape[1:]))
            param_str = f"lr={self.model.optimizer.learning_rate}_nstacks={self.n_stacks}_kernel={self.kernel}_inputshape={input_shape_str}"
            save_path = save_model.joinpath(f"model-{param_str}-{{epoch:02d}}-{{val_loss:.2f}}.hdf5").as_posix()
            print(f"saved model to: {save_path}")

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

    def retrain_model(self, new_train_gen, model=None, new_val_gen=None, pretrained_model_path=None, num_epochs=25, batch_size=10, verbose=1):
        """
        Retrains the model on a new dataset and optionally initializes it with weights from a pretrained model.

        Args:
            new_train_gen: The generator for the new training data.
            new_val_gen: The generator for the new validation data.
            pretrained_model_path (str): Path to the pretrained model.
            num_epochs (int): Number of epochs to retrain the model.
            batch_size (int): Number of samples per gradient update.
            verbose (int): Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch).

        Returns:
            tf.keras.History: Object containing the retraining history.
        """
        self.train_gen = new_train_gen
        self.val_gen = new_val_gen

        new_input_shape = self.train_gen.__getitem__(0)[0].shape[1:]
        new_model = self.create_unet(input_shape=new_input_shape)

        if pretrained_model_path is not None:
            # Load pre-trained weights into the new model
            new_model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)

            # Load the pretrained model to get its layer names
            pretrained_model = tf.keras.models.load_model(pretrained_model_path)
            pretrained_layer_names = [layer.name for layer in pretrained_model.layers]

            # Freeze layers that are both in the new model and the pre-trained model
            for layer in new_model.layers:
                if layer.name in pretrained_layer_names:
                    layer.trainable = False
                else:
                    layer.trainable = True  # New or mismatched layers

        self.model = new_model

        # You might want to set a different optimizer or keep it as is
        self.model.compile(optimizer=self.model.optimizer,
                           loss=self.model.loss,
                           # Add metrics if any
                           )

        return self.run(num_epochs=num_epochs, batch_size=batch_size, verbose=verbose)


    def create_unet(self, input_shape=None, n_stacks=3, kernel=64, batchNormalize=False, verbose=1):
        """
        Creates a U-Net model.

        Args:
            n_stacks (int): Number of encoding and decoding stacks.
            kernel (int): Number of filters in the first convolutional layer.
            batchNormalize (bool): Whether to apply batch normalization.
            verbose (int): Verbosity mode (0 - silent, 1 - summary).

        Returns:
            tf.keras.Model: The U-Net model.
        """

        # Input
        if input_shape is None:
            input_shape = self.train_gen.__getitem__(0)[0].shape[1:]

        input_window = Input(input_shape)


        last_layer = input_window

        if batchNormalize:
            # Apply batch normalization to the input
            last_layer = BatchNormalization()(last_layer)

        # Encoder
        enc_conv = []
        for i in range(n_stacks):
            # Convolutional layer in the encoder
            conv = Conv2D(kernel, (3, 3), activation="relu", padding="same")(last_layer)
            enc_conv.append(conv)

            if i != n_stacks-1:
                # Max pooling layer in the encoder
                pool = MaxPooling2D(pool_size=(2, 2))(conv)

                kernel = kernel * 2
                last_layer = pool
            else:
                # Last convolutional layer in the encoder
                last_layer = conv

        # Decoder
        for i in range(n_stacks):
            if i != n_stacks-1:
                # Convolutional layer in the decoder
                conv = Conv2D(kernel, (3, 3), activation="relu", padding="same")(last_layer)
                up = UpSampling2D((2, 2))(conv)
                conc = Concatenate()([up, enc_conv[-(i+2)]])

                last_layer = conc
                kernel = kernel / 2
            else:
                # Last convolutional layer in the decoder
                decoded = Conv2D(1, (3, 3), activation=None, padding="same")(last_layer)
                decoder = Model(input_window, decoded)

        if verbose > 0:
            # Print model summary
            decoder.summary(line_length=100)

        return decoder

    @staticmethod
    def annealed_loss(y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """
        Calculates the annealed loss between the true and predicted values.

        Args:
            y_true (Union[tf.Tensor, np.ndarray]): The true values.
            y_pred (Union[tf.Tensor, np.ndarray]): The predicted values.

        Returns:
            tf.Tensor: The calculated annealed loss.
        """
        if not tf.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        local_power = 4
        final_loss = K.pow(K.abs(y_pred - y_true) + 0.00000001, local_power)  # Compute the element-wise absolute difference and apply local power
        return K.mean(final_loss, axis=-1)  # Compute the mean of the final loss along the last axis

    @staticmethod
    def mean_squareroot_error(y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """
        Calculates the mean square root error between the true and predicted values.

        Args:
            y_true (Union[tf.Tensor, np.ndarray]): The true values.
            y_pred (Union[tf.Tensor, np.ndarray]): The predicted values.

        Returns:
            tf.Tensor: The calculated mean square root error.
        """
        if not tf.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.mean(K.sqrt(K.abs(y_pred - y_true) + 0.00000001), axis=-1)  # Compute the element-wise absolute difference, apply square root, and compute mean along the last axis
