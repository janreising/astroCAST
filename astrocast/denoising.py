import logging
import os
import pathlib
import random
from pathlib import Path
from typing import Union, List, Tuple, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from astrocast.autoencoders import EarlyStopper
from astrocast.helper import closest_power_of_two
from astrocast.preparation import IO

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

PyTorchLoss = Union[
    Literal['annealed_loss', 'mean_square_root_error'], Callable[[torch.Tensor, torch.Tensor], torch.Tensor], _Loss]


# TODO write plotting for generators
# TODO write plotting for network history

class SubFrameDataset(Dataset):

    def __init__(
            self, paths: Union[str, List[str], Path, List[Path]], input_size: Tuple[int, int] = (32, 32),
            pre_post_frames: Union[int, Tuple[int, int]] = 5, gap_frames: Union[int, Tuple[int, int]] = 0,
            z_steps: float = 0.1, z_select: Union[None, int, List[int]] = None,
            allowed_rotation: Union[int, List[int]] = 0,
            allowed_flip: Union[int, List[int]] = -1, random_offset: bool = False, add_noise: bool = False,
            drop_frame_probability: Union[None, float] = None, max_per_file: Union[None, int] = None, overlap: int = 0,
            padding: Union[None, Literal["symmetric", "edge"]] = None, shuffle: bool = True,
            normalize: Union[None, Literal["local", "global"]] = None, loc: str = "data/",
            output_size: Union[None, Tuple[int, int]] = None, cache_results: bool = False, in_memory: bool = False,
            save_global_descriptive: bool = True, logging_level: int = logging.INFO
    ):

        logging.basicConfig(level=logging_level)

        # adjust input_size to be power of two
        input_size = [closest_power_of_two(v) for v in input_size]

        if not isinstance(paths, list):
            paths = [paths]
        self.paths = paths
        self.loc = loc

        logging.debug(f"data files: {self.paths}")
        logging.debug(f"data loc: {self.loc}")

        self.input_size = input_size
        self.output_size = output_size
        self.save_global_descriptive = save_global_descriptive

        if isinstance(pre_post_frames, int):
            pre_post_frames = (pre_post_frames, pre_post_frames)
        self.signal_frames = pre_post_frames

        if isinstance(gap_frames, int):
            gap_frames = (gap_frames, gap_frames)
        self.gap_frames = gap_frames

        self.z_steps = z_steps
        self.z_select = z_select
        self.max_per_file = max_per_file
        self.item_size = None

        # parse allowed rotations
        if isinstance(allowed_rotation, int):
            allowed_rotation = [allowed_rotation]

        if (1 in allowed_rotation) or (3 in allowed_rotation):
            assert input_size[0] == input_size[
                1], (f"when using 90 or 270 degree rotation (allowed rotation: 1 or 3) the 'input_size' needs to be "
                     f"square. However input size is: {input_size}")
        self.allowed_rotation = allowed_rotation

        # parse allowed flips
        if isinstance(allowed_flip, int):
            allowed_flip = [allowed_flip]
        self.allowed_flip = allowed_flip

        if random_offset and overlap is not None:
            raise ValueError(f"random_offset and overlap are incompatible. Please choose only one.")

        self.overlap = overlap
        self.overlap_x = None
        self.overlap_y = None

        assert padding in [None, "symmetric", "edge"]
        assert not (random_offset and (
                padding is not None)), "cannot use 'padding' and 'random_offset' flag. Please choose one or the other!"
        self.padding = padding

        self.random_offset = random_offset
        self.add_noise = add_noise
        self.drop_frame_probability = drop_frame_probability

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
        self.items = self._generate_items()

        # cache
        self.cache_results = cache_results
        if cache_results:
            logging.warning("using caching may lead to memory leaks. Please set to false if you experience "
                            "Out-Of-Memory errors.")

        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        # get cached value if it exists
        if index in self.cache.keys():
            return self.cache[index]

        # load item information and load from disk
        row = self.items.iloc[index]
        data = self._load_row(row)

        if data.shape != self.fov_size:
            raise RuntimeError(f"loaded data does not match expected FOV size "
                               f"(fov: {self.fov_size}) vs. (load: {data.shape}")

        # apply rotation
        if row.rot != 0:
            data = np.rollaxis(data, 0, 3)
            data = np.rot90(data, k=row.rot)
            data = np.rollaxis(data, 2, 0)

        # apply flip
        if row.flip != -1:
            data = np.flip(data, row.flip)

        # apply noise
        if row.noise is not None:
            data = data + np.random.random(data.shape) * row.noise

        # apply normalization
        if self.normalize == "local":
            sub = np.mean(data)
            data = data - sub

            div = np.std(data)
            data = data / div

        elif self.normalize == "global":
            sub, div = self.descr[row.path]

            data = data - sub
            data = data / div

        # apply frame dropping
        if row.drop_frame != -1:
            data[row.drop_frame, :, :] = np.zeros(data[0, :, :].shape)

        x_indices = list(range(0, self.signal_frames[0])) + list(range(-self.signal_frames[1], 0))
        input_frames = data[x_indices, :, :]

        y_idx = self.signal_frames[0] + self.gap_frames[0]
        target_frame = data[y_idx, :, :]

        if (self.output_size is not None) and (self.output_size != target_frame.shape):
            target_frame = resize(target_frame, self.output_size)

        if self.cache_results:
            self.cache[index] = (input_frames, target_frame)

        return input_frames, target_frame

    def _generate_items(self):

        # define size of each predictive field of view (X, Y)
        dw, dh = self.input_size
        signal_frames = self.signal_frames
        gap_frames = self.gap_frames

        # define prediction length (Z)
        stack_len = signal_frames[0] + gap_frames[0] + 1 + gap_frames[1] + signal_frames[1]
        z_steps = max(1, int(self.z_steps * stack_len))

        self.fov_size = (stack_len, dw, dh)
        self.item_size = (signal_frames[0] + signal_frames[1], dw, dh)

        x_start, y_start, z_start = 0, 0, 0
        # randomize input
        if self.random_offset:
            x_start = np.random.randint(0, dw)
            y_start = np.random.randint(0, dh)
            z_start = np.random.randint(0, stack_len)

        # adjust for overlap
        overlap = self.overlap
        if overlap is not None:
            if overlap < 1:
                overlap_x, overlap_y = int(dw * overlap), int(dh * overlap)
            else:
                overlap_x, overlap_y = overlap, overlap

        else:
            overlap_x, overlap_y = 0, 0

        self.overlap_x, self.overlap_y = overlap_x, overlap_y

        # adjust rotation and flip
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
                raise NotImplementedError(
                    f"filetype is recognized - please provide .h5, .tif or .tiff instead of: {file}"
                )

            if self.z_select is not None:
                Z0 = max(0, self.z_select[0])
                Z1 = min(Z, self.z_select[1])
            else:
                Z0, Z1 = 0, Z

            # Calculate padding (if applicable)
            if self.padding is not None:
                pad_z0 = signal_frames[0] + gap_frames[0]
                pad_z1 = signal_frames[1] + gap_frames[1] + 1
                pad_x1 = dw % X
                pad_y1 = dh % Y
            else:
                pad_z0 = pad_z1 = pad_x1 = pad_y1 = 0

            zRange = list(
                range(
                    Z0 + z_start - pad_z0, Z1 - stack_len - z_start + pad_z1, z_steps
                )
            )
            xRange = list(
                range(
                    x_start, X - x_start + pad_x1 - dw, dw - overlap_x
                )
            )
            yRange = list(
                range(
                    y_start, Y - y_start + pad_y1 - dh, dh - overlap_y
                )
            )

            logging.debug(f"\nz_range: {zRange}")
            logging.debug(f"\nx_range: {xRange}")
            logging.debug(f"\nx_range param > x_start:{x_start}, X:{X} pad_x1:{pad_x1}, dw:{dw}")
            logging.debug(f"\ny_range: {yRange}")

            if self.shuffle:
                random.shuffle(zRange)
                random.shuffle(xRange)
                random.shuffle(yRange)

            for z0 in zRange:
                z1 = z0 + stack_len

                for x0 in xRange:
                    x1 = x0 + dw

                    for y0 in yRange:
                        y1 = y0 + dh

                        # choose modification
                        rot = random.choice(allowed_rotation)
                        flip = random.choice(allowed_flip)

                        # mark dropped frame (if applicable)
                        if (self.drop_frame_probability is not None) and (
                                np.random.random() <= self.drop_frame_probability):
                            drop_frame = np.random.randint(0, np.sum(signal_frames))
                        else:
                            drop_frame = -1

                        # calculate necessary padding
                        padding = np.zeros(6, dtype=int)

                        padding[0] = min(0, z0)
                        padding[1] = max(0, z1 - Z1)
                        padding[3] = max(0, x1 - X)
                        padding[5] = max(0, y1 - Y)

                        padding = np.abs(padding)

                        # cannot pad on empty axis
                        if (padding[0] >= stack_len) or (padding[1] >= stack_len) or (padding[2] >= dw) or (
                                padding[3] >= dh):
                            continue

                        # create item
                        item = {"idx": idx, "path": file, "z0": z0, "z1": z1, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                                "rot": rot, "flip": flip, "Z": Z, "X": X, "Y": Y, "noise": self.add_noise,
                                "drop_frame": drop_frame, "padding": padding}

                        file_container.append(item)

                        idx += 1

            file_container = pd.DataFrame(file_container)

            if len(file_container) < 1:
                raise ValueError("cannot find suitable data chunks.")

            if self.normalize == "global":
                if len(file_container) > 1:

                    local_save = self._get_local_descriptive(file, loc=self.loc)

                    if local_save is None and not self.save_global_descriptive:
                        self.descr[file] = self._bootstrap_descriptive(file_container)

                    elif local_save is None:

                        mean, std = self._bootstrap_descriptive(file_container)
                        self.descr[file] = (mean, std)

                        if self.save_global_descriptive:
                            self._set_local_descriptive(file, loc=self.loc, mean=mean, std=std)

                    else:
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

        self.n = len(items)

        return items

    def _bootstrap_descriptive(
            self, items, frac=0.01, confidence_level=0.75, n_resamples=5000, max_confidence_span=0.2,
            iteration_frac_increase=1.5
    ):

        means = []
        raws = []

        confidence_span = mean_conf = std_conf = max_confidence_span + 1

        def custom_bootstrap(data):

            try:
                res = bootstrap(
                    (np.array(data),), np.median, n_resamples=n_resamples, axis=0, vectorized=False,
                    confidence_level=confidence_level, method="bca"
                )
            except:
                logging.warning("all values are the same. Using np.mean instead of bootstrapping")
                return np.mean(np.array(data)), 0

            # calculate confidence span
            ci = res.confidence_interval
            mean = np.mean((ci.low, ci.high))
            confidence_span = (ci.high - ci.low) / mean

            return mean, confidence_span

        it = 0
        mean_ = None
        std_ = None
        while (frac < 1) and confidence_span > max_confidence_span:

            sel_items = items.sample(frac=frac)
            if len(sel_items) < 1:
                logging.debug(f"items: {items}")
                logging.debug(f"sel items: {sel_items}")
            for _, row in sel_items.iterrows():
                raw = self._load_row(row).flatten()
                raws.append(raw)

                means += [np.nanmedian(raw)]

            mean_, mean_conf = custom_bootstrap(means)

            std_, std_conf = custom_bootstrap([np.std(r - mean_) for r in raws])

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
            std_ = np.nanmean([np.std(r - mean_) for r in raws])

        return mean_, std_

    def _load_row(self, row):

        path, z0, z1, x0, x1, y0, y1, Z, X, Y = row.path, row.z0, row.z1, row.x0, row.x1, row.y0, row.y1, row.Z, row.X, row.Y
        pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1 = row.padding

        # adjust boundaries if padding is needed
        if sum([pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1]) > 0:

            # adjust Z boundaries
            if z0 < 0:
                pad_z0 = abs(z0)
                z0 = 0
            else:
                pad_z0 = 0

            if z1 > Z:
                pad_z1 = abs(Z - z1)
                z1 = Z
            else:
                pad_z1 = 0

            # adjust X boundaries
            if x0 < 0:
                pad_x0 = abs(x0)
                x0 = 0
            else:
                pad_x0 = 0

            if x1 > X:
                pad_x1 = abs(X - x1)
                x1 = X
            else:
                pad_x1 = 0

            # adjust Y boundaries
            if y0 < 0:
                pad_y0 = abs(y0)
                y0 = 0
            else:
                pad_y0 = 0

            if y1 > Y:
                pad_y1 = abs(Y - y1)
                y1 = Y
            else:
                pad_y1 = 0

        if isinstance(self.mem_data, dict):

            # load to memory if necessary
            if path not in self.mem_data.keys():

                if path.suffix == ".h5":
                    with h5.File(path.as_posix(), "r") as f:
                        self.mem_data[path] = f[self.loc][:]

                elif path.suffix in (".tif", ".tiff"):
                    self.mem_data[path] = tiff.imread(path.as_posix())

            data = self.mem_data[path][z0:z1, x0:x1, y0:y1]

        # todo: this should probably utilize helper.IO and not be implemented here
        #  in that case we need to add a x_select and y_select variable in IO.load()
        elif path.suffix == ".h5":
            with h5.File(path.as_posix(), "r") as f:
                data = f[self.loc][z0:z1, x0:x1, y0:y1]

        elif path.suffix in (".tif", ".tiff"):
            data = tiff.imread(path.as_posix(), key=range(z0, z1))
            data = data[:, x0:x1, y0:y1]
        else:
            raise ValueError(f"Unexpected suffix (? {path.suffix}), please provide one of [.tiff, .h5].")

        # pad loaded data if necessary
        if np.sum((pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1)) > 0:
            data = np.pad(
                data, ((pad_z0, pad_z1), (pad_x0, pad_x1), (pad_y0, pad_y1)), mode=self.padding
            )

        return data

    def __get_norm_parameters__(self, index):

        files = self.items[self.items.batch == index].path.unique()

        if len(files) == 1:
            return self.descr[files[0]] if files[0] in self.descr.keys() else (1, 1)
        elif len(files) > 1:
            return {f: self.descr[f] if f in self.descr.keys() else (1, 1) for f in files}
        else:
            return None

    @staticmethod
    def _get_local_descriptive(path, loc):

        if path.suffix != ".h5":
            # local save only implemented for hdf5 files
            return None

        mean_loc = "/descr/" + loc + "/mean"
        std_loc = "/descr/" + loc + "/std"

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
    def _set_local_descriptive(path, loc, mean, std):

        logging.warning("saving results of descriptive")

        if path.suffix != ".h5" or mean is None or std is None:
            # local save only implemented for hdf5 files
            return False

        mean_loc = "/descr/" + loc + "/mean"
        std_loc = "/descr/" + loc + "/std"

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


class SubFrameGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of preprocessed data from video files for neural network training.

    This class is designed to work with video data stored in .h5 files in (Z, X, Y) format. It supports various
    preprocessing options including cropping, rotation, flipping, adding noise, and normalizing data.
    The class can handle single or multiple data paths, and it is capable of generating batches of a specified input
    size.

    Args:
        paths: Path(s) to .h5 file(s) containing video data.
        batch_size: The size of the data batches.
        input_size: The size of each input frame.
        pre_post_frames: Number of frames before and after the central frame to consider.
        gap_frames: Number of frames to skip before and after each central frame.
        z_steps: The step size in the z-direction.
        z_select: Criteria for selecting a subset of frames.
        allowed_rotation: Allowed rotation angles. Set to [0] to prevent rotation.
        allowed_flip: Allowed flip operations. Set to [-1] to prevent flipping.
        random_offset: If True, applies random offset.
        add_noise: If True, adds noise to the data.
        drop_frame_probability: Probability of dropping a frame.
        max_per_file: Maximum data to consider per file.
        overlap: Overlap between consecutive frames.
        padding: Type of padding to apply.
        shuffle: If True, shuffles the data.
        normalize: Normalization mode.
        loc: Dataset name in the .h5 file.
        output_size: The size of the output data.
        cache_results: If True, caches the results.
        in_memory: If True, keeps data in memory, which can speed up processing but might lead to memory leaks.
        save_global_descriptive: If True, saves global descriptive statistics to the .h5 file preventing computation of the same value on subsequent runs.
        logging_level: The logging level.

    Raises:
        AssertionError: If input conditions related to rotation, padding, or normalization are not met.
        ValueError: If 'random_offset' and 'overlap' are set simultaneously.

    Example::

        # Create a SubFrameGenerator instance
        generator = SubFrameGenerator(paths="path/to/data.h5", loc="data/ch0",
            batch_size=32, input_size=(128, 128), pre_post_frame=5,
            shuffle=True, normalize="global"
        )
    """

    def __init__(
            self, paths: Union[str, List[str], Path, List[Path]], batch_size: int,
            input_size: Tuple[int, int] = (128, 128),
            pre_post_frames: Union[int, Tuple[int, int]] = 5, gap_frames: Union[int, Tuple[int, int]] = 0,
            z_steps: float = 0.1, z_select: Union[None, int, List[int]] = None, allowed_rotation: List[int] = 0,
            allowed_flip: List[int] = -1, random_offset: bool = False, add_noise: bool = False,
            drop_frame_probability: Union[None, float] = None, max_per_file: Union[None, int] = None, overlap: int = 0,
            padding: Union[None, Literal["symmetric", "edge"]] = None, shuffle: bool = True,
            normalize: Union[None, Literal["local", "global"]] = None, loc: str = "data/",
            output_size: Union[None, Tuple[int, int]] = None, cache_results: bool = False, in_memory: bool = False,
            save_global_descriptive: bool = True, logging_level: int = logging.INFO
    ):

        logging.basicConfig(level=logging_level)

        # adjust input_size to be power of two
        input_size = [closest_power_of_two(v) for v in input_size]

        if not isinstance(paths, list):
            paths = [paths]
        self.paths = paths
        self.loc = loc

        logging.debug(f"data files: {self.paths}")
        logging.debug(f"data loc: {self.loc}")

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.save_global_descriptive = save_global_descriptive

        if isinstance(pre_post_frames, int):
            pre_post_frames = (pre_post_frames, pre_post_frames)
        self.signal_frames = pre_post_frames

        if isinstance(gap_frames, int):
            gap_frames = (gap_frames, gap_frames)
        self.gap_frames = gap_frames

        self.z_steps = z_steps
        self.z_select = z_select
        self.max_per_file = max_per_file
        self.stack_len = None

        if isinstance(allowed_rotation, int):
            allowed_rotation = [allowed_rotation]

        if (1 in allowed_rotation) or (3 in allowed_rotation):
            assert input_size[0] == input_size[
                1], (f"when using 90 or 270 degree rotation (allowed rotation: 1 or 3) the 'input_size' needs to be "
                     f"square. However input size is: {input_size}")
        self.allowed_rotation = allowed_rotation

        if isinstance(allowed_flip, int):
            allowed_flip = [allowed_flip]
        self.allowed_flip = allowed_flip

        if random_offset and overlap is not None:
            raise ValueError(f"random_offset and overlap are incompatible. Please choose only one.")

        if isinstance(overlap, int):
            overlap = overlap + overlap % 2

        self.overlap = overlap  # float

        assert padding in [None, "symmetric", "edge"]
        assert not (random_offset and (
                padding is not None)), "cannot use 'padding' and 'random_offset' flag. Please choose one or the other!"
        self.padding = padding

        self.random_offset = random_offset
        self.add_noise = add_noise
        self.drop_frame_probability = drop_frame_probability

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
        self.items = self._generate_items()

        # cache
        self.cache_results = cache_results
        if cache_results:
            logging.warning("using caching may lead to memory leaks. Please set to false if you experience "
                            "Out-Of-Memory errors.")

        self.cache = {}

    def _generate_items(self):

        # define size of each predictive field of view (X, Y)
        dw, dh = self.input_size
        signal_frames = self.signal_frames
        gap_frames = self.gap_frames

        # define prediction length (Z)
        stack_len = signal_frames[0] + gap_frames[0] + 1 + gap_frames[1] + signal_frames[1]
        self.stack_len = stack_len
        z_steps = max(1, int(self.z_steps * stack_len))

        self.fov_size = (stack_len, dw, dh)

        x_start, y_start, z_start = 0, 0, 0
        # randomize input
        if self.random_offset:
            x_start = np.random.randint(0, dw)
            y_start = np.random.randint(0, dh)
            z_start = np.random.randint(0, stack_len)

        # adjust for overlap
        overlap = self.overlap
        if overlap is not None:
            if overlap < 1:
                overlap_x, overlap_y = int(dw * overlap), int(dh * overlap)
            else:
                overlap_x, overlap_y = overlap, overlap

        else:
            overlap_x, overlap_y = 0, 0

            # x_start = -overlap_x  # y_start = -overlap_y

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
                raise NotImplementedError(
                    f"filetype is recognized - please provide .h5, .tif or .tiff instead of: {file}"
                )

            if self.z_select is not None:
                Z0 = max(0, self.z_select[0])
                Z1 = min(Z, self.z_select[1])
            else:
                Z0, Z1 = 0, Z

            # Calculate padding (if applicable)
            if self.padding is not None:
                pad_z0 = signal_frames[0] + gap_frames[0]
                pad_z1 = signal_frames[1] + gap_frames[1] + 1
                pad_x1 = dw % X
                pad_y1 = dh % Y
            else:
                pad_z0 = pad_z1 = pad_x1 = pad_y1 = 0

            zRange = list(
                range(
                    Z0 + z_start - pad_z0, Z1 - stack_len - z_start + pad_z1, z_steps
                )
            )
            xRange = list(
                range(
                    x_start, X - x_start + pad_x1 - dw, dw - overlap_x
                )
            )
            yRange = list(
                range(
                    y_start, Y - y_start + pad_y1 - dh, dh - overlap_y
                )
            )

            logging.debug(f"\nz_range: {zRange}")
            logging.debug(f"\nx_range: {xRange}")
            logging.debug(f"\nx_range param > x_start:{x_start}, X:{X} pad_x1:{pad_x1}, dw:{dw}")
            logging.debug(f"\ny_range: {yRange}")

            if self.shuffle:
                random.shuffle(zRange)
                random.shuffle(xRange)
                random.shuffle(yRange)

            for z0 in zRange:
                z1 = z0 + stack_len

                for x0 in xRange:
                    x1 = x0 + dw

                    for y0 in yRange:
                        y1 = y0 + dh

                        # choose modification
                        rot = random.choice(allowed_rotation)
                        flip = random.choice(allowed_flip)

                        # mark dropped frame (if applicable)
                        if (self.drop_frame_probability is not None) and (
                                np.random.random() <= self.drop_frame_probability):
                            drop_frame = np.random.randint(0, np.sum(signal_frames))
                        else:
                            drop_frame = -1

                        # calculate necessary padding
                        padding = np.zeros(6, dtype=int)

                        padding[0] = min(0, z0)
                        padding[1] = max(0, z1 - Z1)

                        # padding[2] = min(0, -x0)
                        padding[3] = max(0, x1 - X)

                        # padding[4] = min(0, -y0)
                        padding[5] = max(0, y1 - Y)

                        padding = np.abs(padding)

                        # print(f"_padding: {padding}, 2: {min(0, -x0)}, 3: {max(0, x1-X)}, 4: {min(0, -y0)}, 5: {max(0, y1 - Y)},")

                        # cannot pad on empty axis
                        if (padding[0] >= stack_len) or (padding[1] >= stack_len) or (padding[2] >= dw) or (
                                padding[3] >= dh):
                            continue

                        # create item
                        item = {"idx": idx, "path": file, "z0": z0, "z1": z1, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                                "rot": rot, "flip": flip, "Z": Z, "X": X, "Y": Y, "noise": self.add_noise,
                                "drop_frame": drop_frame, "padding": padding}

                        file_container.append(item)

                        idx += 1

            file_container = pd.DataFrame(file_container)

            if len(file_container) < 1:
                raise ValueError("cannot find suitable data chunks.")

            if self.normalize == "global":
                if len(file_container) > 1:

                    local_save = self._get_local_descriptive(file, loc=self.loc)

                    if local_save is None and not self.save_global_descriptive:
                        self.descr[file] = self._bootstrap_descriptive(file_container)

                    elif local_save is None:

                        mean, std = self._bootstrap_descriptive(file_container)
                        self.descr[file] = (mean, std)

                        if self.save_global_descriptive:
                            self._set_local_descriptive(file, loc=self.loc, mean=mean, std=std)

                    else:
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
        items.batch = items.batch.astype(int)  # round down

        self.n = len(items)

        return items

    def on_epoch_end(self):

        # called after each epoch
        if self.shuffle:
            if self.random_offset:
                self.items = self._generate_items()
                self.cache = {}
            else:
                self.items = self.items.sample(frac=1).reset_index(drop=True)

    def _bootstrap_descriptive(
            self, items, frac=0.01, confidence_level=0.75, n_resamples=5000, max_confidence_span=0.2,
            iteration_frac_increase=1.5
    ):

        means = []
        raws = []

        confidence_span = mean_conf = std_conf = max_confidence_span + 1

        def custom_bootstrap(data):

            try:
                res = bootstrap(
                    (np.array(data),), np.median, n_resamples=n_resamples, axis=0, vectorized=False,
                    confidence_level=confidence_level, method="bca"
                )
            except:
                logging.warning("all values are the same. Using np.mean instead of bootstrapping")
                return np.mean(np.array(data)), 0

            # calculate confidence span
            ci = res.confidence_interval
            mean = np.mean((ci.low, ci.high))
            confidence_span = (ci.high - ci.low) / mean

            return mean, confidence_span

        it = 0
        mean_ = None
        std_ = None
        while (frac < 1) and confidence_span > max_confidence_span:

            sel_items = items.sample(frac=frac)
            if len(sel_items) < 1:
                logging.debug(f"items: {items}")
                logging.debug(f"sel items: {sel_items}")
            for _, row in sel_items.iterrows():
                raw = self._load_row(row).flatten()
                raws.append(raw)

                means += [np.nanmedian(raw)]

            mean_, mean_conf = custom_bootstrap(means)

            std_, std_conf = custom_bootstrap([np.std(r - mean_) for r in raws])

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
            std_ = np.nanmean([np.std(r - mean_) for r in raws])

        return mean_, std_

    def _load_row(self, row):

        path, z0, z1, x0, x1, y0, y1, Z, X, Y = row.path, row.z0, row.z1, row.x0, row.x1, row.y0, row.y1, row.Z, row.X, row.Y
        pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1 = row.padding

        if sum([pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1]) > 0:

            # adjust Z boundaries
            if z0 < 0:
                pad_z0 = abs(z0)
                z0 = 0
            else:
                pad_z0 = 0

            if z1 > Z:
                pad_z1 = abs(Z - z1)
                z1 = Z
            else:
                pad_z1 = 0

            # adjust X boundaries
            if x0 < 0:
                pad_x0 = abs(x0)
                x0 = 0
            else:
                pad_x0 = 0

            if x1 > X:
                pad_x1 = abs(X - x1)
                x1 = X
            else:
                pad_x1 = 0

            # adjust Y boundaries
            if y0 < 0:
                pad_y0 = abs(y0)
                y0 = 0
            else:
                pad_y0 = 0

            if y1 > Y:
                pad_y1 = abs(Y - y1)
                y1 = Y
            else:
                pad_y1 = 0

        if isinstance(self.mem_data, dict):

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

        else:
            raise ValueError(f"please provide path as tif or h5 file.")

        if np.sum((pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1)) > 0:
            data = np.pad(
                data, ((pad_z0, pad_z1), (pad_x0, pad_x1), (pad_y0, pad_y1)), mode=self.padding
            )

        return data

    def __getitem__(self, index):

        if index in self.cache.keys():
            return self.cache[index]

        X = []
        y = []
        for _, row in self.items[self.items.batch == index].iterrows():

            data = self._load_row(row)

            assert data.shape == self.fov_size, f"loaded data does not match expected FOV size " \
                                                f"(fov: {self.fov_size}) vs. (load: {data.shape}"

            if row.rot != 0:
                data = np.rollaxis(data, 0, 3)
                data = np.rot90(data, k=row.rot)
                data = np.rollaxis(data, 2, 0)

            if row.flip != -1:
                data = np.flip(data, row.flip)

            if row.noise is not None:
                data = data + np.random.random(data.shape) * row.noise

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

        return X, y

    def __len__(self):
        # return self.n // self.batch_size
        return len(self.items.batch.unique())

    def __get_norm_parameters__(self, index):

        files = self.items[self.items.batch == index].path.unique()

        if len(files) == 1:
            return self.descr[files[0]] if files[0] in self.descr.keys() else (1, 1)
        elif len(files) > 1:
            return {f: self.descr[f] if f in self.descr.keys() else (1, 1) for f in files}
        else:
            return None

    def infer(
            self, model: Union[Model, str, Path], output: Union[str, Path] = None, out_loc: str = None,
            dtype: Union[Literal["same"], np.dtype] = "same", chunk_size: Union[str, Tuple[int, int, int]] = None,
            rescale: bool = True
    ) -> Union[np.ndarray, Path, None]:
        """
        Performs inference on video data using a provided model and generates output in specified format.

        This method applies a deep learning model to the video data to perform tasks such as denoising or segmentation.
        It supports different input and output formats, including .h5 and .tif files. The method also allows for optional
        rescaling of the output and handles data chunking for efficient processing.

        Raises:
            FileNotFoundError: If the model file or directory cannot be found.
            ValueError: If 'random_offset' and 'overlap' are set simultaneously or incompatible arguments are provided.
            AssertionError: If provided 'model' is not of the expected type or if data dimensions mismatch.

        Args:
            model: A Keras model or the path to a model file/directory for inference.
            output: Path to the file where the output will be saved. If None, the output array is returned.
            out_loc: Location within the .h5 file to store the output. Required if output is an .h5 file.
            dtype: Data type of the output. 'same' uses the same dtype as the input data.
            # TODO chunk_size should probably be updated
            chunk_size: Size of chunks for .h5 file output. Can be 'infer', an integer, or None.
            rescale: Whether to rescale the output based on global descriptive statistics.

        Returns:
            Depending on 'output', either a numpy array of the processed data, a Path object pointing to the saved file, or None.

        Example::

            # Assuming a SubFrameGenerator instance 'generator' and a Keras model 'model'
            output_data = generator.infer(model, output="output_path.h5",
                out_loc="inference_results", dtype="float32")
        """
        f = None

        # load model if not provided
        if isinstance(model, (str, pathlib.Path)):
            model = Path(model)

            if os.path.isdir(model):

                models = list(model.glob("*.h*5")) + list(model.glob("*.pth"))

                if len(models) < 1:
                    raise FileNotFoundError(f"cannot find model in provided directory: {model}")

                models.sort(key=lambda x: os.path.getmtime(x))
                model_path = models[0]
                logging.info(f"directory provided. Selected most recent model: {model_path}")

            elif os.path.isfile(model):
                model_path = model

            else:
                raise FileNotFoundError(f"cannot find model: {model}")

            if model_path.suffix in [".h5", ".hdf5", ".H5", ".HDF5"]:
                model = load_model(
                    model_path, custom_objects={"annealed_loss": Network.annealed_loss,
                                                "mean_squareroot_error": Network.mean_squareroot_error}
                )
            else:
                raise ValueError(f"unknown model suffix: {model_path}")

        else:
            logging.warning(f"providing model via parameter. Model type: {type(model)}")

            # deals with keras.__version__ > 2.10.0
            expected_model_type = [UNet]
            try:
                expected_model_type.append(keras.engine.functional.Functional)
            except AttributeError:
                expected_model_type.append(keras.src.engine.functional.Functional)

            assert isinstance(model, tuple(expected_model_type)), f"Please provide keras/pytorch model, " \
                                                                  f"file_path or dir_path instead of {type(model)}"

        # enforce pathlib.Path
        if output is not None:
            output = Path(output)

        # create output arrays
        assert len(
            self.items.path.unique()
        ) < 2, f"inference from multiple files is currently not implemented: {self.items.path.unique()}"
        items = self.items

        if "padding" in items.columns:
            pad_z_max = items.padding.apply(lambda x: x[1]).max()
            pad_x_max = items.padding.apply(lambda x: x[3]).max()
            pad_y_max = items.padding.apply(lambda x: x[5]).max()
        else:
            pad_z_max, pad_x_max, pad_y_max = 0, 0, 0

        output_shape = (items.z1.max() - pad_z_max, items.x1.max() - pad_x_max, items.y1.max() - pad_y_max)

        if dtype == "same":
            x, _ = self[self.items.batch.tolist()[0]]  # raw data
            dtype = x.dtype
            logging.warning(f"choosing dtype: {dtype}")

        if output is not None and output.suffix in (".h5", ".hdf5"):

            assert out_loc is not None, "when exporting results to .h5 file please provide 'out_loc' flag"

            if chunk_size is None:
                io = IO()
                chunk_size = io.infer_chunks(output_shape, dtype, strategy="Z")

            f = h5.File(output, "a")
            rec = f.create_dataset(out_loc, shape=output_shape, chunks=chunk_size, dtype=dtype)
        else:
            rec = np.zeros(output_shape, dtype=dtype)

        # infer frames
        for batch in tqdm(self.items.batch.unique()):

            x, _ = self[batch]  # raw data
            y = model.predict(x)  # denoised data

            if dtype != y.dtype:

                if isinstance(y, torch.Tensor):
                    y = y.numpy()
                    y = np.moveaxis(y, 1, -1)
                    logging.warning(f"y.shape: {y.shape}")

                y = y.astype(dtype)

            x_items = items[items.batch == batch]  # meta data

            assert len(x) == len(
                x_items
            ), f"raw and meta data must have the same length: raw ({len(x)}) vs. meta ({len(x_items)})"

            c = 0
            for _, row in x_items.iterrows():

                im = y[c, :, :, 0]
                im_shape_orig = im.shape

                pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1 = row.padding
                overlap_x_half, overlap_y_half = int(self.overlap / 2), int(self.overlap / 2)

                x0, x0_ = (0, pad_x0) if row.x0 == 0 else (row.x0 + overlap_x_half, overlap_x_half + pad_x0)
                y0, y0_ = (0, pad_y0) if row.y0 == 0 else (row.y0 + overlap_y_half, overlap_y_half + pad_y0)

                x1, x1_ = (row.x1, -pad_x1) if row.x1 >= row.X else (
                    row.x1 - overlap_x_half - pad_x1, -overlap_x_half - pad_x1)
                y1, y1_ = (row.y1, -pad_y1) if row.y1 >= row.Y else (
                    row.y1 - overlap_y_half - pad_y1, -overlap_y_half - pad_y1)

                if x1_ == 0:
                    x1_ = None

                if y1_ == 0:
                    y1_ = None

                im = im[x0_:x1_, y0_:y1_]

                if rescale:
                    mean, std = self.descr[self.items.iloc[0].path]
                    im = (im * std) + mean

                gap = self.signal_frames[0] + self.gap_frames[0]
                rec[row.z0 + gap, x0:x1, y0:y1] = im

                c += 1

        if output is None:
            return rec

        elif output.suffix in (".tiff", ".tif"):
            tiff.imwrite(output, data=rec)
            return output

        elif output.suffix in (".h5", ".hdf5"):
            f.close()
            return output

    @staticmethod
    def _get_local_descriptive(path, loc):

        if path.suffix != ".h5":
            # local save only implemented for hdf5 files
            return None

        mean_loc = "/descr/" + loc + "/mean"
        std_loc = "/descr/" + loc + "/std"

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
    def _set_local_descriptive(path, loc, mean, std):

        logging.warning("saving results of descriptive")

        if path.suffix != ".h5" or mean is None or std is None:
            # local save only implemented for hdf5 files
            return False

        mean_loc = "/descr/" + loc + "/mean"
        std_loc = "/descr/" + loc + "/std"

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


class UNet(nn.Module):
    def __init__(self, item_size: Tuple[int, int, int], kernels: int = 64, kernel_size: int = 3, padding: int = 1,
                 n_stacks: int = 3, batch_normalize: bool = False):
        super(UNet, self).__init__()

        self.n_stacks = n_stacks
        item_z, item_x, item_y = item_size
        kernels = closest_power_of_two(kernels)
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.batch_normalize = batch_normalize

        if batch_normalize:
            logging.warning(f"batch normalization is currently not implemented for UNet.")
            # todo implement
            # self.batch_norm = nn.BatchNorm2d(n_channels) if batch_normalize else None

        self.input_ = nn.Conv2d(item_z, kernels, kernel_size=kernel_size, padding=padding)

        # Encoder layers
        self.encoders = nn.ModuleList()
        in_channels = kernels
        out_channels = kernels
        encoder_outputs = []
        for i in range(n_stacks):
            # Convolution layer
            self.encoders.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            encoder_outputs.append(out_channels)

            # Max pooling layer for downsampling (except for the last stack)
            if i < n_stacks - 1:
                self.encoders.append(nn.MaxPool2d(2))

                # Double the out_channels for the next level in the stack after pooling
                in_channels = out_channels
                out_channels *= 2

        # Decoder layers
        self.decoders = nn.ModuleList()
        in_channels = encoder_outputs[-1]
        out_channels = encoder_outputs[-1] // 2
        last_out_channels = 0
        for i in range(n_stacks):

            if i < n_stacks - 1:

                # Convolution layer in decoder
                self.decoders.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))

                last_out_channels = out_channels

                # Upsampling layer
                self.decoders.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

                # Adjust the in and output_channels considering concatenating
                in_channels = out_channels * 2
                out_channels = encoder_outputs[-i - 2] // 2

            else:

                # Last layer
                in_channels = last_out_channels + encoder_outputs[0]
                self.decoders.append(nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding))

    def forward(self, x):

        x = self.input_(x)

        # Pass input through the encoder layers and store outputs
        encoder_outputs = []
        for i, layer in enumerate(self.encoders):
            x = layer(x)

            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)
            encoder_outputs.append(x)

        # Pass through the decoder layers and concatenate with encoder outputs
        for i in range(0, len(self.decoders)):
            layer = self.decoders[i]
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                # Apply ReLU activation for all but the last convolutional layer
                if i < len(self.decoders) - 1:
                    x = F.relu(x)
            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                # Concatenate - ensure the spatial dimensions match before concatenation
                concat_layer = encoder_outputs[-(i + 2)]
                x = torch.cat((x, concat_layer), dim=1)

        return x

    def predict(self, input_tensor):
        """
        Make a prediction using the U-Net model.

        Args:
            input_tensor (torch.Tensor): The input tensor for prediction.
                                         Shape should match the model's expected input shape.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        # Ensure the model is in evaluation mode
        self.eval()

        # Disable gradient computation for inference
        with torch.no_grad():

            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor)

            if input_tensor.dtype != torch.float32:
                input_tensor = input_tensor.to(torch.float32)

            # Check if the input tensor is on the same device as the model
            if next(self.parameters()).is_cuda:
                input_tensor = input_tensor.to('cuda')
            else:
                input_tensor = input_tensor.to('cpu')

            # Forward pass through the model
            output_tensor = self.forward(input_tensor)

        return output_tensor


class PyTorchNetwork:
    def __init__(
            self, train_dataset: SubFrameDataset, val_dataset: SubFrameDataset = None,
            test_dataset: SubFrameDataset = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4,
            learning_rate: float = 0.001, momentum: float = 0.9, decay_rate: float = 0.1, decay_steps: int = 30,
            n_stacks: int = 3, kernels: int = 64, kernel_size: int = 3, batch_normalize: bool = False,
            loss: PyTorchLoss = "annealed_loss",
            load_model: Union[str, Path] = None, use_cpu: bool = False):

        # define attributes
        self.optimizer = None
        self.iterator = None
        self.n_stacks = n_stacks
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.batch_normalize = batch_normalize
        self.item_size = train_dataset.item_size
        self.input_size = train_dataset.input_size

        # define generators
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)

        if val_dataset is not None:
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers)
        else:
            self.val_dataloader = None

        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
        else:
            self.test_dataloader = None

        # Create fresh UNet
        self.model = UNet(item_size=self.item_size, n_stacks=n_stacks,
                          batch_normalize=batch_normalize,
                          kernels=kernels, kernel_size=kernel_size)

        # load model
        if load_model is not None:
            self._load_model(model_path=load_model)

        # Define loss
        self.loss = loss

        # define decay rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        # choose device and load model on device
        self.device = 'cpu' if use_cpu or not torch.cuda.is_available() else 'cuda'

        self.model = self.model.to(self.device)

    def run(self, num_epochs: int = 25, save_model: Union[str, Path] = None,
            patience: int = None, min_delta: float = 0.005, model_prefix: str = "model",
            show_mode: Literal["auto", "progress", "notebook"] = None, save_checkpoints: bool = False):

        # todo: keep history

        # create directory to save model
        save_model = Path(save_model) if save_model is not None else None
        if save_model is not None and not save_model.is_dir():
            logging.info(f"Created save dir at: {save_model}")
            save_model.mkdir()

        # define loss
        if self.loss == 'annealed_loss':
            criterion = self._annealed_loss
        elif self.loss == 'mean_square_root_error':
            criterion = self._mean_square_root_error
        elif isinstance(self.loss, Callable):
            criterion = self.loss
        else:
            raise ValueError(
                f"Invalid loss, please provide one of 'annealed_loss' or 'mean_square_root_error' "
                f"or torch Loss object")

        # define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.optimizer = optimizer

        # define learning rate decay
        if self.decay_rate is not None:
            from torch.optim.lr_scheduler import StepLR
            lr_scheduler = StepLR(optimizer=optimizer, step_size=self.decay_steps, gamma=self.decay_rate)
        else:
            lr_scheduler = None

        # define early stopping
        if patience is not None:
            early_stopping = EarlyStopper(patience=patience, min_delta=min_delta)
        else:
            early_stopping = None

        # loop over the dataset multiple times
        self.iterator = None
        min_loss = float('inf')
        running_loss = float('inf')
        val_loss = float('inf')
        epoch = 0
        for epoch in range(num_epochs):

            # run one round of training
            running_loss = self._train_one_epoch(self.train_dataloader, optimizer, criterion, lr_scheduler,
                                                 update_weights=True)

            # run one round of validation
            if self.val_dataloader is not None:
                val_loss = self._train_one_epoch(self.val_dataloader, optimizer, criterion, lr_scheduler,
                                                 update_weights=False)
            else:
                val_loss = running_loss

            # user feedback
            self._show_progress(num_epochs=num_epochs, epoch=epoch, show_mode=show_mode, training_loss=running_loss,
                                validation_loss=val_loss if self.val_dataloader is not None else None,
                                patience_counter=early_stopping.counter if early_stopping is not None else None)

            # check early stopping
            if early_stopping is not None and early_stopping(val_loss):
                logging.warning(f"Early stopping, patience exceeded.")
                break

            # implement checkpoint saving
            if val_loss < min_loss and save_checkpoints and save_model is not None:
                checkpoint_path = save_model.joinpath(f"{model_prefix}_epoch_{epoch}_val_loss_{val_loss:.2E}.pth")
                extras = dict(epoch=epoch, training_loss=running_loss, validation_loss=val_loss)
                self.save(checkpoint_path, extras=extras)

        logging.info('Finished Training')

        if self.test_dataloader is not None:
            test_loss = self._train_one_epoch(self.test_dataloader, optimizer, criterion, lr_scheduler,
                                              update_weights=False)
            logging.info(f"Test loss: {test_loss:.4f}")

        if save_model is not None:
            model_path = save_model.joinpath(f"{model_prefix}.pth")
            extras = dict(num_epochs=epoch, training_loss=running_loss, validation_loss=val_loss)
            self.save(model_path, extras=extras)
            logging.info(f"Model saved to {model_path}")

        return self.model

    def retrain_model(self, frozen_epochs: int = 25, unfrozen_epochs: int = 5, patience: int = 3,
                      min_delta: float = 0.005, save_model: Union[str, Path] = None,
                      model_prefix: str = "retrain", show_mode: Literal["auto", "progress", "notebook"] = None):

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the first encoder layer
        for param in self.model.encoders[0].parameters():
            param.requires_grad = True

        # Unfreeze the last decoder layer
        for param in self.model.decoders[-1].parameters():
            param.requires_grad = True

        # Run frozen epochs
        self.run(num_epochs=frozen_epochs, patience=None, show_mode=show_mode)

        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

        # Run unfrozen epochs
        self.run(num_epochs=unfrozen_epochs, patience=patience, min_delta=min_delta, save_model=save_model,
                 model_prefix=model_prefix, show_mode=show_mode)

    def infer(self, dataset: SubFrameDataset, output: Union[str, Path] = None,
              out_loc: str = None, batch_size: int = 16, num_workers: int = 4,
              dtype: Union[Literal["same"], np.dtype] = "same", chunk_size: Union[str, Tuple[int, int, int]] = None,
              rescale: bool = True
              ) -> Union[np.ndarray, Path, None]:
        """
        Performs inference on video data using a provided model and generates output in specified format.

        This method applies a deep learning model to the video data to perform tasks such as denoising or segmentation.
        It supports different input and output formats, including .h5 and .tif files. The method also allows for optional
        rescaling of the output and handles data chunking for efficient processing.

        Raises:
            FileNotFoundError: If the model file or directory cannot be found.
            ValueError: If 'random_offset' and 'overlap' are set simultaneously or incompatible arguments are provided.
            AssertionError: If provided 'model' is not of the expected type or if data dimensions mismatch.

        Args:
            output: Path to the file where the output will be saved. If None, the output array is returned.
            out_loc: Location within the .h5 file to store the output. Required if output is an .h5 file.
            dtype: Data type of the output. 'same' uses the same dtype as the input data.
            chunk_size: Size of chunks for .h5 file output. Automatically chosen if not provided.
            rescale: Whether to rescale the output based on global descriptive statistics.

        Returns:
            Depending on 'output', either a numpy array of the processed data, a Path object pointing to the saved file, or None.

        Example::

            # Assuming a SubFrameGenerator instance 'generator' and a Keras model 'model'
            output_data = generator.infer(model, output="output_path.h5",
                out_loc="inference_results", dtype="float32")
        """

        f = None  # h5 file reference

        # ensure inference on single file
        if len(dataset.items.path.unique()) > 1:
            raise ValueError(
                f"inference from multiple files is currently not implemented: {dataset.items.path.unique()}")

        items = dataset.items

        # get padding values
        if "padding" in items.columns:
            pad_z_max = items.padding.apply(lambda x_: x_[1]).max()
            pad_x_max = items.padding.apply(lambda x_: x_[3]).max()
            pad_y_max = items.padding.apply(lambda x_: x_[5]).max()
        else:
            pad_z_max, pad_x_max, pad_y_max = 0, 0, 0

        output_shape = (items.z1.max() - pad_z_max, items.x1.max() - pad_x_max, items.y1.max() - pad_y_max)

        if dtype == "same":
            x, _ = dataset[0]
            dtype = x.dtype
            logging.warning(f"choosing dtype: {dtype}")

        if output is not None:
            output = Path(output)

        if output is not None and output.suffix in (".h5", ".hdf5"):

            if out_loc is None:
                raise ValueError(f"when exporting results to .h5 file please provide 'out_loc' flag")

            if chunk_size is None:
                io = IO()
                chunk_size = io.infer_chunks(output_shape, dtype, strategy="Z")

            f = h5.File(output, "a")
            rec = f.create_dataset(out_loc, shape=output_shape, chunks=chunk_size, dtype=dtype)
        else:
            rec = np.zeros(output_shape, dtype=dtype)

        # infer frames
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for i, (inputs, _) in enumerate(dataloader):

            y = self.model.predict(inputs)  # denoised data

            if isinstance(y, torch.Tensor):
                y = y.numpy()

            if y.dtype != dtype:
                y = y.astype(dtype)

            idx_start, idx_stop = int(i * batch_size), int(i * batch_size + batch_size)
            rows = dataset.items.iloc[idx_start:idx_stop]

            if len(rows) != len(y):
                raise ValueError(
                    f"loaded metadata ({len(rows)} must have the same length as denoised frames ({len(y)})")

            c = 0
            for _, row in rows.iterrows():

                im = y[c, 0, :, :]
                im_shape_orig = im.shape

                pad_z0, pad_z1, pad_x0, pad_x1, pad_y0, pad_y1 = row.padding
                overlap_x_half, overlap_y_half = int(dataset.overlap_x / 2), int(dataset.overlap_y / 2)

                x0, x0_ = (0, pad_x0) if row.x0 == 0 else (row.x0 + overlap_x_half, overlap_x_half + pad_x0)
                y0, y0_ = (0, pad_y0) if row.y0 == 0 else (row.y0 + overlap_y_half, overlap_y_half + pad_y0)

                x1, x1_ = (row.x1, -pad_x1) if row.x1 >= row.X else (
                    row.x1 - overlap_x_half - pad_x1, -overlap_x_half - pad_x1)
                y1, y1_ = (row.y1, -pad_y1) if row.y1 >= row.Y else (
                    row.y1 - overlap_y_half - pad_y1, -overlap_y_half - pad_y1)

                if x1_ == 0:
                    x1_ = None

                if y1_ == 0:
                    y1_ = None

                im = im[x0_:x1_, y0_:y1_]

                if rescale:
                    mean, std = dataset.descr[dataset.items.iloc[0].path]
                    im = (im * std) + mean

                gap = dataset.signal_frames[0] + dataset.gap_frames[0]
                rec[row.z0 + gap, x0:x1, y0:y1] = im

                c += 1

        if output is None:
            return rec

        elif output.suffix in (".tiff", ".tif"):
            tiff.imwrite(output, data=rec)
            return output

        elif output.suffix in (".h5", ".hdf5"):
            f.close()
            return output

    def save(self, path, extras: dict = None):

        model_dict = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'item_size': self.item_size,
                      'n_stacks': self.n_stacks, 'kernels': self.kernels, 'kernel_size': self.kernel_size,
                      'batch_normalize': self.batch_normalize, 'input_size': self.input_size}

        if extras is not None:
            model_dict.update(extras)

        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
            logging.warning(f"changed path suffix to {path}")

        torch.save(model_dict, path)
        logging.info(f"saved model to {path}")

    def _load_model(self, model_path):

        model_path = Path(model_path)

        if model_path.is_dir():

            model_files = list(model_path.glob("*.pth"))

            if len(model_files) == 0:
                raise FileNotFoundError(f"No model files found in directory with '.pth' extension: {model_path}")

            model_files.sort(key=lambda x: os.path.getmtime(x))
            model_path = model_files[0]

            logging.info(f"directory provided. Selected most recent model: {model_path}")

        load_dict = torch.load(model_path.as_posix())
        n_stacks = load_dict['n_stacks']
        batch_normalize = load_dict['batch_normalize']
        kernels = load_dict['kernels']
        kernel_size = load_dict['kernel_size']
        item_size = load_dict['item_size']
        input_size = load_dict['input_size']

        if self.item_size != item_size:
            raise ValueError(f"loaded module expects a stack length of {item_size}, got {self.item_size}."
                             f"Stack length is defined by the pre and post frames during dataset initiation.")

        if self.input_size != input_size:
            raise ValueError(f"loaded module expects an input_size of {input_size}, got {self.input_size}."
                             f"Input size is defined during dataset initiation.")

        # Load the saved model weights
        if 'model_state_dict' in load_dict:
            self.model.load_state_dict(load_dict['model_state_dict'])
        else:
            raise KeyError("The loaded dictionary does not have a 'model_state_dict' key.")

        # update the class attributes
        self.n_stacks = n_stacks
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.batch_normalize = batch_normalize
        self.item_size = item_size
        self.input_size = input_size

    def _train_one_epoch(self, data_generator: DataLoader, optimizer, criterion, lr_scheduler=None,
                         update_weights: bool = False):

        # switch to training/evaluation mode and toggle gradient calculations
        if update_weights:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_generator):

            # convert to PyTorch tensors
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)

            if inputs.dtype != torch.float32:
                inputs = inputs.to(torch.float32)

            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            if labels.dtype != torch.float32:
                labels = labels.to(torch.float32)

            # move data to appropriate device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            if update_weights:
                # Backward pass and optimization only during training
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

            # Accumulate loss
            running_loss += loss.item()

        # Enable gradient calculations again, if needed later
        torch.set_grad_enabled(True)

        return running_loss

    def _show_progress(self, num_epochs: int, epoch: int, training_loss: float,
                       show_mode: Literal["auto", "progress", "notebook"] = None,
                       validation_loss: float = None,
                       patience_counter: int = None):

        if show_mode is None:
            return None

        # define iterator if it is None
        if self.iterator is None:

            if show_mode == "progress":
                self.iterator = tqdm(range(num_epochs), total=num_epochs)
            elif show_mode == "notebook":
                from IPython.display import clear_output
                from IPython.core.display_functions import display
                self.iterator = range(num_epochs)
            else:
                self.iterator = range(num_epochs)

        # define which show mode to execute
        if show_mode == "auto":
            if self._is_notebook():
                show_mode = "notebook"
            else:
                show_mode = "progress"

        # progress
        if show_mode == "progress":

            if isinstance(training_loss, (list, tuple)):
                training_loss = training_loss[-1]

            progress_description = f"train>{training_loss:.2E} "

            if validation_loss is not None:

                if isinstance(validation_loss, (list, tuple)):
                    validation_loss = validation_loss[-1]

                progress_description += f"val>{validation_loss:.2E} "

            if patience_counter is not None:
                progress_description += f"patience>{patience_counter}"

            self.iterator.set_description(progress_description)
            self.iterator.update()

        elif show_mode == "notebook":

            from IPython.core.display_functions import clear_output, display
            import matplotlib.pyplot as plt

            plt.clf()
            clear_output(wait=True)

            fig, axx = plt.subplots(1, 2, figsize=(9, 4))

            axx[0].plot(training_loss, color="black", label="training")
            if validation_loss is not None:
                axx[0].plot(np.array(validation_loss).flatten(), color="green", label="validation")

            axx[0].set_title(f"losses")
            axx[0].set_yscale("log")
            axx[0].legend()

            # set title
            title = f"Epoch {epoch}/{num_epochs}"
            if patience_counter is not None:
                title += f"P{patience_counter}"

            plt.tight_layout()

            display(fig)

        else:
            raise ValueError(f"please choose a show_mode from ['auto', 'progress', 'notebook']")

    @staticmethod
    def _is_notebook() -> bool:
        """
        Returns whether we are running in a Jupyter notebook.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type, probably standard Python interpreter
        except NameError:
            return False  # Probably standard Python interpreter

    @staticmethod
    def _annealed_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the annealed loss between the true and predicted values.

        Args:
            y_true: The true values.
            y_pred: The predicted values.

        Returns:
            torch.Tensor: The calculated annealed loss
        """
        local_power = 4
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Compute the element-wise absolute difference and apply local power
        final_loss = torch.pow(
            torch.abs(y_pred - y_true) + 1e-8, local_power
        )

        # Compute the mean of the final loss
        return torch.mean(final_loss)

    @staticmethod
    def _mean_square_root_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean square root error between the true and predicted values.

        Args:
            y_true: The true values.
            y_pred: The predicted values.

        Returns:
            torch.Tensor: The calculated mean square root error.
        """
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Compute the element-wise absolute difference, apply square root and compute mean
        return torch.mean(torch.sqrt(torch.abs(y_pred - y_true) + 1e-8))


class Network:
    """
        A neural network class designed for image processing tasks, typically utilizing U-Net architecture.

        This class facilitates the creation, training, and evaluation of a U-Net based neural network model.
        It is equipped to handle training with custom generators, various configurations for the model architecture,
        and supports multiple loss functions. The class is designed to be flexible and adaptable for a wide range of
        image processing tasks.

        Args:
            train_generator: A :class:`~astrocast.denoising.SubFrameGenerator` object for training data.
            val_generator: A :class:`~astrocast.denoising.SubFrameGenerator` object for validation data, used for evaluating model performance during training.
            learning_rate: Initial learning rate for the optimizer.
            decay_rate: Decay rate for learning rate reduction over training epochs.
            decay_steps: Number of steps after which the learning rate decays.
            n_stacks: Number of stacks (or depth) in the U-Net model.
            kernel: The number of filters in the initial convolution layer of the U-Net model.
            batchNormalize: Flag to enable or disable batch normalization in the model.
            loss: The loss function used for model training. Supports custom loss functions.
            pretrained_weights: Path to the pretrained weights for model initialization.
            use_cpu: Flag to enforce training on CPU, useful in GPU-constrained environments.

        Raises:
            FileNotFoundError: If the provided model path does not exist or is invalid.
            ValueError: If incompatible arguments are provided.
            AssertionError: For invalid input conditions related to the model configuration.

        Example::

            from astrocast.denoising import SubFrameGenerator, Network

            # Creating an instance of the Network class
            train_gen = SubFrameGenerator("/path/to/train/data")
            val_gen = SubFrameGenerator("/path/to/val/data")
            net = Network(train_gen, val_gen, learning_rate=0.001, n_stacks=3, kernel=64)
        """

    def __init__(
            self, train_generator: SubFrameGenerator, val_generator: SubFrameGenerator = None,
            learning_rate: float = 0.001, decay_rate: float = 0.99, decay_steps: int = 250, n_stacks: int = 3,
            kernel: int = 64, batchNormalize: bool = False,
            loss: Union[Literal['annealed_loss', 'mean_squareroot_error'], tf.keras.losses.Loss] = "annealed_loss",
            pretrained_weights: Union[str, Path] = None, use_cpu: bool = False
    ):

        if use_cpu:
            # Set the visible GPU devices to an empty list to use CPU
            tf.config.set_visible_devices([], 'GPU')

        # Assign the train and validation generators
        self.train_gen = train_generator
        self.val_gen = val_generator

        # Create the U-Net model
        self.n_stacks = n_stacks
        self.kernel = kernel
        self.model = self._create_unet(n_stacks=n_stacks, kernel=kernel, batchNormalize=batchNormalize)

        if decay_rate is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
            )
        else:
            lr_schedule = learning_rate

        if pretrained_weights is not None:

            if isinstance(pretrained_weights, str):
                pretrained_weights = Path(pretrained_weights)

            if pretrained_weights.is_file():
                self.model = load_model(
                    pretrained_weights, custom_objects={"annealed_loss": Network.annealed_loss,
                                                        "mean_squareroot_error": Network.mean_squareroot_error}
                )

            elif pretrained_weights.is_dir():

                latest_weights = tf.train.latest_checkpoint(pretrained_weights)

                if latest_weights is not None:
                    logging.info(f"Loading previous weights from: {pretrained_weights}")
                    self.model.load_weights(latest_weights)
                else:
                    logging.warning(f"Couldn't find pretrained weights in {pretrained_weights}")

            else:
                logging.warning(f"pretrained_weights is neither file nor dir: {pretrained_weights}")

        # Set the optimizer and compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule), loss=self.annealed_loss if loss == 'annealed_loss' else loss
        )

    def run(
            self, batch_size: int = 10, num_epochs: int = 25, save_model: Union[str, Path] = None, patience: int = 3,
            min_delta: float = 0.005, monitor: str = "val_loss", model_prefix: str = "model", verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Trains the neural network model using the provided data generators and specified parameters.

        This method facilitates the training of the model with features like early stopping, model checkpointing, and verbose output control. It is designed to offer flexibility in training configuration, allowing for customization of batch size, number of epochs, and other key training parameters. The method is well-suited for training deep learning models in tasks that require iterative learning and model evaluation.

        Args:
            batch_size: Number of samples per gradient update.
            num_epochs: Number of epochs to train the model.
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            monitor: Quantity to be monitored during training.
            save_model: Directory to save the model and checkpoints.
            model_prefix: Prefix for naming saved model files.
            verbose: Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch).

        Returns:
            A History object containing the training history metrics.

        Example::

            # Assuming an instance 'net' of the Network class
            history = net.run(batch_size=32, num_epochs=100, save_model='./model_save', verbose=1)
            print(history.history)
        """

        save_model = Path(save_model) if save_model is not None else None
        if save_model is not None and not save_model.is_dir():
            logging.info("Created save dir at: %s", save_model)
            save_model.mkdir()

        callbacks = []
        if patience is not None:
            callbacks.append(EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, verbose=verbose))

        if save_model is not None:

            if isinstance(save_model, str):
                save_model = Path(save_model)

            if not save_model.is_dir():
                save_model.mkdir()

            callbacks.append(
                ModelCheckpoint(
                    filepath=save_model.as_posix(), save_weights_only=False, monitor=monitor, mode='min',
                    save_best_only=True, )
            )

        # Start model training
        history = self.model.fit(
            self.train_gen, batch_size=batch_size, validation_data=self.val_gen, epochs=num_epochs, callbacks=callbacks,
            shuffle=False, verbose=verbose,  # Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch)
        )

        # Save the final model
        if save_model is not None:
            # Create a filename with parameters
            save_path = save_model.joinpath(f"{model_prefix}.h5").as_posix()
            logging.info(f"saved model to: {save_path}")
            self.model.save(save_path)

        return history

    def retrain_model(
            self, frozen_epochs: int = 25, unfrozen_epochs: int = 5, batch_size: int = 10, patience: int = 3,
            min_delta: float = 0.005, monitor: str = "val_loss", save_model: Union[str, Path] = None,
            model_prefix: str = "retrain", verbose: int = 1
    ):
        """
        Retrains the model with a new dataset, employing a two-phase training process with frozen and unfrozen layers.

        In the first phase, the model is trained with its internal layers frozen, allowing only the final layers to
        adjust. In the second phase, all layers are unfrozen for additional training. This method is particularly
        useful when adapting a pre-trained model to new data, leveraging transfer learning.

        Args:
            frozen_epochs: Number of epochs to train with frozen layers.
            unfrozen_epochs: Number of epochs to train after unfreezing all layers.
            batch_size: Number of samples per gradient update.
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            monitor: Quantity to be monitored during training.
            save_model: Directory to save the retrained model and checkpoints.
            model_prefix: Prefix for naming saved model files.
            verbose: Verbosity mode (0 - silent, 1 - progress bar, 2 - one line per epoch).

        Example::

            # Assuming an instance 'net' of the Network class
            net.retrain_model(frozen_epochs=20, unfrozen_epochs=10, batch_size=32, save_model='./retrain_model_save')
        """

        model = self.model

        # set layers other than input and output not trainable
        for layer in model.layers[1:-1]:
            layer.trainable = False

        logging.info(model.summary(line_length=100))

        _ = self.run(
            num_epochs=frozen_epochs, batch_size=batch_size, patience=patience, min_delta=min_delta, monitor=monitor,
            save_model=save_model, model_prefix=model_prefix, verbose=verbose
        )

        if unfrozen_epochs is not None:
            for layer in model.layers:
                layer.trainable = True

            logging.info(model.summary(line_length=100))

            history_frozen = self.run(
                num_epochs=unfrozen_epochs, batch_size=batch_size, patience=patience, min_delta=min_delta,
                monitor=monitor, save_model=save_model, model_prefix=model_prefix, verbose=verbose
            )

    def _create_unet(self, n_stacks=3, kernel=64, batchNormalize=False, verbose=1):
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

            if i != n_stacks - 1:
                # Max pooling layer in the encoder
                pool = MaxPooling2D(pool_size=(2, 2))(conv)

                kernel = kernel * 2
                last_layer = pool
            else:
                # Last convolutional layer in the encoder
                last_layer = conv

        # Decoder
        decoder = None
        for i in range(n_stacks):
            if i != n_stacks - 1:
                # Convolutional layer in the decoder
                conv = Conv2D(kernel, (3, 3), activation="relu", padding="same")(last_layer)
                up = UpSampling2D((2, 2))(conv)
                logging.warning(f"tensorflow i: {-(i + 2)}")
                conc = Concatenate()([up, enc_conv[-(i + 2)]])

                last_layer = conc
                kernel = kernel / 2
            else:
                # Last convolutional layer in the decoder
                decoded = Conv2D(1, (3, 3), activation=None, padding="same")(last_layer)
                decoder = Model(input_window, decoded)

        if verbose > 0 and decoder is not None:
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
        final_loss = K.pow(
            K.abs(y_pred - y_true) + 0.00000001, local_power
        )  # Compute the element-wise absolute difference and apply local power
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
        return K.mean(
            K.sqrt(K.abs(y_pred - y_true) + 0.00000001), axis=-1
        )  # Compute the element-wise absolute difference, apply square root, and compute mean along the last axis
