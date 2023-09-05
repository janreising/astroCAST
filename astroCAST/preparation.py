import itertools
import logging
import multiprocessing
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import czifile
import dask
import h5py
import napari_plot
import numpy as np
import pandas as pd
import psutil
import tifffile
import tiledb
import dask.array as da
import dask_image.imread
from dask.distributed import Client, LocalCluster
from napari_plot._qt.qt_viewer import QtViewer
from napari.utils.events import Event
from scipy import signal
from skimage.transform import resize
from skimage.util import img_as_uint
from deprecated import deprecated
from scipy.ndimage import minimum_filter1d

from astroCAST.helper import get_data_dimensions
from dask.diagnostics import ProgressBar


class Input:

    """Class for loading input data, performing data processing, and saving the processed data."""

    def __init__(self, logging_level=logging.INFO):
        logging.basicConfig(level=logging_level)

    def run(self, input_path, output_path=None,
            sep="_", channels=1, z_slice=None, lazy=True,
            subtract_background=None, subtract_func="mean",
            rescale=None, dtype=np.uint,
            in_memory=False, h5_loc="data", chunks=None, compression=None):

        """ Loads input data from a specified path, performs data processing, and optionally saves the processed data.

        Args:
            input_path (str or pathlib.Path): Path to the input file or directory.
            output_path (str or pathlib.Path, optional): Path to save the processed data. If None, the processed data is returned. (default: None)
            sep (str, optional): Separator used for sorting file names. (default: "_")
            channels (int or dict, optional): Number of channels or dictionary specifying channel names. (default: 1)
            subtract_background (np.ndarray, str, or callable, optional): Background to subtract or channel name to use as background. (default: None)
            subtract_func (str or callable, optional): Function to use for background subtraction. (default: "mean")
            rescale (float, int, or tuple, optional): Scale factor or tuple specifying the new dimensions. (default: None)
            dtype (numpy.dtype, optional): Data type to convert the processed data. (default: np.uint)
            in_memory (bool, optional): If True, the processed data is loaded into memory. (default: False)
            prefix (str, optional): Prefix to use when saving the processed data. (default: "data")
            chunks (tuple or int, optional): Chunk size to use when saving to HDF5 or TileDB. (default: None)
            compression (str or int, optional): Compression method to use when saving to HDF5 or TileDB. (default: None)

        Returns:
            numpy.ndarray or dict: Processed data if output_path is None, otherwise None.

        """

        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        assert isinstance(input_path, Path), "please provide 'input_path' as str or input_pathlib.input_path"
        assert input_path.is_file() or input_path.is_dir(), f"cannot find input: {input_path}"

        logging.info("loading data ...")
        io = IO()
        data = io.load(input_path, sep=sep, z_slice=z_slice, lazy=lazy, chunks=(1, -1, -1))

        logging.info("preparing data ...")
        data = self.prepare_data(data, channels=channels, subtract_background=subtract_background,
                                 subtract_func=subtract_func, rescale=rescale, dtype=dtype, in_memory=in_memory)

        logging.debug(f"data type: {type(data[list(data.keys())[0]])}")

        # rechunk
        if chunks is not None:
            for k in data:
                if data[k].chunksize != chunks:
                    data[k] = da.rechunk(data[k], chunks=chunks)

        # return result
        if output_path is None:
            return data

        logging.info("saving data ...")
        io.save(output_path, data, h5_loc=h5_loc, chunks=chunks, compression=compression)

    @staticmethod
    def subtract_background(data, channels, subtract_background, subtract_func):

        """
        Subtract the background from the data.

        Args:
            data (dict): A dictionary mapping channel names to data arrays.
            channels (dict): A dictionary mapping channel indices to names.
            subtract_background (np.ndarray or str or callable): The background image to subtract or a string specifying
                the channel name to use as the background, or a callable function for background reduction.
            subtract_func (str or callable): The reduction function to use for background subtraction if
                `subtract_background` is a string or a callable function.

        Returns:
            dict: A dictionary mapping channel names to the data arrays after subtracting the background.

        Raises:
            ValueError: If the type of `subtract_background` is not np.ndarray, str, or callable.
            ValueError: If the shape of the subtracted background is not compatible with the data arrays.
            ValueError: If the specified background channel is not found or there are multiple channels with the same name.
            ValueError: If the reduction function is not found or not callable.
            ValueError: If the shape of the reduced background is not compatible with the data arrays.
        """

        if isinstance(subtract_background, np.ndarray):
            # Check if the shape of the subtracted background is compatible with the data arrays
            img_0 = list(data.values())[0][0, :, :]
            if subtract_background.shape != img_0.shape:
                raise ValueError(f"please provide background as np.ndarray of shape {img_0.shape}")

            # Subtract the background from each channel
            for key in data.keys():
                data[key] = data[key] - subtract_background

        elif isinstance(subtract_background, str) or callable(subtract_background):
            # Select the background channel and delete it from the data dictionary
            background_keys = [k for k in channels.keys() if channels[k] == subtract_background]

            if len(background_keys) != 1:
                raise ValueError(f"cannot find channel to subtract or found too many. Choose only one of : {list(channels.values())}.")

            background = data[background_keys[0]]
            for k in background_keys:
                del data[k]

            # Reduce the background dimension using the specified reduction function
            if callable(subtract_func):
                reducer = subtract_func
            else:
                func_reduction = {"mean": da.mean, "std": da.std, "min": da.min, "max": da.max}
                assert subtract_func in func_reduction.keys(), \
                    f"cannot find reduction function. Please provide callable function or one of {func_reduction.keys()}"
                reducer = func_reduction[subtract_func]

            background = reducer(background, axis=0)

            # Check if the shape of the reduced background is compatible with the data arrays
            img_0 = list(data.values())[0][0, :, :]
            if background.shape != img_0.shape:
                raise ValueError(f"incorrect dimension after reduction: data.shape {img_0.shape} vs. reduced.shape {background.shape}")

            # Subtract the reduced background from each channel
            for k in data.keys():
                data[k] = data[k] - background

        else:
            raise ValueError("Please provide 'subtract_background' flag with one of: np.ndarray, callable function or str")

        return data

    @staticmethod
    def rescale_data(data, rescale):
        """
        Rescale the data arrays to a new size.

        Args:
            data (dict): A dictionary mapping channel names to data arrays.
            rescale (tuple, list, int, float): The rescaling factor or factors to apply to the data arrays.
                If a tuple or list, it should contain two elements representing the scaling factors for the X and Y axes.
                If an int or float, the same scaling factor will be applied to both axes.
                If given an int, it will assume that this is the requested final size.
                If given a float, it will multiply the current size by that value.

        Returns:
            dict: A dictionary mapping channel names to the rescaled data arrays.

        Raises:
            ValueError: If the rescale type is mixed (e.g., int and float) or not one of tuple, list, int, or float.
            ValueError: If the length of the rescale tuple or list is not 2.
            TypeError: If the rescale type is not tuple, list, int, or float.
        """

        # Get the original size
        X, Y = list(data.values())[0][0, :, :].shape

        # Convert numbers to tuple (same factor for X and Y)
        if isinstance(rescale, (int, float)):
            rescale = (rescale, rescale)

        # validate rescale value
        if type(rescale[0]) != type(rescale[1]):
            raise ValueError(f"mixed rescale type not allowed for 'rescale' flag: {type(rescale[0])} vs {type(rescale[1])}")
        elif len(rescale) != 2:
            raise ValueError("please provide 'rescale' flag as 2D tuple, list or number")

        # Calculate the new size
        if isinstance(rescale[0], int):
            rX, rY = rescale[0], rescale[1]
        elif isinstance(rescale[0], float):
            rX, rY = (int(X * rescale[0]), int(Y * rescale[1]))
        else:
            raise TypeError("'rescale' flag should be of type tuple, list, int or float")

        # Apply resizing to each channel
        for k in data.keys():
            # Rescale the data array using the specified scaling factors and anti-aliasing
            data[k] = data[k].map_blocks(
                lambda chunk: resize(chunk, (chunk.shape[0], rX, rY), anti_aliasing=True),
                chunks=(1, rX, rY))

        return data

    def prepare_data(self, data, channels=1,
                     subtract_background=None, subtract_func="mean",
                     rescale=None, dtype=np.uint,
                     in_memory=False):

        """Prepares the input data by applying various processing steps.

        Args:
            data (numpy.ndarray or dask.array.Array): Input data to be prepared. Should be a 3D array.
            channels (int or dict, optional): Number of channels or dictionary specifying channel names. (default: 1)
            subtract_background (numpy.ndarray, str, or callable, optional): Background to subtract or channel name to use as background. (default: None)
            subtract_func (str or callable, optional): Function to use for background subtraction. (default: "mean")
            rescale (float, int, or tuple, optional): Scale factor or tuple specifying the new dimensions. (default: None)
            dtype (numpy.dtype, optional): Data type to convert the processed data. (default: np.uint)
            in_memory (bool, optional): If True, the processed data is loaded into memory. (default: False)

        Returns:
            dict: A dictionary mapping channel names to the processed data arrays.

        Raises:
            TypeError: If the input data type is not numpy.ndarray or dask.array.Array.
            NotImplementedError: If the input data dimensions are not equal to 3.
            ValueError: If the channels input is not of type int or dict.
            ValueError: If the number of channels does not divide the number of frames evenly.
        """
        with ProgressBar(minimum=10, dt=1):

            # Convert data to a dask array if it's a ndarray, otherwise validate the input type
            stack = da.from_array(data, chunks=(1, -1, -1)) if isinstance(data, np.ndarray) else data
            if not isinstance(stack, (da.Array, da.core.Array)):
                raise TypeError(f"Please provide data as np.ndarray or dask.array.Array instead of {type(data)}")

            # Check if the data has the correct dimensions
            if len(stack.shape) != 3:
                raise NotImplementedError(f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D")

            # Validate the channels input and determine the number of channels
            if not isinstance(channels, (int, dict)):
                raise ValueError(f"please provide channels as int or dictionary.")

            num_channels = channels if isinstance(channels, int) else len(len(channels.keys()))

            if stack.shape[0] % num_channels != 0:
                logging.warning(f"cannot divide frames into channel number: {stack.shape[0]} % {num_channels} != 0. May lead to unexpacted behavior")

            channels = channels if isinstance(channels, dict) else {i: f"ch{i}" for i in range(num_channels)}

            # Split the data into channels based on the given channel indices or names
            prep_data = {}
            for channel_key in channels.keys():
                prep_data[channel_key] = stack[channel_key::num_channels, :, :]

            # Subtract background if specified
            if subtract_background is not None:
                prep_data = self.subtract_background(prep_data, channels, subtract_background, subtract_func)

            # Rescale the prep_data if specified
            if (rescale is not None) and rescale != 1 and rescale != 1.0:
                self.rescale_data(prep_data, rescale)

            # Convert the prep_data type if specified
            if dtype is not None:
                prep_data = self.convert_dtype(prep_data, dtype=dtype)

            # Load the prep_data into memory if requested
            prep_data = dask.compute(prep_data)[0] if in_memory else prep_data

            # Rename the channels in the output dictionary
            return {channels[i]: prep_data[i] for i in prep_data.keys()}

    def convert_dtype(self, data, dtype):

        if dtype == np.uint:
            def func(chunk):
                return img_as_uint(chunk)

        else:
            def func(chunk):
                return chunk.astype(dtype)

        for k in data.keys():
            data[k] = data[k].map_blocks(lambda chunk: func(chunk), dtype=dtype)

        return data


    def save(self, path, data, h5_loc=None, chunks=None, compression=None):

        """Save the processed data to a specified path.

        Args:
            path (str or pathlib.Path): Path to save the processed data.
            data (numpy.ndarray or dict): Processed data to be saved.
            prefix (str, optional): Prefix to use when saving the processed data to HDF5. (default: None)
            chunks (tuple or int, optional): Chunk size to use when saving to HDF5 or TileDB. (default: None)
            compression (str or int, optional): Compression method to use when saving to HDF5 or TileDB. (default: None)
        """

        io = IO()
        io.save(path=path, data=data, h5_loc=h5_loc, chunks=chunks, compression=compression)

class IO:

    def load(self, path, h5_loc=None, sep="_", z_slice=None, lazy=False, chunks="auto"):

        """
        Loads data from a specified file or directory.

        Args:
            path (str or pathlib.Path): The path to the file or directory.
            h5_loc (str): The location of the dataset in an HDF5 file (default: None).
            sep (str): Separator used for sorting file names (default: "_").

        Returns:
            numpy.ndarray or dask.array.core.Array: The loaded data.

        Raises:
            ValueError: If the file format is not recognized.
            FileNotFoundError: If the specified file or folder cannot be found.

        """

        if isinstance(path, (str, Path)):
            path = Path(path)

            if path.suffix in [".tdb"]:
                data =  self._load_tdb(path, lazy=lazy, chunks=chunks)  # Call private method to load TDB file

            elif path.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]:
                data =  self._load_tiff(path, sep, lazy=lazy)  # Call private method to load TIFF file

            elif path.suffix in [".czi", ".CZI"]:
                data =  self._load_czi(path, lazy=lazy)  # Call private method to load CZI file

            elif path.suffix in [".h5", ".hdf5", ".H5", ".HDF5"]:
                data =  self._load_h5(path, h5_loc=h5_loc, lazy=lazy, chunks=chunks)  # Call private method to load HDF5 file

            elif path.suffix in [".npy", ".NPY"]:
                data =  self._load_npy(path, lazy=lazy, chunks=chunks)

            elif path.is_dir():

                # If the path is a directory, load multiple TIFF files
                files = [f for f in path.glob("*") if f.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]]
                if len(files) < 1:
                    raise FileNotFoundError("couldn't find files in folder. Recognized ext: [.tif, .tiff, .TIF, .TIFF]")

                else:
                    data =  self._load_tiff(path, sep, lazy=lazy)  # Call private method to load TIFF files from directory

            else:
                raise ValueError("unrecognized file format! Choose one of [.tiff, .h5, .tdb, .czi]")

        elif isinstance(path, np.ndarray):
            data = da.from_array(path, chunks=chunks)

        elif isinstance(path, da.Array):
            data = da.rechunk(path, chunks=chunks)

        if z_slice is not None:

            if not isinstance(z_slice, (tuple, list)) or len(z_slice) != 2:
                raise ValueError("please provide z_slice as tuple or list of (z_start, z_end)")

            # todo would be better not to load all the data first
            z0, z1 = z_slice
            data = data[z0:z1, :, :]

        return data

    def _load_npy(self, path, lazy=False, chunks="auto"):

        if lazy:
            try:
                return da.from_npy_stack(path)

            except NotADirectoryError:
                mmap = np.load(path, mmap_mode="r")
                return da.from_array(mmap, chunks=chunks)

        else:
            return np.load(path.as_posix(), allow_pickle=True)

    def _load_tdb(self, path, lazy=False, chunks="auto"):

        """
        Loads data from a TileDB file.

        Args:
            path (pathlib.Path): The path to the TileDB file.

        Returns:
            numpy.ndarray: The loaded data.

        """

        if lazy:
            tdb = tiledb.open(path.as_posix(), "r")
            data = da.from_array(tdb, chunks=chunks)

        else:

            with tiledb.open(path.as_posix(), "r") as tdb:
                data = tdb[:]  # Read all data from TileDB array

        return data

    def _load_h5(self, path, h5_loc, lazy=False, chunks="auto"):

        """
        Loads data from an HDF5 file.

        Args:
            path (pathlib.Path): The path to the HDF5 file.
            h5_loc (str): The location of the dataset in the HDF5 file.

        Returns:
            numpy.ndarray: The loaded data.

        """

        if lazy:
            data = h5py.File(path, "r")

            if h5_loc not in data:
                raise ValueError(f"cannot find dataset in file ({path}): {list(data.keys())}")

            data = data[h5_loc]
            data = da.from_array(data, chunks=chunks)

        else:
            with h5py.File(path, "r") as data:

                if h5_loc not in data:
                    raise ValueError(f"cannot find dataset in file ({path}): {list(data.keys())}")

                data = data[h5_loc][:] # Read all data from HDF5 file

        return data

    @staticmethod
    def _load_czi(path, lazy=False):

        """
        Loads a CZI file from the specified path and returns the data.

        Args:
            path (str or pathlib.Path): The path to the CZI file.

        Returns:
            numpy.ndarray: The loaded data from the CZI file.

        """

        if lazy:
            raise NotImplementedError("currently czi loading is not implemented with lazy loading. Use 'lazy=False'.")

        # Convert path to a pathlib.Path object if it's provided as a string
        path = Path(path) if isinstance(path, str) else path

        # Validate path
        assert isinstance(path, Path), "please provide 'path' as str or pathlib.Path"
        assert path.is_file(), f"cannot find file: {path}"

        # Read the CZI file using czifile
        data = czifile.imread(path.as_posix())

        # Remove single-dimensional entries from the shape of the data
        data = np.squeeze(data)

        # TODO would be useful to be able to drop non-1D axes. Not sure how to implement this though
        # if ignore_dimensions is not None:
        #     ignore_dimensions = list(ignore_dimensions) if isinstance(ignore_dimensions, int) else ignore_dimensions
        #     assert isinstance(ignore_dimensions, (list, tuple)), "please provide 'ignore_dimensions' as int, list or tuple"
        #
        #   for d in ignore_dimensions:
        #       np.delete(data, axis=k) # not tested that this actually works

        if len(data.shape) != 3:
            logging.warning(
                f"the dataset is not 3D but instead: {data.shape}. This will most likely create errors downstream in the pipeline.")

        return data

    @staticmethod
    def sort_alpha_numerical_names(file_names, sep="_"):
        """
        Sorts a list of file names in alpha-numeric order based on a given separator.

        Args:
            file_names (list): A list of file names to be sorted.
            sep (str, optional): Separator used for sorting file names. (default: "_")

        Returns:
            list: A sorted list of file names.

        Raises:
            None
        """
        # Check if file_names contains Path objects
        use_path = True if isinstance(file_names[0], Path) else False

        if use_path:
            # Convert Path objects to string paths
            file_names = [f.as_posix() for f in file_names]

        # Sort file names based on the numeric part after the separator
        file_names = sorted(file_names, key=lambda x: int(x.split(".")[0].split(sep)[-1]))

        if use_path:
            # Convert string paths back to Path objects
            file_names = [Path(f) for f in file_names]

        return file_names

    @staticmethod
    def _load_tiff(path, sep="_", lazy=False):

        """
        Loads TIFF image data from the specified path and returns a Dask array.

        Args:
            path (str or pathlib.Path): The path to the TIFF file or directory containing TIFF files.
            sep (str): The separator used in sorting the filenames (default: "_").

        Returns:
            dask.array.core.Array: The loaded TIFF data as a Dask array.

        Raises:
            AssertionError: If the provided path is not a string or pathlib.Path object.
            AssertionError: If the specified path or directory does not exist.
            NotImplementedError: If the dimensions of the TIFF data are not 3D.

        """

        # Convert path to a pathlib.Path object if it's provided as a string
        path = Path(path) if isinstance(path, str) else path

        # Validate path
        assert isinstance(path, Path), f"please provide a valid data location instead of: {path}"

        if path.is_dir():
            # If the path is a directory, load multiple TIFF files

            # Get a list of TIFF files in the directory
            files = [f for f in path.glob("*") if f.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]]
            assert len(files) > 0, "couldn't find .tiff files. Recognized extension: [tif, tiff, TIF, TIFF]"

            # Sort the file names in alphanumeric order
            files = IO.sort_alpha_numerical_names(file_names=files, sep=sep)

            # Read the TIFF files using dask.array and stack them
            stack = da.stack([dask_image.imread.imread(f.as_posix()) for f in files])
            stack = np.squeeze(stack)

            if len(stack.shape) != 3:
                raise NotImplementedError(f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D")

        elif path.is_file():
            # If the path is a file, load a single TIFF file

            if lazy:
                stack = dask_image.imread.imread(path)
            else:
                stack = tifffile.imread(path.as_posix())

        else:
            raise FileNotFoundError(f"cannot find directory or file: {path}")

        return stack

    @staticmethod
    def save(path, data, h5_loc=None, chunks=None, compression=None):

        """
        Save data to a specified file format.

        Args:
            path (str or pathlib.Path): The path to the output file.
            data (dict or np.ndarray or dask.array.Array): A dictionary containing the data to be saved, with keys as channel names and values as arrays.
            h5_loc (str): Name of the dataset within the file (applicable only for HDF5 format).
            chunks (tuple or None): The chunk size to be used when saving Dask arrays (applicable only for HDF5 format).
            compression (str or None): The compression method to be used when saving Dask arrays (applicable only for HDF5 format).

        Returns:
            list: A list containing the paths of the saved files.

        Raises:
            TypeError: If the provided path is not a string or pathlib.Path object.
            TypeError: If the provided data is not a dictionary.
            TypeError: If the provided data is not in a supported format.

        """

        # Cast the path to a pathlib.Path object if it's provided as a string
        if isinstance(path, (str, Path)):
            path = Path(path)
        else:
            raise TypeError("please provide 'path' as str or pathlib.Path data type")

        # Check if the data is a dictionary or data array, otherwise raise an error
        if isinstance(data, (np.ndarray, da.Array)):
            data = {"ch0": data}
        elif not isinstance(data, dict):
            raise TypeError("please provide data as dict of {channel_name:array} or np.ndarray")

        saved_paths = []  # Initialize an empty list to store the paths of the saved files
        for k in data.keys():
            channel = data[k]

            # infer chunks if necessary
            if chunks == "infer":

                new_chunks = []
                for dim in channel.shape:
                    dim = int(dim * 0.1)
                    dim = min(100, dim)
                    dim = max(1, dim)
                    new_chunks.append(dim)

                chunks = tuple(new_chunks)
                logging.warning(f"inferred chunk size: {chunks}")

            # infer compression
            if compression == "infer":
                size = channel.size * channel.itemsize
                if size > 10e9 and path.suffix in [".h5", ".hdf5"]:
                    compression = "gzip"
                    logging.warning(f"inferred compression: {compression}")
                else:
                    compression = None

            # Check if the channel is a numpy.ndarray or a dask.array.Array, otherwise raise an error
            if not isinstance(channel, (np.ndarray, da.Array)):
                raise TypeError("please provide data as either 'numpy.ndarray' or 'da.array.Array'")

            if path.suffix in [".h5", ".hdf5"]:
                # Save as HDF5 format

                fpath = path

                # create dataset location
                if isinstance(h5_loc, dict):
                        loc = h5_loc[k]

                elif h5_loc is None:
                    loc = k if "/" in str(k) else f"io/{k}"

                elif len(data) == 1:
                        loc = f"{h5_loc}/{k}" if "/" not in h5_loc[:-1] else h5_loc

                else:
                    loc = f"{h5_loc}/{k}"

                logging.warning(f"loc: {loc}")
                logging.info(f"saving channel {k} to '{loc}'")

                if isinstance(channel, da.Array):
                    # Save Dask array
                    with ProgressBar(minimum=10, dt=1):
                        da.to_hdf5(fpath, loc, channel, chunks=chunks, compression=compression, shuffle=False)

                else:
                    # Save NumPy array
                    with h5py.File(fpath, "a") as f:
                        ds = f.create_dataset(loc, shape=channel.shape, chunks=chunks,
                                              compression=compression, shuffle=False, dtype=channel.dtype)
                        ds[:] = channel

                saved_paths.append(fpath)
                logging.info(f"dataset saved to {fpath}::{loc}")

            elif path.suffix == ".tdb":
                # Save as TileDB format

                if isinstance(channel, np.ndarray):
                    channel = da.from_array(channel, chunks=chunks if chunks is not None else "auto")

                fpath = path.with_suffix(f".{k}.tdb") if len(data.keys()) > 1 else path
                with ProgressBar(minimum=10, dt=1):
                    da.to_tiledb(channel, fpath.as_posix(), compute=True)

                saved_paths.append(fpath)
                logging.info(f"dataset saved to {fpath}")

            elif path.suffix in [".tiff", ".TIFF", ".tif", ".TIF"]:
                # Save as TIFF format

                fpath = path.with_suffix(f".{k}.tiff") if len(data.keys()) > 1 else path
                tifffile.imwrite(fpath, data=channel)

                saved_paths.append(fpath)
                logging.info(f"saved data to {fpath}")

            elif path.suffix in [".czi", ".CZI"]:
                raise NotImplementedError("currently we are not aware that python can save images in .czi format.")

            elif path.suffix in [".npy", ".NPY"]:

                fpath = path.with_suffix(f".{k}.npy") if len(data.keys()) > 1 else path

                if isinstance(channel, np.ndarray):
                    np.save(file=fpath.as_posix(), arr=channel)

                else:
                    with ProgressBar(minimum=10, dt=1):
                        da.to_npy_stack(fpath, x=channel, axis=0)

                saved_paths.append(fpath)
                logging.info(f"saved data to {fpath}")

            else:
                raise TypeError("please provide output format as .h5, .tdb, .npy or .tiff file")

        return saved_paths if len(saved_paths) > 1 else saved_paths[0]  # Return the list of saved file paths

class MotionCorrection:

    """
    Class for performing motion correction using the Caiman library.

    Args:
        working_directory (str or Path, optional): Working directory for temporary files.
            If not provided, the temporary directory is created.

    Attributes:
        working_directory (str or Path): Working directory for temporary files.
        tempdir: Temporary directory path.
        io: Instance of the IO class for input/output operations.
        dummy_folder_name (str): Name of the dummy folder used for motion correction.
        dview: Cluster setup for Caiman.
        mmap_path: Path to the memory-mapped file.

    Methods:
        run: Runs the motion correction algorithm.
        validate_input: Validates the input data for motion correction.
        clean_up: Cleans up temporary files created during motion correction.
        get_frames_per_file (deprecated): Computes the number of frames per file.
        get_data: Retrieves the motion-corrected data.

    """

    def __init__(self, working_directory=None, logging_level=logging.INFO):

        """
        Initializes the MotionCorrection object.

        Args:
            working_directory (str or Path, optional): Working directory for temporary files.
                If not provided, the temporary directory is created.

        """

        # is only relevant if provided with a .tdb or np.ndarray
        # otherwise the .mmap file is created in the same folder
        # as the input file.

        logging.basicConfig(level=logging_level)

        self.working_directory = working_directory
        self.tempdir = None

        self.io = IO()

        # needed if only one dataset in .h5 files. Weird behavior from caiman.MotionCorrection
        self.dummy_folder_name = "delete_me"

        # output location
        self.mmap_path = None
        self.tiff_path = None

    def run(self, input_, h5_loc="",
            max_shifts=(50, 50), niter_rig=3, splits_rig=14, num_splits_to_process_rig=None,
            strides=(48, 48), overlaps=(24, 24), pw_rigid=False, splits_els=14,
            num_splits_to_process_els=None, upsample_factor_grid=4, max_deviation_rigid=3,
            nonneg_movie=True, gSig_filt=(20, 20)):

        """

        Runs the motion correction algorithm on the input data. Adapted from caiman.motion_correction.MotionCorrect.

        max_shifts: tuple
            maximum allow rigid shift

        niter_rig':int
            maximum number of iterations rigid motion correction, in general is 1. 0
            will quickly initialize a template with the first frames

        splits_rig': int
         for parallelization split the movies in  num_splits chuncks across time

        num_splits_to_process_rig: list,
            if none all the splits are processed and the movie is saved, otherwise at each iteration
            num_splits_to_process_rig are considered

        strides: tuple
            intervals at which patches are laid out for motion correction

        overlaps: tuple
            overlap between pathes (size of patch strides+overlaps)

        pw_rigid: bool, default: False
            flag for performing motion correction when calling motion_correct

        splits_els':list
            for parallelization split the movies in  num_splits chuncks across time

        num_splits_to_process_els: list,
            if none all the splits are processed and the movie is saved  otherwise at each iteration
             num_splits_to_process_els are considered

        upsample_factor_grid:int,
            upsample factor of shifts per patches to avoid smearing when merging patches

        max_deviation_rigid:int
            maximum deviation allowed for patch with respect to rigid shift

        nonneg_movie: boolean
            make the SAVED movie and template mostly nonnegative by removing min_mov from movie

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        var_name_hdf5: str, default: 'mov'
            If loading from hdf5, name of the variable to load

         is3D: bool, default: False
            Flag for 3D motion correction

         indices: tuple(slice), default: (slice(None), slice(None))
            Use that to apply motion correction only on a part of the FOV

        """

        # import NormCorre module
        from jnormcorre import motion_correction

        input_ = self._validate_input(input_, h5_loc=h5_loc)
        self.input_ = input_

        # Create MotionCorrect instance
        mc = motion_correction.MotionCorrect(input_, var_name_hdf5=h5_loc,
                max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                num_splits_to_process_rig=num_splits_to_process_rig, strides=strides, overlaps=overlaps,
                pw_rigid=pw_rigid, splits_els=splits_els, num_splits_to_process_els=num_splits_to_process_els,
                upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                nonneg_movie=nonneg_movie, gSig_filt=gSig_filt)

        # Perform motion correction
        obj, registered_filename = mc.motion_correct(save_movie=True)
        self.shifts = mc.shifts_rig

        logging.info(f"result saved to: {registered_filename}")

        # Check if the motion correction generated the mmap file
        if len(mc.fname_tot_rig) < 1 or not Path(mc.fname_tot_rig[0]).is_file():
            raise FileNotFoundError(f"motion correction failed unexpectedly. mmap path: {mc.mmap}")

        # Set the mmap_path attribute to the generated mmap file
        self.mmap_path = mc.fname_tot_rig[0]
        self.tiff_path = registered_filename[0]

    def _validate_input(self, input_, h5_loc):
        """
        Validate and process the input for motion correction.

        Args:
            input_ (Union[str, Path, np.ndarray]): Input data for motion correction.
            h5_loc (str): Dataset name in case of input being an HDF5 file.

        Returns:
            Union[Path, np.ndarray]: Validated and processed input.

        Raises:
            FileNotFoundError: If the input file is not found.
            ValueError: If the input format is not supported or required arguments are missing.
            NotImplementedError: If the input format is not implemented.

        Notes:
            - Motion Correction fails with custom h5_loc names in cases where there is only one folder (default behavior incorrect).
            - A temporary .tiff file is created if the input is an ndarray, which needs to be deleted later using the 'clean_up()' method.
        """

        if isinstance(input_, (str, Path)):
            # If input is a string or Path object

            input_ = Path(input_)

            if not input_.exists():
                raise FileNotFoundError(f"cannot find input_: {input_}")

            if input_.suffix in [".h5", ".hdf5"]:
                # If input is an HDF5 file

                if h5_loc is None:
                    raise ValueError("Please provide 'h5_loc' argument when providing .h5 file as data input.")

                with h5py.File(input_.as_posix(), "a") as f:
                    if h5_loc not in f:
                        raise ValueError(f"cannot find dataset {h5_loc} in provided .h5 file.")

                    # Motion Correction fails with custom h5_loc names in cases where there is only one folder (default behavior incorrect)
                    if len(f.keys()) < 2:
                        f.create_group(self.dummy_folder_name)

                return input_

            elif input_.suffix in [".tiff", ".TIFF", ".tif", ".TIF"]:
                # If input is a TIFF file
                return input_

            else:
                raise ValueError(f"unknown input type. Please provide .h5 or .tiff file.")

        elif isinstance(input_, np.ndarray):
            # If input is a ndarray create a temporary TIFF file to run the motion correction on

            logging.warning("caiman.motion_correction requires a .tiff or .h5 file to perform the correction. A temporary .tiff file is created which needs to be deleted later by calling the 'clean_up()' method of this module.")

            if self.working_directory is None:
                self.working_directory = tempfile.TemporaryDirectory()

            if isinstance(self.working_directory, tempfile.TemporaryDirectory):
                temp_h5_path = Path(self.working_directory.name)
            else:
                temp_h5_path = Path(self.working_directory)

            assert temp_h5_path.exists(), f"working directory doesn't exist: {temp_h5_path}"

            temp_h5_path = temp_h5_path.joinpath(f"{self.dummy_folder_name}.tiff").as_posix()
            tifffile.imwrite(temp_h5_path, input_)

            return temp_h5_path

        else:
            raise ValueError(f"please provide input_ as one of: np.ndarray, str, Path")

    def clean_up(self):
        """
        Clean up temporary files and resources associated with motion correction.

        Args:
            input_ (Union[str, Path]): Input data used for motion correction.

        Notes:
            - This method should be called after motion correction is completed to remove temporary files and resources.

        Raises:
            FileNotFoundError: If the input file is not found.

        """

        input_ = self.input_

        if input_.suffix in [".h5", ".hdf5"]:
            # If input is an HDF5 file

            with h5py.File(input_.as_posix(), "a") as f:
                # Delete dummy folder if created earlier; see validation method
                if self.dummy_folder_name in f:
                    del f[self.dummy_folder_name]

        # Remove mmap result
        if self.mmap_path is not None and Path(self.mmap_path).is_file():
            os.remove(self.mmap_path)

        if self.tiff_path is not None and Path(self.tiff_path).is_file():
            os.remove(self.tiff_path)

        # Remove temp .h5 if necessary
        if self.working_directory is not None:
            temp_h5_path = Path(self.working_directory.name) if isinstance(self.working_directory, tempfile.TemporaryDirectory) else Path(self.working_directory)
            temp_h5_path = temp_h5_path.joinpath(f"{self.dummy_folder_name}.h5").as_posix()
            if temp_h5_path.is_file():
                os.remove(temp_h5_path.as_posix())

    @staticmethod
    @deprecated("use caiman's built-in file splitting function instead")
    def get_frames_per_file(input_, frames_per_file, loc=None):

        if frames_per_file == "auto":

            (Z, X, Y), chunksize, dtype = get_data_dimensions(input_, loc=loc, return_dtype=True)
            byte_num = np.dtype(dtype).itemsize
            array_size = Z * X * Y * byte_num

            ram_size = psutil.virtual_memory().total

            if ram_size < array_size * 2:
                logging.warning(f"available RAM ({ram_size}) is smaller than twice the data size ({array_size}. Automatically splitting files into smaller fragments. Might lead to unexpected behavior on the boundary between fragments.")
                frames_per_file = int (Z  / np.floor(array_size / ram_size) / 2)

        elif isinstance(frames_per_file, int):
            pass

        elif isinstance(frames_per_file, float):
            frames_per_file = int(frames_per_file)

        else: raise ValueError(f"Please provide one of these options for 'split_file' flag: None, 'auto', int, float")

        return frames_per_file

    def save(self, output=None, h5_loc="mc", chunks=None, compression=None, remove_intermediate=True):

        """
        Retrieve the motion-corrected data and optionally save it to a file.

        Args:
            output (Optional[Union[str, Path]]): Output file path where the data should be saved.
            loc (Optional[str]): Location within the HDF5 file to save the data (required when output is an HDF5 file).
            prefix (str): Prefix to be added to the keys when saving to an HDF5 file.
            chunks (Optional[Tuple[int]]): Chunk shape for creating a dask array when saving to an HDF5 file.
            compression (Optional[str]): Compression algorithm to use when saving to an HDF5 file.
            remove_mmap (bool): Whether to remove the mmap file associated with motion correction after retrieving the data.

        Returns:
            np.ndarray or None: The motion-corrected data as a NumPy array. If 'output' is specified, returns None.

        Raises:
            ValueError: If the mmap_path is None or the mmap file is not found.
            ValueError: If 'output' is an HDF5 file but 'loc' is not provided.
            ValueError: If 'output' is not None, str, or pathlib.Path.

        Notes:
            - This method should be called after motion correction is completed by using the 'run()' function.
            - If 'output' is specified, the motion-corrected data is saved to the specified file using the I/O module.
            - If 'remove_mmap' is set to True, the mmap file associated with motion correction is deleted after retrieving the data.

        """

        # Check if the tiff output is available
        tiff_path = self.tiff_path
        if tiff_path is None:
            raise ValueError("tiff_path is None. Please compute motion correction first by using the 'run()' function")

        tiff_path = Path(tiff_path)
        if not tiff_path.is_file():
            raise FileNotFoundError(f"could not find tiff file: {tiff_path}. Maybe the 'clean_up()' function was called too early?")

        data = tifffile.imread(tiff_path.as_posix())

        # If output is None, return the motion-corrected data as a NumPy array
        if output is None:
            return data

        elif isinstance(output, (str, Path)):
            output = Path(output) if isinstance(output, Path) else output

            # Create a dask array from the memory-mapped data with specified chunking and compression
            data = da.from_array(data, chunks=chunks)

            # Check if the output file is an HDF5 file and loc is provided
            if output.suffix in [".h5", ".hdf5"] and h5_loc is None:
                raise ValueError("when saving to .h5 please provide a location to save to instead of 'h5_loc=None'")

            # split location into channel and folder information
            split_h5 = h5_loc.split("/")
            if len(split_h5) < 2:
                loc, channel = "mc", h5_loc
            elif len(split_h5) == 2:
                loc, channel = split_h5
            elif len(split_h5) > 2:
                loc = "/".join(split_h5[:-1])
                channel = split_h5[-1]
            else:
                raise ValueError(f"please provide h5_loc as 'channel_name' or 'folder/channel_name' instead of {h5_loc}")

            # Save the motion-corrected data to the output file using the I/O module
            self.io.save(output, data={channel:data}, h5_loc=loc, chunks=chunks, compression=compression)

        else:
            raise ValueError(f"please provide output as None, str or pathlib.Path instead of {output}")

        # If remove_mmap is True, delete the mmap file associated with motion correction
        if remove_intermediate:
            self.clean_up()

class Delta:

    """
    The Delta class provides methods for calculating the delta signal from input data.
    The input data can be either a numpy ndarray, a TileDB array, or a file path.
    The class supports various preprocessing options, such as loading data into memory,
    creating a Dask array, or using shared memory.

    Methods:
    - run(method="background", window=None, overwrite_first_frame=True, use_dask=True):
        Runs the delta calculation on the input data and returns the result.
        The 'method' parameter specifies the delta calculation method.
        The 'window' parameter sets the size of the minimum filter window.
        The 'overwrite_first_frame' parameter determines whether to overwrite the first frame of the result.
        The 'use_dask' parameter determines whether to use Dask for parallel processing.

    - load_to_memory(path, loc=None):
        Loads data from the specified path into memory and returns it as a numpy ndarray.
        The 'loc' parameter specifies the location of the data in the HDF5 file.

    - save_to_tdb(arr: np.ndarray) -> str:
        Saves a numpy ndarray to a TileDB array and returns the path to the TileDB array.

    - prepare_data(input_, in_memory=True, shared=True, use_dask=True):
        Preprocesses the input data by converting it to a TileDB array and optionally loading it into memory
        or creating a Dask array.

    - calculate_background_pandas(arr: np.ndarray, window: int, method="background",
                                  inplace: bool = True) -> np.ndarray:
        [DEPRECATED] Calculates the background signal using a pandas-based implementation.
        The 'arr' parameter is the input data.
        The 'window' parameter sets the size of the rolling minimum window.
        The 'method' parameter specifies the type of delta calculation.
        The 'inplace' parameter determines whether to modify the input data in place.

    - calculate_delta_min_filter(arr: np.ndarray, window: int, method="background", inplace=False) -> np.ndarray:
        Calculates the delta signal using the minimum filter approach.
        The 'arr' parameter is the input data.
        The 'window' parameter sets the size of the minimum filter window.
        The 'method' parameter specifies the type of delta calculation.
        The 'inplace' parameter determines whether to modify the input data in place.

    """

    def __init__(self, input_, loc=None):
        """
        Initializes a Delta object.

        Args:
        - input_: The input data to be processed. It can be a file path (str or Path object),
          a numpy ndarray, or a TileDB array.
        - loc: The location of the data in the HDF5 file. This parameter is optional and only
          applicable when 'input_' has the .h5 extension.
        - in_memory: A boolean flag indicating whether the data should be loaded into memory
          or kept on disk. Default is True.
        - parallel: A boolean flag indicating whether to use parallel processing. Default is False.

        """
        # Convert the input to a Path object if it is a string
        self.input_ = Path(input_) if isinstance(input_, str) else input_
        self.res = None

        # Get the dimensions and chunk size of the input data
        self.dim, self.chunksize = get_data_dimensions(self.input_, loc=loc)

        # The location of the data in the HDF5 file (optional, only applicable for .h5 files)
        self.loc = loc

    def run(self, method="background", window=None, chunks="infer", output_path=None,
            overwrite_first_frame=True, lazy=True):
        """
        Runs the delta calculation on the input data.

        Args:
        - method: The method to use for delta calculation. Options are "background", "dF", or "dFF".
          Default is "background".
        - window: The size of the window for the minimum filter. If None, the window size will be
          automatically determined based on the dimensions of the input data. Default is None.
        - overwrite_first_frame: A boolean flag indicating whether to overwrite the values of the
          first frame with the second frame after delta calculation. Default is True.
        - use_dask: A boolean flag indicating whether to use Dask for lazy loading and computation.
          Default is True.

        Returns:
        - The delta calculation results as a numpy ndarray.

        Raises:
        - NotImplementedError: If the input data type is not recognized.

        """
        # Prepare the data for processing
        data = self.prepare_data(self.input_, h5_loc=self.loc, chunks=chunks, output_path=output_path, lazy=lazy)

        # Sequential execution
        if isinstance(data, np.ndarray):
            # Calculate delta using the minimum filter on the input data
            res = self.calculate_delta_min_filter(data, window, method=method, inplace=False)

        # Parallel from .tdb file
        elif isinstance(data, (str, Path)) and Path(data).suffix in (".tdb"):
            # Warning message for overwriting the .tdb file
            logging.warning("This function will overwrite the provided .tdb file!")

            # Define a wrapper function for parallel execution
            calculate_delta_min_filter = self.calculate_delta_min_filter

            def wrapper(path, ranges):
                (x0, x1), (y0, y1) = ranges

                # Open the TileDB array and load the specified range
                with tiledb.open(path, mode="r") as tdb:
                    data = tdb[:, x0:x1, y0:y1]
                    res = calculate_delta_min_filter(data, window, method=method, inplace=False)

                # Overwrite the range with the calculated delta values
                with tiledb.open(path, mode="w") as tdb:
                    tdb[:, x0:x1, y0:y1] = calculate_delta_min_filter(data, window, method=method, inplace=False)

            # Extract the path from the input data
            path = data.as_posix()

            # Get the dimensions and chunk size of the .tdb file
            (Z, X, Y), chunksize = get_data_dimensions(path, loc=None)
            assert chunksize is not None
            cz, cx, cy = chunksize

            # Execute the calculation in parallel using Dask
            with LocalCluster() as lc:
                with Client(lc) as client:
                    futures = []
                    for x0 in range(0, X, cx):
                        for y0 in range(0, X, cy):
                            range_ = ((x0, x0 + cx), (y0, y0 + cy))
                            futures.append(client.submit(wrapper, path, range_))

                    # Gather the results from parallel executions
                    client.gather(futures)

            # Load the modified .tdb file into memory
            io = IO()
            res = io.load(Path(path))

        elif isinstance(data, da.core.Array):
            # Calculate delta using Dask array for lazy loading and computation
            res = data.map_blocks(self.calculate_delta_min_filter,
                                  window=window,
                                  method=method,
                                  inplace=False,
                                  dtype=float).compute()

        else:
            raise ValueError(f"Input data type not recognized: {type(data)}")

        # Overwrite the first frame with the second frame if required
        if overwrite_first_frame:
            res[0] = res[1]

        self.res = res
        return res

    def save(self, output_path, h5_loc="df", chunks=(-1, "auto", "auto"), compression=None):

        io = IO()
        io.save(output_path, data=self.res, h5_loc=h5_loc, chunks=chunks, compression=compression)

    def prepare_data(self, input_, chunks="infer", h5_loc=None, output_path=None, lazy=True):

        """
        Preprocesses the input data by converting it to a TileDB array and optionally loading it into memory
        or creating a Dask array.

        Args:
        - input_: A Path object or numpy ndarray representing the input data to be preprocessed.
        - in_memory: A boolean flag indicating whether the data should be loaded into memory or kept on disk.
        - shared: A boolean flag indicating whether to create a shared memory array or a Dask array. This flag
          is only applicable if 'in_memory' is True.

        Returns:
        - If 'in_memory' is False, returns the path to the TileDB array on disk (created if necessary).
        - If 'in_memory' is True and 'shared' is False, returns the data as a numpy ndarray.
        - If 'in_memory' is True and 'shared' is True, returns the data as a Dask array.

        Raises:
        - TypeError: If the input data type is not recognized.
        """

        io = IO()

        if not lazy:
            data = io.load(input_, h5_loc=h5_loc, lazy=lazy)

            if not isinstance(data, da.Array):
                data = da.from_array(data, chunks=(self.dim))
            data = da.rechunk(data, chunks=(-1, "auto", "auto"))
            return data

        # convert to .tdb
        elif isinstance(input_, Path) and input_.suffix == ".tdb":
            return io.load(input_, lazy=lazy)

        elif isinstance(input_, Path):
            # if the input is a file path, load it into memory and convert to TileDB array

            data = io.load(input_, h5_loc=self.loc)

            new_path = input_.with_suffix(".tdb") if output_path is None else Path(output_path)
            if not new_path.suffix in (".tdb"):
                raise ValueError(f"Please provide an output_path with '.tdb' ending instead of {new_path.suffix}")

            io.save(new_path, data=data, chunks=chunks)

            return io.load(new_path, lazy=lazy)

        elif isinstance(input_, (np.ndarray, da.Array)):
            # if the input is a numpy ndarray, convert to TileDB array

            if output_path is None:
                raise ValueError("when providing an array as input, an output_path needs to be provided as well.")
            else:
                new_path = Path(output_path)

            if not new_path.suffix in (".tdb"):
                raise ValueError(f"Please provide an output_path with '.tdb' ending instead of {new_path.suffix}")

            io.save(new_path, data=input_, chunks=chunks)

            return io.load(new_path, lazy=lazy)

        else:
            raise TypeError(f"do not recognize data type: {type(input_)}")

    @staticmethod
    @deprecated(reason="faster implementation but superseded by: calculate_background_even_faster")
    def calculate_background_pandas(arr: np.ndarray, window: int, method="background",
                                    inplace: bool = True) -> np.ndarray:

        if len(np.squeeze(arr)) < 2:
            arr = np.expand_dims(arr, axis=0)

        arr = np.atleast_3d(arr)

        if not inplace:
            res = np.zeros(arr.shape, arr.dtype)

        methods = {
            "background": lambda x, background: background,
            "dF": lambda x, background: x - background,
            "dFF": lambda x, background: np.divide(x - background, background)
        }
        if method not in methods.keys(): raise ValueError(
            f"please provide a valid argument for 'method'; one of : {methods.keys()}")

        delta = methods[method]  # choose method

        # iterate over pixels
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):

                z = arr[:, x, y]

                # Pad the trace with the edge values
                padded = pd.Series(np.pad(z, window, mode='edge'))

                # Compute the rolling minimum with the specified window size
                MIN = padded.rolling(window).min().values[window:]

                # Take the maximum of the values to produce the final background signal
                background = np.zeros((2, len(z)))
                background[0, :] = MIN[:-window]
                background[1, :] = MIN[window:]
                background = np.nanmax(background, axis=0)

                if inplace:
                    arr[:, x, y] = delta(z, background)
                else:
                    res[:, x, y] = delta(z, background)

        return np.squeeze(arr) if inplace else np.squeeze(res)

    @staticmethod
    def calculate_delta_min_filter(arr: np.ndarray, window: int, method="background", inplace=False) -> np.ndarray:

        original_dims = arr.shape

        # Ensure array is at least 3D
        if len(np.squeeze(arr)) < 2:
            arr = np.expand_dims(arr, axis=0)  # necessary to preserve order in case of 1D array

        arr = np.atleast_3d(arr)

        # choose delta function
        methods = {
            "background": lambda x, background: background,
            "dF": lambda x, background: x - background,
            "dFF": lambda x, background: np.divide(x - background, background)
        }
        if method not in methods.keys(): raise ValueError(
            f"please provide a valid argument for 'method'; one of : {methods.keys()}")

        delta_func = methods[method]

        # create result array if not inplace
        if not inplace:
            res = np.zeros(arr.shape, arr.dtype)

        # iterate over pixels
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):

                # Get the signal for the current pixel
                z = arr[:, x, y]

                # Pad the signal with the edge values and apply the minimum filter
                MIN = minimum_filter1d(np.pad(z, pad_width=(0, window), mode='edge'), size=window + 1, mode="nearest",
                                       origin=int(window / 2))

                # Shift the minimum signal by window/2 and take the max of the two signals
                background = np.zeros((2, len(z)))
                background[0, :] = MIN[:-window]
                background[1, :] = MIN[window:]
                background = np.nanmax(background, axis=0)

                if inplace:
                    arr[:, x, y] = delta_func(z, background)
                else:
                    res[:, x, y] = delta_func(z, background)

        if inplace:
            res = arr

        # restore initial dimensions
        res = np.reshape(res, original_dims)

        return res

class XII:

    def __init__(self, file_path, dataset_name, num_channels=1, sampling_rate=None, channel_names=None):
        self.container = self.load_xii(file_path, dataset_name, num_channels, sampling_rate, channel_names)

    @staticmethod
    def load_xii(file_path, dataset_name, num_channels=1, sampling_rate=None, channel_names=None):

        """
        :param unit:
        :param dataset_name:
        :param file:
        :param channels:
        :param unify_timeline:
        :param sampling_rate: in ms
        :return:
        """


        # define sampling rate
        if sampling_rate is None:
            sampling_rate = 1

        elif isinstance(sampling_rate, (float, int)):
            sampling_rate = float(sampling_rate)

        elif isinstance(sampling_rate, str):

            units = OrderedDict(
                [("ps", 1e-12), ("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("s", 1), ("min", 60), ("h", 60*60)]
            )

            found_unit = False
            for key, value in units.items():

                if sampling_rate.endswith(key):
                    sampling_rate = sampling_rate.replace(key, "")
                    sampling_rate = float(sampling_rate) * value
                    found_unit = True

                    break

            if not found_unit:
                raise ValueError(f"when providing the sampling_rate as string, the value has to end in one of these units: {units.keys()}")

        # define steps
        timestep = sampling_rate * num_channels

        # load data
        with h5py.File(file_path, "r") as f:

            if dataset_name not in f:
                raise ValueError(f"cannot find dataset in file. Choose one of: {list(f.keys())}")

            data = f[dataset_name][:]

        # split data
        container = {}
        for ch in range(num_channels):

            data_ch = data[ch::num_channels]

            if isinstance(timestep, int):
                idx = pd.RangeIndex(timestep*ch, timestep*ch + len(data_ch)*timestep, timestep)
            elif isinstance(timestep, float):
                idx = pd.Index(np.arange(timestep*ch, timestep*ch + len(data_ch)*timestep, timestep))
            else:
                raise ValueError(f"sampling_rate should be able to be cast to int or float instead of: {type(timestep)}")

            data_ch = pd.Series(data_ch, index=idx)
            if  channel_names is None:
                ch_name = f"ch{ch}"
            else:
                ch_name = channel_names[ch]

            container[ch_name] = data_ch

        return container

    def __getitem__(self, item):

        if item not in self.container.keys():
            raise ValueError(f"cannot find {item}. Provide one of: {self.container.keys()}")

        return self.container[item]

    def get_camera_timing(self, dataset_name, downsample=100, prominence=0.5):

        camera_out = self.container[dataset_name]

        peaks, _ = signal.find_peaks(camera_out.values[::downsample], prominence=prominence)
        peaks = pd.Series([camera_out.index[p * downsample] for p in peaks])

        return peaks

    def detrend(self, dataset_name, window=25, inplace=True):

        trace = self.container[dataset_name]
        trend = trace.rolling(window, center=False).min()

        de_trended = trace - trend
        de_trended = de_trended.iloc[window:-window]

        if inplace:
            self.container[dataset_name] = de_trended

        return de_trended

    @staticmethod
    def align(video, timing, idx_channel=0, num_channels=2, offset_start=0, offset_stop=0):

        idx = np.arange(offset_start+idx_channel, offset_start + len(video)*2 - offset_stop, num_channels)

        if len(idx) != len(video):
            raise ValueError(f"video length and indices don't align: video ({len(video)}) vs. idx ({len(idx)}). \n{idx}")

        mapping = timing.to_dict()
        idx = pd.Index([np.round(mapping[id_], decimals=3) for id_ in idx])

        return idx, mapping

    def show(self, dataset_name, mapping, viewer=None, viewer1d=None, down_sample=100,
             colormap=None, window=160, ylabel="XII", xlabel="step"):

        # todo: test with Video

        xii = self.container[dataset_name][::down_sample]

        if viewer1d is None:
            v1d = napari_plot.ViewerModel1D()
            qt_viewer = QtViewer(v1d)
        else:
            v1d = viewer1d

        v1d.axis.y_label = ylabel
        v1d.axis.x_label = xlabel
        v1d.text_overlay.visible = True
        v1d.text_overlay.position = "top_right"

        # create attachable qtviewer
        X, Y = xii.index, xii.values
        line = v1d.add_line(np.c_[X, Y], name=ylabel, color=colormap)

        def update_line(event: Event):
            Z, _, _ = event.value
            z0, z1 = Z-window, Z

            if z0 < 0:
                z0 = 0

            t0, t1 = mapping[z0], mapping[z1]

            xii_ = xii[(xii.index >= t0) & (xii.index <= t1)]

            x_, y_ = xii_.index, xii_.values
            line.data = np.c_[x_, y_]

            v1d.reset_x_view()
            v1d.reset_y_view()

        viewer.dims.events.current_step.connect(update_line)

        if viewer1d is None:
            viewer.window.add_dock_widget(qt_viewer, area="bottom", name=ylabel)

        return viewer, v1d
