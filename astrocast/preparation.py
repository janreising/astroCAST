import logging
import os
import shutil
import tempfile
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple, Literal, Callable, List

import czifile
import dask
import dask.array as da
import dask_image.imread
import h5py
import numpy as np
import pandas as pd
import psutil
import tifffile
import tiledb
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from deprecated import deprecated
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import minimum_filter1d
from skimage.transform import resize
from skimage.util import img_as_uint
from tqdm import tqdm

from astrocast.helper import get_data_dimensions


class Input:
    """ Class for loading time series images and converting to an astroCAST compatible format.

    Args:
            logging_level: Sets the level at which information is logged to the console as an integer value.
                The built-in levels in the logging module are, in increasing order of severity:
                debug (10), info (20), warning (30), error (40), critical (50).

    **Example**::

            inp = Input()
            inp.run('path/to/images', output_path='path/to/output.h5' channels=1, loc_out='data')

    """

    def __init__(self, logging_level: int = logging.INFO):
        logging.basicConfig(level=logging_level)

    def run(
            self, input_path: Union[str, Path], output_path: Union[str, Path] = None, sep: str = "_",
            channels: Union[int, dict] = 1, z_slice: Tuple[int, int] = None, lazy: bool = True,
            subtract_background: Union[str, np.ndarray] = None,
            subtract_func: Union[Literal['mean', 'max', 'min', 'std'], Callable] = "mean",
            rescale: Union[float, Tuple[int, int]] = None, dtype: type = int, in_memory: bool = False,
            loc_in: str = None, loc_out: str = "data", chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced",
            chunks: Tuple[int, int, int] = None, compression: Literal['gzip', 'szip', 'lz4'] = None
    ) -> Union[np.ndarray, dict]:

        """ Loads input data from a specified path, performs data processing, and optionally saves the processed data.

        Args:
            input_path: Path to the input file or directory.
            output_path: Path to save the processed data. If None, the processed data is returned.
            loc_in: Input dataset in the HDF5 file that is loaded.
            loc_out: Output dataset in the HDF5 file that is saved.
            z_slice: Selection of frames that are processed.
            sep: Separator used for sorting file names, `['file_01.tiff', 'file_02.tiff']`.
            channels: Number of channels or dictionary specifying channel names.
            subtract_background: Either channel name or array that is subtracted.
            subtract_func: Function to use for background subtraction.
            rescale: Scale factor or tuple specifying the new dimensions.
            dtype: Data type to convert the processed data.
            in_memory: If True, the processed data is loaded into memory.
            chunk_strategy: Strategy to use when inferring size of chunks.
            chunks: Chunk size to use when saving to HDF5 or TileDB.
            compression: Compression method to use when saving to HDF5 or TileDB.
            lazy: If True, the data is loaded on demand.

        """

        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        assert isinstance(input_path, Path), "please provide 'input_path' as str or input_pathlib.input_path"
        assert input_path.is_file() or input_path.is_dir(), f"cannot find input: {input_path}"

        logging.info("loading data ...")
        io = IO()
        data = io.load(input_path, loc=loc_in, sep=sep, z_slice=z_slice, lazy=lazy, chunks=(1, -1, -1))

        logging.info("preparing data ...")
        if isinstance(rescale, int) or (isinstance(rescale, (tuple, list)) and isinstance(rescale[0], int)):
            absolute_rescale = True
        else:
            absolute_rescale = False

        data = self._prepare_data(
            data, channels=channels, loc_out=loc_out, subtract_background=subtract_background,
            subtract_func=subtract_func, rescale=rescale, absolute_rescale=absolute_rescale, dtype=dtype, lazy=in_memory
        )

        logging.debug(f"data type: {type(data[list(data.keys())[0]])}")

        # return result
        if output_path is None:
            return data

        logging.info("saving data ...")
        io.save(
            output_path, data, loc=loc_out, chunk_strategy=chunk_strategy, chunks=chunks, compression=compression
        )

    @staticmethod
    def _subtract_background(data, channels, subtract_background, subtract_func):

        """
        Subtract the background from the data.

        Args:
            data (dict): A dictionary mapping channel names to data arrays.
            channels (dict): A dictionary mapping channel indices to names.
            subtract_background (np.ndarray or str or callable): The background image to subtract or a string specifying
                the channel name to use as the background, or a callable function for background reduction.
            subtract_func (str or callable): The reduction function to use for background subtraction if
                `subtract_background` is a string or a callable function. Possible options: [mean, std, min, max].

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
                raise ValueError(
                    f"cannot find channel to subtract or found too many. Choose only one of : {list(channels.values())}."
                )

            background = data[background_keys[0]]
            for k in background_keys:
                del data[k]

            # Reduce the background dimension using the specified reduction function
            if callable(subtract_func):
                reducer = subtract_func
            else:
                func_reduction = {"mean": da.mean, "std": da.std, "min": da.min, "max": da.max}
                assert subtract_func in func_reduction.keys(), f"cannot find reduction function. Please provide callable function or one of {func_reduction.keys()}"
                reducer = func_reduction[subtract_func]

            background = reducer(background, axis=0)

            # Check if the shape of the reduced background is compatible with the data arrays
            img_0 = list(data.values())[0][0, :, :]
            if background.shape != img_0.shape:
                raise ValueError(
                    f"incorrect dimension after reduction: data.shape {img_0.shape} vs. reduced.shape {background.shape}"
                )

            # Subtract the reduced background from each channel
            for k in data.keys():
                dtype = data[k].dtype

                data[k] = data[k] - background
                data[k] = data[k].astype(dtype)

        else:
            raise ValueError(
                "Please provide 'subtract_background' flag with one of: np.ndarray, callable function or str"
            )

        return data

    @staticmethod
    def _rescale_data(data: Union[np.ndarray, da.Array, dict], rescale, absolute_rescale=False):
        """
        Rescale the data arrays to a new size.

        Args:
            data (dict): A dictionary mapping channel names to data arrays.
            rescale (tuple, list, int, float): The rescaling factor or factors to apply to the data arrays.
                If a tuple or list, it should contain two elements representing the scaling factors for the X and Y axes.
                If an int or float, the same scaling factor will be applied to both axes.
                If given an int, it will assume that this is the requested final size.
                If given a float, it will multiply the current size by that value.
            absolute_rescale (bool): If True, rescale is assumed to be in absolute units (e.g., pixels), otherwise it is in relative units (e.g., percent).

        Returns:
            dict: A dictionary mapping channel names to the rescaled data arrays.

        Raises:
            ValueError: If the rescale type is mixed (e.g., int and float) or not one of tuple, list, int, or float.
            ValueError: If the length of the rescale tuple or list is not 2.
            TypeError: If the rescale type is not tuple, list, int, or float.

        """

        return_array = False
        if isinstance(data, (np.ndarray, da.Array)):
            data = {"dummy": data}
            return_array = True

        # Convert numbers to tuple (same factor for X and Y)
        if isinstance(rescale, (int, float)):
            rescale = (rescale, rescale)

        # validate the rescale type
        if not isinstance(rescale, (tuple, list, int, float)):
            raise ValueError("please provide 'rescale' flag as 2D tuple, list or number")
        elif isinstance(rescale, (tuple, list)) and len(rescale) != 2:
            raise ValueError("please provide 'rescale' flag as 2D tuple or list")
        elif isinstance(rescale, (tuple, list)) and not isinstance(rescale[0], type(rescale[1])):
            raise ValueError(
                f"mixed rescale type not allowed for 'rescale' flag:"
                f" {type(rescale[0])} vs {type(rescale[1])}"
            )

        # Apply resizing to each channel
        for k in data.keys():
            # Rescale the data array using the specified scaling factors and antialiasing

            arr = data[k]

            # Get the original size
            Z, X, Y = arr.shape
            dtype_original = arr.dtype
            chunks_original = arr.chunks

            # convert to relative scale if absolute size was provided
            if absolute_rescale:
                rescale = rescale[0] / X, rescale[1] / Y

            # Calculate the requested output dimensions
            new_shape = (Z, int(X * rescale[0]), int(Y * rescale[1]))
            new_chunks = (
                tuple([c for c in chunks_original[0]]), tuple([int(c * rescale[0]) for c in chunks_original[1]]),
                tuple([int(c * rescale[1]) for c in chunks_original[2]]))

            logging.warning(f"Resizing {k} from {arr.shape} to {new_shape}")
            logging.warning(f"Rescaling {k} from {chunks_original} to {new_chunks}")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                def custom_resize(chunk):
                    dtype = chunk.dtype
                    z, x, y = chunk.shape

                    new_shape_ = (z, int(x * rescale[0]), int(y * rescale[1]))

                    chunk = resize(image=chunk.astype(float), output_shape=new_shape_, anti_aliasing=True)
                    chunk = chunk.astype(dtype)

                    return chunk

                arr = arr.map_blocks(
                    lambda chunk: custom_resize(chunk), chunks=new_chunks,
                    meta=np.zeros(new_shape, dtype=dtype_original)
                )

            # restore initial chunks and convert to original dtype
            data[k] = arr.astype(dtype_original)

        if return_array:
            data = data["dummy"]

        return data

    def _prepare_data(
            self, data: Union[np.ndarray, da.Array], channels: Union[int, dict] = 1, subtract_background: str = None,
            subtract_func: Union[Literal['mean', 'max', 'min', 'std'], Callable] = "mean", loc_out: str = None,
            rescale: Union[float, Tuple[int, int]] = None, absolute_rescale: bool = False, dtype: type = int,
            lazy: bool = True
    ) -> dict:
        """Prepares the input data by applying various processing steps.

        Args:
            data: Input data to be prepared. Should be a 3D array.
            channels: Number of channels or dictionary specifying channel names.
            subtract_background: Background to subtract or channel name to use as background.
            subtract_func: Function to use for background subtraction.
            loc_out: Location to save the processed data.
            absolute_rescale: If True, rescale is expected to be the final image size after rescaling.
            rescale: Scale factor or tuple specifying the new dimensions.
            dtype: Data type to convert the processed data.
            lazy: If False, the processed data is loaded into memory.

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
                raise NotImplementedError(
                    f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D"
                )

            # Create a dictionary of channel names or indices
            if isinstance(channels, dict):

                num_channels = len(channels.keys())

            elif isinstance(channels, int):

                if loc_out is None:
                    loc_out = "io"
                    logging.warning(f"no location specified. data will be saved to {loc_out}")

                if loc_out[-1] != "/":
                    loc_out += "/"

                num_channels = channels
                channels = {i: f"{loc_out}ch{i}" for i in range(num_channels)}

            else:
                raise ValueError(f"please provide channels as int or dictionary instead of {type(channels)}.")

            if stack.shape[0] % num_channels != 0:
                logging.warning(
                    f"cannot divide frames into channel number: {stack.shape[0]} % {num_channels} != 0. "
                    f"May lead to unexpected behavior"
                )

            # Split the data into channels based on the given channel indices or names
            prep_data = {}
            for channel_key in channels.keys():
                prep_data[channel_key] = stack[channel_key::num_channels, :, :]

            # Subtract background if specified
            if subtract_background is not None:
                prep_data = self._subtract_background(prep_data, channels, subtract_background, subtract_func)

            # Rescale the prep_data if specified
            if (rescale is not None) and rescale != 1 and rescale != 1.0:
                self._rescale_data(prep_data, rescale, absolute_rescale=absolute_rescale)

            # Convert the prep_data type if specified
            if dtype is not None:
                prep_data = self._convert_dtype(prep_data, dtype=dtype)

            # Load the prep_data into memory if requested
            prep_data = prep_data if lazy else dask.compute(prep_data)[0]

            # Rename the channels in the output dictionary
            return {channels[i]: prep_data[i] for i in prep_data.keys()}

    @staticmethod
    def _convert_dtype(data, dtype):

        if dtype == np.uint:
            def func(chunk):
                return img_as_uint(chunk)

        else:
            def func(chunk):
                return chunk.astype(dtype)

            for k in data.keys():
                data[k] = data[k].map_blocks(lambda chunk: func(chunk), dtype=dtype)

        return data

    @staticmethod
    def _save(
            path: Union[str, Path], data: Union[np.ndarray, da.Array, dict], loc: str = '',
            chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced", chunks: Tuple[int, int, int] = None,
            compression: Literal['gzip', 'szip', 'lz4'] = None
    ):

        """Save the processed data to a specified path.

        Args:
            path: Path to save the processed data.
            data: Processed data to be saved.
            loc: Location of the dataset in an HDF5 file.
            chunk_strategy: Strategy used to infer appropriate chunk size.
            chunks: User-defined chunk size. Ignores chunk_strategy if provided.
            compression: Compression method to use when saving to HDF5 or TileDB.
        """

        io = IO()
        io.save(
            path=path, data=data, loc=loc, chunk_strategy=chunk_strategy, chunks=chunks, compression=compression
        )


class IO:

    def load(
            self, path: Union[str, Path], loc: str = '', sep: str = "_", z_slice: Tuple[int, int] = None,
            lazy: True = False, chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced",
            chunks: Tuple[int, int, int] = None
    ) -> Union[np.ndarray, da.Array]:

        """
        Loads data from a specified file or directory.

        Args:
            path: The path to the file or directory.
            loc: The location of the dataset in an HDF5 file.
            sep: Separator used for sorting file names.
            z_slice: Range of frames that are selective loaded.
            lazy: Flag to load the data on demand or to memory (lazy = False).
            chunk_strategy: Strategy to infer the chunks.
            chunks: User-defined chunk size. Ignores `chunk_strategy` if provided.

        Raises:
            ValueError: If the file format is not recognized.
            FileNotFoundError: If the specified file or folder cannot be found.

        """

        if z_slice is not None:
            if not isinstance(z_slice, (tuple, list)) or len(z_slice) != 2:
                raise ValueError("please provide z_slice as tuple or list of (z_start, z_end)")

        if isinstance(path, (str, Path)):
            path = Path(path)

            if path.suffix in [".tdb"]:
                data = self._load_tdb(
                    path, lazy=lazy, chunks=chunks, infer_strategy=chunk_strategy, z_slice=z_slice
                )

            elif path.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]:
                data = self._load_tiff(path, sep, lazy=lazy, infer_strategy=chunk_strategy, z_slice=z_slice)

            elif path.suffix in [".czi", ".CZI"]:
                data = self._load_czi(path, lazy=lazy, chunks=chunks, infer_strategy=chunk_strategy, z_slice=z_slice)

            elif path.suffix in [".h5", ".hdf5", ".H5", ".HDF5"]:
                data = self._load_h5(
                    path, loc=loc, lazy=lazy, infer_strategy=chunk_strategy, chunks=chunks, z_slice=z_slice
                )

            elif path.suffix in [".npy", ".NPY"]:
                data = self._load_npy(path, lazy=lazy, chunks=chunks, infer_strategy=chunk_strategy, z_slice=z_slice)

            elif path.suffix in [".csv", ".CSV"]:
                data = self._load_csv(path, chunks=chunks, infer_strategy=chunk_strategy, z_slice=z_slice)

            elif path.is_dir():

                # If the path is a directory, load multiple TIFF files
                files = [f for f in path.glob("*") if f.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]]
                if len(files) < 1:
                    raise FileNotFoundError("couldn't find files in folder. Recognized ext: [.tif, .tiff, .TIF, .TIFF]")

                else:
                    data = self._load_tiff(
                        path, sep, lazy=lazy, infer_strategy=chunk_strategy, chunks=chunks, z_slice=z_slice
                    )

            else:
                raise ValueError("Unrecognized file format! Choose one of [.tiff, .h5, .tdb, .czi, .npy, .csv]")

        elif isinstance(path, np.ndarray):

            if z_slice is not None:
                z0, z1 = z_slice
                path = path[z0:z1]

            if lazy:
                chunks = self.infer_chunks_from_array(arr=path, strategy=chunk_strategy, chunks=chunks)
                data = da.from_array(path, chunks=chunks)
            else:
                data = path

        elif isinstance(path, da.Array):
            if z_slice is not None:
                z0, z1 = z_slice
                path = path[z0:z1]

            if chunks is not None and path.chunks != chunks:
                data = da.rechunk(path, chunks=chunks)
            else:
                data = path

        else:
            raise ValueError(
                f"unrecognized file format! Choose a path (str, Path), a numpy/dask array {type(path)}"
            )

        return data

    def _load_npy(self, path: Union[Path, str], lazy=False, chunks=None, infer_strategy="balanced", z_slice=None):

        if isinstance(path, str):
            path = Path(path)

        # load data
        if path.is_dir():
            data = da.from_npy_stack(path.as_posix(), mmap_mode='r')

        elif path.is_file():
            data = np.load(path.as_posix(), mmap_mode='r')
        else:
            raise FileNotFoundError(f"{path} is neither directory nor file")

        # select frames
        if z_slice is not None:
            z0, z1 = z_slice
            data = data[z0:z1]

        # convert to lazy/not-lazy
        if lazy:
            chunks = self.infer_chunks_from_array(arr=data, strategy=infer_strategy, chunks=chunks)

            # re-chunk if necessary
            if isinstance(data, da.Array) and data.chunksize != chunks:
                data = data.rechunk(chunks=chunks)

            # convert to da.Array
            else:
                data = da.from_array(x=data, chunks=chunks)

        else:

            if isinstance(data, da.Array):
                data = data.compute()

        return data

    def _load_tdb(self, path, lazy=False, chunks=None, infer_strategy="balanced", z_slice=None):

        """
        Loads data from a TileDB file.

        Args:
            path (pathlib.Path): The path to the TileDB file.

        Returns:
            numpy.ndarray: The loaded data.

        """

        z0 = z1 = None
        if z_slice is not None:
            z0, z1 = z_slice

        if lazy:
            tdb = tiledb.open(path.as_posix(), "r")

            if z_slice is not None:
                tdb = tdb[z0:z1]

            chunks = self.infer_chunks_from_array(arr=tdb, strategy=infer_strategy, chunks=chunks)
            data = da.from_array(tdb, chunks=chunks)

        else:

            with tiledb.open(path.as_posix(), "r") as tdb:
                if z_slice is not None:
                    data = tdb[z0:z1]
                else:
                    data = tdb[:]  # Read all data from TileDB array

        return data

    def _load_h5(self, path, loc, lazy=False, chunks=None, infer_strategy="balanced", z_slice=None):

        """
        Loads data from an HDF5 file.

        Args:
            path (pathlib.Path): The path to the HDF5 file.
            loc (str): The location of the dataset in the HDF5 file.

        Returns:
            numpy.ndarray: The loaded data.

        """

        if loc is None:
            raise ValueError(f"Please provide the location of the dataset in the HDF5 file")

        z0 = z1 = None
        if z_slice is not None:
            z0, z1 = z_slice

        if lazy:
            data = h5py.File(path, "r")

            if loc not in data:
                raise ValueError(f"cannot find dataset in file ({path}): {list(data.keys())}")

            data = data[loc]

            if z_slice is not None:
                data = data[z0:z1]

            chunks = self.infer_chunks_from_array(arr=data, strategy=infer_strategy, chunks=chunks)
            data = da.from_array(data, chunks=chunks)

        else:
            with h5py.File(path, "r") as data:

                if loc not in data:
                    raise ValueError(f"cannot find dataset in file ({path}): {list(data.keys())}")

                if z_slice is not None:
                    data = data[loc][z0:z1]
                else:
                    data = data[loc][:]  # Read all data from HDF5 file

        return data

    def _load_csv(self, path, chunks=None, infer_strategy="balanced", z_slice=None):

        df = pd.read_csv(path)

        if isinstance(df, pd.Series):
            df = df.values

            if z_slice is not None:
                z0, z1 = z_slice
                df = df[z0:z1]

            chunks = self.infer_chunks_from_array(arr=df, strategy=infer_strategy, chunks=chunks)
            return da.from_array(df, chunks=chunks)

        return df

    def _load_czi(self, path, lazy=False, chunks=None, infer_strategy="balanced", z_slice=None):

        """
        Loads a CZI file from the specified path and returns the data.

        Args:
            path (str or pathlib.Path): The path to the CZI file.

        Returns:
            numpy.ndarray: The loaded data from the CZI file.

        """

        # Convert path to a pathlib.Path object if it's provided as a string
        path = Path(path) if isinstance(path, str) else path

        # Validate path
        assert isinstance(path, Path), "please provide 'path' as str or pathlib.Path"
        assert path.is_file(), f"cannot find file: {path}"

        # Read the CZI file using czifile
        data = czifile.imread(path.as_posix())

        if z_slice is not None:
            z0, z1 = z_slice
            data = data[z0:z1]

        # Remove single-dimensional entries from the shape of the data
        data = np.squeeze(data)

        # convert to dask array
        if lazy:
            chunks = self.infer_chunks_from_array(arr=data, strategy=infer_strategy, chunks=chunks)
            data = da.from_array(data, chunks=chunks)

        if len(data.shape) != 3:
            logging.warning(
                f"the dataset is not 3D but instead: {data.shape}. This will most likely create errors downstream in the pipeline."
            )

        return data

    @staticmethod
    def sort_alpha_numerical_names(file_names, sep="_"):
        """
        Sorts a list of file names in alphanumeric order based on a given separator.

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

    def _load_tiff(self, path, sep="_", lazy=False, chunks=None, infer_strategy="balanced", z_slice=None):

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

        z0 = z1 = None
        if z_slice is not None:
            z0, z1 = z_slice

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

            if z_slice is not None:
                files = files[z0:z1]

            # Read the TIFF files using dask.array and stack them
            stack = da.stack([dask_image.imread.imread(f.as_posix()) for f in files])
            stack = np.squeeze(stack)

            chunks = self.infer_chunks_from_array(stack, strategy=infer_strategy, chunks=chunks)
            stack = da.rechunk(stack, chunks=chunks)

            if len(stack.shape) != 3:
                raise NotImplementedError(
                    f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D"
                )

        elif path.is_file():
            # If the path is a file, load a single TIFF file

            if lazy:
                stack = tifffile.imread(path.as_posix())

                chunks = self.infer_chunks_from_array(stack, strategy=infer_strategy, chunks=chunks)
                stack = da.from_array(stack, chunks=chunks)

            else:
                stack = tifffile.imread(path.as_posix())

            if z_slice is not None:
                stack = stack[z0:z1]

        else:
            raise FileNotFoundError(f"cannot find directory or file: {path}")

        return stack

    def save(
            self, path: Union[str, Path], data: Union[np.ndarray, da.Array, dict], loc: str = None,
            chunks: Tuple[int, int, int] = None, chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced",
            compression: Literal['gzip', 'lz4', 'szip'] = None, overwrite: bool = False
    ) -> Union[str, Path, List[str]]:
        """
        Save data to a specified file format.

        Args:
            path: The path to the output file.
            data: Data in numpy/dask array format or a dictionary `{'channel name': arr}`.
            loc: Name of the dataset within the file (applicable only for HDF5 format).
            chunk_strategy: Strategy utilized to find optimal chunk sizes.
            chunks: User-defined chunk size. Ignores `chunk_strategy` if provided.
            compression: The compression method to be used when saving Dask arrays.
            overwrite: Flag to toggle overwriting of existing files.

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
            raise TypeError(f"please provide 'path' as str or pathlib.Path data type not: {type(path)}")

        # choose default output location if None is provided
        if loc is None:
            loc = "io"
            logging.warning(f"choosing automatic dataset name: {loc}.")

        # Check if the data is a dictionary or data array, otherwise raise an error
        if isinstance(data, (np.ndarray, da.Array)):
            data = {loc: data}

        elif not isinstance(data, dict):
            raise TypeError("please provide data as dict of {channel_name:array} or np.ndarray")

        saved_paths = []  # Initialize an empty list to store the paths of the saved files
        for k in data.keys():

            # Check if the channel is a numpy.ndarray or a dask.array.Array, otherwise raise an error
            arr = data[k]
            if not isinstance(arr, (np.ndarray, da.Array)):
                raise TypeError("please provide data as either 'numpy.ndarray' or 'da.array.Array'")

            # infer chunks if necessary
            if chunks is None and chunk_strategy is not None:
                chunks = self.infer_chunks(arr.shape, arr.dtype, strategy=chunk_strategy)
                logging.warning(f"inferred chunk size: {chunks} (chunks > {chunks}, strategy > {chunk_strategy})")

            # infer compression
            if compression == "infer":
                size = arr.size * arr.itemsize
                if size > 10e9 and path.suffix in [".h5", ".hdf5"]:
                    compression = "gzip"
                    logging.warning(f"inferred compression: {compression}")
                else:
                    compression = None

            if path.suffix in [".h5", ".hdf5"]:
                # Save as HDF5 format

                fpath = path

                # create dataset location
                if isinstance(loc, dict):
                    loc_out = loc[k]
                elif isinstance(loc, str):
                    loc_out = k
                else:
                    raise ValueError(f"Please provide 'loc' as None, str, or dict data type not: {type(loc)}")

                self.exists_and_clean(fpath, loc=loc_out, overwrite=overwrite)

                if isinstance(arr, da.Array):

                    # Save Dask array
                    with ProgressBar(minimum=10, dt=1):
                        da.to_hdf5(fpath, loc_out, arr, chunks=chunks, compression=compression, shuffle=False)

                else:
                    # Save NumPy array
                    with h5py.File(fpath, "a") as f:

                        if loc_out in f:
                            raise ValueError(f"dataset already exists in file {fpath}:{loc_out}\n{list(f.keys())}")

                        _ = f.create_dataset(
                            name=loc_out, data=arr, shape=arr.shape, chunks=chunks, compression=compression,
                            shuffle=False, dtype=arr.dtype
                        )

                saved_paths.append(fpath)
                logging.info(f"dataset saved to {fpath}::{loc_out}")

            elif path.suffix == ".tdb":
                # Save as TileDB format

                if isinstance(arr, np.ndarray):
                    arr = da.from_array(arr, chunks=chunks if chunks is not None else "auto")

                fpath = path.with_suffix(f".{k}.tdb") if len(data.keys()) > 1 else path
                self.exists_and_clean(fpath, overwrite=overwrite)
                with ProgressBar(minimum=10, dt=1):
                    da.to_tiledb(arr, fpath.as_posix(), compute=True)

                saved_paths.append(fpath)
                logging.info(f"dataset saved to {fpath}")

            elif path.suffix in [".tiff", ".TIFF", ".tif", ".TIF"]:
                # Save as TIFF format

                fpath = path.with_suffix(f".{k}.tiff") if len(data.keys()) > 1 else path
                self.exists_and_clean(fpath, overwrite=overwrite)

                tifffile.imwrite(fpath, data=arr)

                saved_paths.append(fpath)
                logging.info(f"saved data to {fpath}")

            elif path.suffix in [".czi", ".CZI"]:
                raise NotImplementedError("currently we are not aware that python can save images in .czi format.")

            elif path.suffix in [".npy", ".NPY"]:

                fpath = path.with_suffix(f".{k}.npy") if len(data.keys()) > 1 else path
                self.exists_and_clean(fpath, overwrite=overwrite)

                if isinstance(arr, np.ndarray):
                    arr = da.from_array(arr)

                with ProgressBar(minimum=10, dt=1):
                    da.to_npy_stack(fpath, x=arr, axis=0)

                saved_paths.append(fpath)
                logging.info(f"saved data to {fpath}")

            elif path.suffix in [".avi", ".AVI"]:

                import cv2

                fpath = path.with_suffix(f".{k}.avi") if len(data.keys()) > 1 else path
                self.exists_and_clean(fpath, overwrite=overwrite)

                # Get the dimensions of the numpy array
                frames, X, Y = arr.shape

                # Define the codec and create a VideoWriter object

                if isinstance(fpath, Path):
                    fpath = fpath.as_posix()

                out = cv2.VideoWriter(
                    fpath, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=16, frameSize=(X, Y), isColor=True
                )

                for i in tqdm(range(frames)):
                    # Get the current frame
                    frame = arr[i]

                    # Since the array has only one channel, convert it to a 3-channel array (BGR)
                    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    # Write the frame to the AVI file
                    out.write(frame)

                # Release the VideoWriter
                out.release()

                saved_paths.append(fpath)
                logging.info(f"saved data to {fpath}")

            else:
                raise TypeError("please provide output format as .h5, .tdb, .npy, .avi or .tiff file")

        return saved_paths if len(saved_paths) > 1 else saved_paths[0]  # Return the list of saved file paths

    @staticmethod
    def exists_and_clean(path, loc="", overwrite=False):

        path = Path(path)

        if not path.exists():
            return True

        # H5
        if path.suffix in (".h5", ".hdf5"):

            with h5py.File(path, "a") as f:

                if loc in ("", None):
                    raise ValueError(f"Please provide a valid h5 dataset location instead of {loc}")

                if loc in f:

                    logging.warning(f"deleting previous result: {path} [{loc}]")

                    if overwrite:
                        del f[loc]
                    else:
                        raise FileExistsError(
                            f"output dataset exists {path} [{loc}]. "
                            f"Please choose a different dataset or set 'overwrite=True'"
                        )

        # Everything else
        else:

            if overwrite:

                logging.warning(f"deleting previous result: {path}")

                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

            else:
                raise FileExistsError(
                    f"output exists ({path}). Please choose a different output or set 'overwrite=True'"
                )

    @staticmethod
    def infer_chunks(shape, dtype, strategy="balanced", chunk_bytes=int(1e6), chunks=None):

        """
        Infer the chunks for the input data.
        """

        if chunks is not None and isinstance(chunks, (tuple, list)):
            return chunks

        if chunks is None and strategy is None:
            return None

        Z = X = Y = 1
        if len(shape) > 3:
            raise ValueError(
                f"Chunk inference is only implemented for 1D, 2D and 3D arrays, not {len(shape)}D."
                f"Please provide a user-defined chunk size ('chunks')"
            )
        elif len(shape) == 3:
            Z, X, Y = shape
        elif len(shape) == 2:
            Z, X = shape
        elif len(shape) == 1:
            Z = shape

        item_size = np.dtype(dtype).itemsize

        if strategy == "balanced":

            exp = 0
            while (2 ** exp) ** 3 * item_size < chunk_bytes:
                exp += 1

            c = int(2 ** exp)
            cz = min(Z, c)
            cx = min(X, c)
            cy = min(Y, c)

        elif strategy == "Z":

            exp = 0
            while Z * (2 ** exp) ** 2 * item_size < chunk_bytes:
                exp += 1

            c = int(2 ** exp)
            cz = Z
            cx = min(X, c)
            cy = min(Y, c)

        elif strategy == "XY":

            exp = 0
            while X * Y * (2 ** exp) * item_size < chunk_bytes:
                exp += 1

            c = int(2 ** exp)
            cz = min(Z, c)
            cx = X
            cy = Y

        else:
            raise ValueError(f"Unknown strategy, please provide one of 'balanced', 'Z' or 'XY'")

        if len(shape) == 3:
            return cz, cx, cy
        elif len(shape) == 2:
            return cz, cx
        elif len(shape) == 1:
            return cz

    def infer_chunks_from_array(self, arr, strategy="balanced", chunk_bytes=int(1e6), chunks=None):
        return self.infer_chunks(arr.shape, arr.dtype, strategy=strategy, chunk_bytes=chunk_bytes, chunks=chunks)


class MotionCorrection:
    # noinspection GrazieInspection
    """ Class for performing motion correction based on the Jax-accelerated implementation of NoRMCorre.

        Args:
            working_directory: Working directory for temporary files.
                If not provided, the temporary directory is created.
            logging_level: Sets the level at which information is logged to the console as an
                integer value. The built-in levels in the logging module are, in increasing order of severity:
                debug (10), info (20), warning (30), error (40), critical (50).

        .. note::

            For more information see the `accelerated <https://github.com/apasarkar/jnormcorre>`_ (used here),
            `original implementation <https://github.com/flatironinstitute/NoRMCorre>`_ and the associated
            publication Pnevmatikakis et al. 2017 [#normcorre]_

        .. caution::

            Non-rigid motion correction is not always necessary. Sometimes, rigid motion correction will be sufficient,
            and it will lead to significant performance gains in terms of speed. Check your data before and after rigid
            motion correction to decide what is best (pw_rigid flag; see below).

        **Example**::

            mc = MotionCorrection()
            mc.run('path/to/file.h5', loc='data/ch0')
            mc.save(output='path/to/file.h5', loc='mc/ch0')

        .. rubric:: Footnotes

        .. [#normcorre] Pnevmatikakis EA, Giovannucci A. NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. Journal of neuroscience methods. 2017 Nov 1;291:83-94. `https://doi.org/10.1016/j.jneumeth.2017.07.031 <https://doi.org/10.1016/j.jneumeth.2017.07.031>`_.

        """

    def __init__(self, working_directory: Union[str, Path] = None, logging_level: int = logging.INFO):

        self.shifts = None
        logging.basicConfig(level=logging_level)

        self.path = None
        self.working_directory = working_directory
        self.tempdir = None

        self.io = IO()

        # output location
        self.mmap_path = None
        self.tiff_path = None
        self.temp_path = None
        self.frames = self.X = self.Y = None

    def run(
            self, path: Union[str, Path], loc: str = "", max_shifts: Tuple[int, int] = (50, 50), niter_rig: int = 3,
            splits_rig: int = 14, num_splits_to_process_rig: int = None, strides: Tuple[int, int] = (48, 48),
            overlaps: Tuple[int, int] = (24, 24), pw_rigid: bool = False, splits_els: int = 14,
            num_splits_to_process_els: int = None, upsample_factor_grid: int = 4, max_deviation_rigid: int = 3,
            nonneg_movie: bool = True, gSig_filt: Tuple[int, int] = (20, 20), bigtiff: bool = True
    ) -> None:
        """Reduces motion artifacts by performing piecewise rigid motion correction.

        Args:
            path: The input data to be motion corrected.
            loc: The dataset name in the .h5 file the data is stored in. Only relevant if
                path is an .h5 file.
            max_shifts: A tuple specifying the maximum allowed rigid shift in pixels.
            niter_rig: The maximum number of iterations for rigid motion correction. More iterations can improve
                motion correction quality, but increases runtime.
            splits_rig: The number of splits to parallelize the motion correction for rigid correction.
            num_splits_to_process_rig: A list specifying the number of splits to process at each iteration for rigid correction.
            strides: A tuple specifying the intervals at which patches are laid out for motion correction.
            overlaps: A tuple specifying the overlaps between patches for motion correction.
            pw_rigid: A boolean indicating whether to perform piecewise or standard rigid motion correction.

            splits_els: The number of splits to parallelize the motion correction for elastic correction.
            num_splits_to_process_els: A list specifying the number of splits to process at each iteration for elastic correction.

            upsample_factor_grid: The upsample factor for the grid in elastic motion correction.
            max_deviation_rigid: The maximum deviation from rigid motion allowed in pixels.
            nonneg_movie: A boolean indicating whether to enforce non-negativity in the motion corrected movie.
            gSig_filt: A tuple specifying the size of the Gaussian filter for filtering the movie.
            bigtiff: A boolean indicating whether to save the motion corrected movie as a BigTIFF file.
                Prevents errors when correcting videos dimensions exceeding the capabilities of the standard tiff format.

        """

        from jnormcorre import motion_correction

        path = self._validate_input(path, loc=loc)
        self.path = path

        # validate parameters
        if max_shifts[0] >= int(self.X / 2):
            max_shifts_adj = int(self.X / 2) - 1
            logging.warning(
                f"dimension 1 of max_shifts parameter > 1/2 img.X ({max_shifts[0]}>{int(self.X / 2)}."
                f"Automatically adjusting to: {max_shifts_adj}"
            )
            max_shifts = tuple((max_shifts_adj, max_shifts[1]))

        if max_shifts[1] >= int(self.Y / 2):
            max_shifts_adj = int(self.Y / 2) - 1
            logging.warning(
                f"dimension 1 of max_shifts parameter > 1/2 img.X ({max_shifts[1]}>{int(self.Y / 2)}."
                f"Automatically adjusting to: {max_shifts_adj}"
            )
            max_shifts = tuple((max_shifts[0], max_shifts_adj))

        # Create MotionCorrect instance
        mc = motion_correction.MotionCorrect(
            path, var_name_hdf5=loc, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
            num_splits_to_process_rig=num_splits_to_process_rig, strides=strides, overlaps=overlaps, pw_rigid=pw_rigid,
            splits_els=splits_els, num_splits_to_process_els=num_splits_to_process_els,
            upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
            nonneg_movie=nonneg_movie, gSig_filt=gSig_filt, bigtiff=bigtiff
        )

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

    def _validate_input(self, path: Union[str, Path], loc: str) -> Path:
        """
        Validate and process the input for motion correction.

        Args:
            path (Union[str, Path, np.ndarray]): Input data for motion correction.
            loc (str): Dataset name in case of input being an HDF5 file.

        Returns:
            Union[Path, np.ndarray]: Validated and processed input.

        Raises:
            FileNotFoundError: If the input file is not found.
            ValueError: If the input format is not supported or required arguments are missing.
            NotImplementedError: If the input format is not implemented.

        Notes:
            - A temporary .tiff file is created if the input is an array, which needs to be deleted later using the 'clean_up()' method.
        """

        if isinstance(path, (str, Path)):
            # If input is a string or Path object

            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"cannot find input_: {path}")

            if path.suffix in [".h5", ".hdf5"]:
                # If input is an HDF5 file

                if loc is None:
                    raise ValueError("Please provide 'loc' argument when providing .h5 file as data input.")

                with h5py.File(path.as_posix(), "a") as f:
                    if loc not in f:
                        raise ValueError(f"cannot find dataset {loc} in provided in {path}.")

                    self.frames, self.X, self.Y = f[loc].shape

                return path

            elif path.suffix in [".tiff", ".TIFF", ".tif", ".TIF"]:
                # If input is a TIFF file

                with tifffile.TiffFile(path.as_posix()) as tif:

                    self.frames = len(tif.pages)  # number of pages in the file
                    page = tif.pages[0]  # get shape and dtype of image in first page
                    self.X, self.Y = page.shape

                return path

            else:
                raise ValueError(f"unknown input type. Please provide .h5 or .tiff file.")

        elif isinstance(path, np.ndarray):
            # If input is a ndarray create a temporary TIFF file to run the motion correction on

            self.frames, self.X, self.Y = path.shape

            logging.warning(
                "caiman.motion_correction requires a .tiff or .h5 file to perform the correction. A temporary .tiff file is created which needs to be deleted later by calling the 'clean_up()' method of this module."
            )

            if self.working_directory is None:
                self.working_directory = tempfile.TemporaryDirectory()

            if isinstance(self.working_directory, tempfile.TemporaryDirectory):
                temp_dir = Path(self.working_directory.name)
            else:
                temp_dir = Path(self.working_directory)

            assert temp_dir.exists(), f"working directory doesn't exist: {temp_dir}"

            self.temp_path = temp_dir.joinpath(f"temp.tiff")
            tifffile.imwrite(self.temp_path.as_posix(), path)

            return self.temp_path

        else:
            raise ValueError(f"please provide input_ as one of: np.ndarray, str, Path")

    def _clean_up(self):
        """
        Clean up temporary files and resources associated with motion correction.

        Notes:
            - This method should be called after motion correction is completed to remove temporary files and resources.

        Raises:
            FileNotFoundError: If the input file is not found.

        """

        # Remove mmap result
        if self.mmap_path is not None and Path(self.mmap_path).is_file():
            os.remove(self.mmap_path)

        if self.tiff_path is not None and Path(self.tiff_path).is_file():
            os.remove(self.tiff_path)

        # Remove temp .tiff if necessary
        if self.working_directory is not None:
            if self.temp_path is not None and self.temp_path.is_file():
                self.temp_path.unlink()

    @staticmethod
    @deprecated("use caiman's built-in file splitting function instead")
    def _get_frames_per_file(input_, frames_per_file, loc=None):

        if frames_per_file == "auto":

            (Z, X, Y), chunksize, dtype = get_data_dimensions(input_, loc=loc, return_dtype=True)
            byte_num = np.dtype(dtype).itemsize
            array_size = Z * X * Y * byte_num

            ram_size = psutil.virtual_memory().total

            if ram_size < array_size * 2:
                logging.warning(
                    f"available RAM ({ram_size}) is smaller than twice the data size ({array_size}. Automatically splitting files into smaller fragments. Might lead to unexpected behavior on the boundary between fragments."
                )
                frames_per_file = int(Z / np.floor(array_size / ram_size) / 2)

        elif isinstance(frames_per_file, int):
            pass

        elif isinstance(frames_per_file, float):
            frames_per_file = int(frames_per_file)

        else:
            raise ValueError(f"Please provide one of these options for 'split_file' flag: None, 'auto', int, float")

        return frames_per_file

    def save(
            self, output: Union[str, Path] = None, loc: str = "mc/ch0",
            chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced", chunks: Tuple[int, int, int] = None,
            compression: Literal['gzip', 'lzf', 'szip'] = None, remove_intermediate: bool = True
    ) -> Union[np.ndarray, None]:

        """
        Retrieve the motion-corrected data and optionally save it to a file.

        Args:
            output: Output file path where the data should be saved.
            loc: Location within the HDF5 file to save the data (required when output is an HDF5 file).
            chunk_strategy: Chunk strategy to use when saving to an HDF5 file.
            chunks: Chunk shape for creating a dask array when saving to an HDF5 file.
            compression: Compression algorithm to use when saving to an HDF5 file.
            remove_intermediate: Whether to remove the intermediate files associated with motion correction after retrieving the data.

        Notes:
            - This method should be called after motion correction is completed by using the `run()` function.
            - If `output` is specified, the motion-corrected data is saved to the specified file using the `IO` class.
            - If `remove_intermediate` is set to `True`, the mmap file associated with motion correction is deleted after retrieving the data.

        """

        # enforce path
        output = Path(output) if output is not None else output

        # Check if the tiff output is available
        tiff_path = self.tiff_path
        if tiff_path is None:
            raise ValueError("tiff_path is None. Please compute motion correction first by using the 'run()' function")

        tiff_path = Path(tiff_path)
        if not tiff_path.is_file():
            raise FileNotFoundError(
                f"could not find tiff file: {tiff_path}. Maybe the 'clean_up()' function was called too early?"
            )

        data = tifffile.imread(tiff_path.as_posix())

        # If output is None, return the motion-corrected data as a NumPy array
        if output is None:
            return data

        elif isinstance(output, (str, Path)):

            output = Path(output) if isinstance(output, Path) else output

            # Save the motion-corrected data to the output file using the I/O module
            self.io.save(
                output, data=data, loc=loc, chunk_strategy=chunk_strategy, chunks=chunks, compression=compression
            )

        else:
            raise ValueError(f"please provide output as None, str or pathlib.Path instead of {output}")

        # If remove_mmap is True, delete the mmap file associated with motion correction
        if remove_intermediate:
            self._clean_up()


class Delta:
    """ Provides methods for bleach correction of input data.

    Args:
        data: The input data to be processed.
        loc: The location of the data in the HDF5 file. This parameter is optional and only applicable when data has the .h5 extension.

    **Example**::

        delta = Delta('/path/to/input.h5', loc="data/ch0")
        delta.run(window=10, method="dF")
        delta.save(output_path='/path/to/input.h5', loc="df/ch0", chunk_strategy="balanced", compression="gzip")

    """

    def __init__(self, data: Union[str, Path, np.ndarray, da.Array], loc: str = None):

        # Convert the input to a Path object if it is a string
        self.data = Path(data) if isinstance(data, str) else data
        self.result = None
        self.prep_data = None

        # Get the dimensions and chunk size of the input data
        self.dim, self.chunksize = get_data_dimensions(self.data, loc=loc)

        # The location of the data in the HDF5 file (optional, only applicable for .h5 files)
        self.loc = loc

    def run(
            self, window: int, method: Literal['background', 'dF', 'dFF'] = "dF",
            chunk_strategy: Literal['balanced', 'XY', 'Z'] = "Z", chunks=None, overwrite_first_frame: bool = True,
            compute: bool = False
    ) -> Union[np.ndarray, da.Array]:
        # noinspection GrazieInspection
        """
                Performs bleach correction on the input data using specified methods and parameters.

                Args:
                    window: The size of the window for the minimum filter.
                    method: The method to use for delta calculation.
                    chunk_strategy: Strategy to infer appropriate chunk size
                    chunks: User-defined chunk size (ignores inference strategy).
                    overwrite_first_frame: A flag indicating whether to overwrite the values of the first frame with
                        the second frame after delta calculation.
                    compute: A flag indicating whether to return a dask array (False) or a NumPy array (True).

                Raises:
                    ValueError: If the input data type is not recognized.

                Notes:
                    The function supports different types of input data, including numpy arrays, file paths
                    (specifically .tdb and .h5 files), and Dask arrays. It also handles parallel execution for large
                    datasets, especially when input is a .tdb file.

                .. warning::

                    For .tdb files as input, this function will overwrite the provided file.

                """

        # Prepare the data for processing
        data = self._prepare_data(self.data, loc=self.loc, chunk_strategy=chunk_strategy, chunks=chunks)
        self.prep_data = data

        # Sequential execution
        if isinstance(data, np.ndarray):
            # Calculate delta using the minimum filter on the input data
            res = self._calculate_delta_min_filter(data, window, method=method, inplace=False)

        # Parallel from .tdb file
        elif isinstance(data, (str, Path)) and Path(data).suffix in [".tdb", ".TDB"]:
            # Warning message for overwriting the .tdb file
            logging.warning("This function will overwrite the provided .tdb file!")

            # Define a wrapper function for parallel execution
            calculate_delta_min_filter = self._calculate_delta_min_filter

            def wrapper(_path, ranges):
                (_x0, x1), (_y0, y1) = ranges

                # Open the TileDB array and load the specified range
                with tiledb.open(_path, mode="r") as tdb:
                    _data = tdb[:, _x0:x1, _y0:y1]

                _res = calculate_delta_min_filter(_data, window, method=method, inplace=False)

                # Overwrite the range with the calculated delta values
                with tiledb.open(path, mode="w") as tdb:
                    tdb[:, _x0:x1, _y0:y1] = _res

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
                            futures.append(client.submit(wrapper, (path, range_)))

                    # Gather the results from parallel executions
                    client.gather(futures)

            # Load the modified .tdb file into memory
            io = IO()
            res = io.load(Path(path))

        elif isinstance(data, da.Array):
            # Calculate delta using Dask array for lazy loading and computation

            def xy_delta(x):
                return self._calculate_delta_min_filter(x, window=window, method=method, inplace=False)

            res = data.map_blocks(xy_delta, dtype=float)

        else:
            raise ValueError(f"Input data type not recognized: {type(data)}")

        # Overwrite the first frame with the second frame if required
        if overwrite_first_frame:
            res[0, :, :] = res[1, :, :]

        # convert to numpy array if requested
        if compute and isinstance(res, da.Array):
            res = res.compute()

        self.result = res
        return res

    def save(
            self, output_path: Union[str, Path], loc: str = "df",
            chunk_strategy: Literal['balanced', 'XY', 'Z'] = "balanced", chunks: Tuple[int, int, int] = None,
            compression: Literal['gzip', 'szip', 'lz4'] = None, overwrite: bool = False
    ):
        # noinspection GrazieInspection
        """
                Saves the result data to a specified file.

                This method wraps the functionality of the `IO` class's `save` method, allowing for saving the data in different
                chunk strategies and with various compression methods.

                Parameters:
                    output_path: Path to the file where the data will be saved.
                    loc: The dataset name within the HDF5 file to store the data.
                    chunk_strategy: Strategy to infer appropriate chunk size when saving.
                    chunks: User-defined chunk size. Ignores `chunk_strategy`.
                    compression: Compression method to use for storing the data.
                    overwrite: Whether to overwrite the file if it already exists.

                .. note::

                    The 'loc' parameter defaults to 'df', and the 'chunk_strategy' defaults to 'balanced'. If 'chunks' is
                    not specified, the method will infer appropriate chunk sizes based on the strategy. The 'overwrite' flag is
                    set to False by default, ensuring that existing files are not overwritten unless explicitly intended.
                """
        io = IO()
        io.save(
            output_path, data=self.result, loc=loc, chunk_strategy=chunk_strategy, chunks=chunks,
            compression=compression,
            overwrite=overwrite
        )

    @staticmethod
    def _prepare_data(
            data: Union[str, Path, np.ndarray, da.Array], chunk_strategy: Literal['balanced', 'XY', 'Z'] = 'Z',
            chunks: Tuple[int, int, int] = None, loc: str = ''
    ) -> da.Array:

        """ Preprocesses the input data by converting it to a dask compatible format or TileDB array and optionally
            loading it into memory or creating a Dask array.

        Args:
            data: A data object or file path to the data to be processed.
            chunk_strategy: The strategy to use for inferring appropriate chunk size of the input data.
            chunks: User-defined chunk size for the input data. Ignores `chunk_strategy`
            loc: Dataset name for the input data if a .h5 file is provided.

        Raises:
            TypeError: If the input data type is not recognized.
        """

        io = IO()
        if isinstance(data, Path):

            data = io.load(data, loc=loc, chunk_strategy=chunk_strategy, chunks=chunks)
            data = data.astype(int)
            return data

        elif isinstance(data, (np.ndarray, da.Array)):

            chunks = io.infer_chunks_from_array(arr=data, strategy=chunk_strategy, chunks=chunks)

            if not isinstance(data, da.Array):
                data = da.from_array(data, chunks=chunks)

            if data.chunks != chunks:
                data = data.rechunk(chunks)

            data = data.astype(int)  # TODO this should not be hardcoded

            return data

        else:
            raise TypeError(f"do not recognize data type: {type(data)}")

    @staticmethod
    @deprecated(reason="faster implementation but superseded by: calculate_background_even_faster")
    def _calculate_background_pandas(
            arr: np.ndarray, window: int, method="dF", inplace: bool = True
    ) -> np.ndarray:

        res = None

        if len(np.squeeze(arr)) < 2:
            arr = np.expand_dims(arr, axis=0)

        arr = np.atleast_3d(arr)

        if not inplace:
            res = np.zeros(arr.shape, arr.dtype)

        # define the possible methods
        methods = {"background": lambda _x, _background: _background, "dF": lambda _x, _background: _x - _background,
                   "dFF": lambda _x, _background: np.divide(_x - _background, _background)}

        if method not in methods.keys():
            raise ValueError(
                f"please provide a valid argument for 'method'; one of : {methods.keys()}"
            )

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
    def _calculate_delta_min_filter(
            arr: np.ndarray, window: int, method: Literal['background', 'dF', 'dFF'] = "dF", inplace: bool = False
    ) -> np.ndarray:
        # noinspection GrazieInspection
        """
                Calculates a delta value based on a specified method, applied to a 3D numpy array with a minimum filter over a window.

                This method applies a minimum filter across a window of specified size to each pixel in the input array.
                The delta is then calculated based on the selected method, either as a background value, a difference from the background (dF),
                or a relative difference from the background (dFF).

                Parameters:
                    arr: A 3D numpy array to which the delta min filter will be applied.
                    window: The size of the window over which the minimum filter is applied.
                    method: Method for calculating delta.
                    inplace: If True, the operation modifies the array in place.

                Returns:
                    np.ndarray: A numpy array of the same shape as the input, containing the calculated delta values.

                Raises:
                    ValueError: If an invalid method is specified.

                .. note::
                    The 'background' method returns the background value itself.
                    The 'dF' method returns the difference from the background.
                    The 'dFF' method returns the relative difference from the background.

                Example:
                    >>> rand_data = np.random.rand(10, 10, 10)
                    >>> result = Delta._calculate_delta_min_filter(rand_data, window=5, method='dF')
                """

        original_dims = arr.shape
        res = None

        # Ensure array is at least 3D
        if len(np.squeeze(arr)) < 2:
            arr = np.expand_dims(arr, axis=0)  # necessary to preserve order in case of 1D array

        arr = np.atleast_3d(arr)

        # choose delta function
        methods = {"background": lambda _x, _background: _background, "dF": lambda _x, _background: _x - _background,
                   "dFF": lambda _x, _background: np.divide(_x - _background, _background)}
        if method not in methods.keys():
            raise ValueError(
                f"please provide a valid argument for 'method'; one of : {methods.keys()}"
            )

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
                shift = int(window / 2)
                z_padded = np.pad(z, pad_width=(shift, shift), mode='edge')
                z_even = z_padded[::2]
                z_odd = z_padded[1::2]

                # MIN = minimum_filter1d(z_padded, size=window+1, mode="nearest", origin=0)
                MIN_even = minimum_filter1d(z_even, size=window + 1, mode="nearest", origin=0)
                MIN_odd = minimum_filter1d(z_odd, size=window + 1, mode="nearest", origin=0)

                # Duplicate each value in the even and odd series to make them the same length as the original series
                MIN_even_expanded = np.repeat(MIN_even, 2)[:len(z_padded)]
                MIN_odd_expanded = np.repeat(MIN_odd, 2)[:len(z_padded)]

                # padding
                if len(MIN_even_expanded) < len(z_padded):
                    MIN_even_expanded = np.pad(
                        MIN_even_expanded, pad_width=(0, len(z_padded) - len(MIN_even_expanded)), mode="edge"
                    )

                if len(MIN_odd_expanded) < len(z_padded):
                    MIN_odd_expanded = np.pad(
                        MIN_odd_expanded, pad_width=(0, len(z_padded) - len(MIN_odd_expanded)), mode="edge"
                    )

                # Get the maximum value at each point from the two expanded series to get the new baseline
                MIN = np.maximum(MIN_even_expanded, MIN_odd_expanded)

                # Shift the minimum signal by window/2 and take the max of the two signals
                background = MIN[shift:-shift]

                if inplace:
                    arr[:, x, y] = delta_func(z, background)
                else:
                    res[:, x, y] = delta_func(z, background)

        if inplace:
            res = arr

        # restore initial dimensions
        res = np.reshape(res, original_dims)

        return res

    def plot(self, pixels: Union[Tuple[int, int], List[Tuple[int, int]]], show_original: bool = False,
             separate_panels: bool = False, twin_y_axis: bool = False, alpha: float = 1,
             figsize: Tuple[int, int] = (10, 5), colors: List = None, labels: List[str] = None):
        """
        Render a plot based on delta data.

        This method creates a plot based on the delta data. It allows for customization through various parameters,
        including whether to plot the original data, whether to show separate plots, the size of the figure, etc.

        Args:
            pixels: The pixel(s) to plot. E.g. (10, 10) or [(5, 5), (10, 10)]
            show_original: A boolean to decide whether to plot the original data.
            separate_panels: A boolean to decide whether to show separate plots for each pixel.
            figsize: Dimensions of the figure (width x height)
            twin_y_axis: Show original and subtracted (Delta) pixel trace on separate y-axes.
            alpha: The alpha value for the plot line.
            colors: List of colors used for plotting.
            labels: List of labels for time series (one for each pixel)

        Returns:
          A matplotlib figure object containing the generated plot.
        """

        # parameter quality control
        if self.result is None:
            raise RuntimeError("please subtract delta first by using the 'run' function of this class first.")

        if isinstance(pixels, tuple):
            pixels = [pixels]

        if colors is None:
            import seaborn as sns
            colors = sns.color_palette("husl", n_colors=len(pixels))

        if labels is not None and len(pixels) != len(labels):
            logging.warning(
                f"received incorrect number of labels: len(labels) {len(labels)} vs. len(pixels) {len(pixels)}")
            labels = None

        if labels is None:
            labels = [f"pixel {x}x{y}" for x, y in pixels]

        # get data
        delta_trace = self.result
        original = self.prep_data

        # Create figure layout
        if separate_panels:
            fig, axx = plt.subplots(len(pixels), figsize=figsize)

            if len(pixels) > 1:
                axx = list(axx.flatten())
            else:
                axx = [axx]

            twin_axx = [ax.twinx() for ax in axx]

        else:
            fig, ax = plt.subplots(figsize=figsize)
            axx = [ax for _ in range(len(pixels))]
            twinned_axis = ax.twinx()
            twin_axx = [twinned_axis for _ in range(len(pixels))]

        # Plot pixels
        for i, ((x, y), color, ax, twinned_axis, lbl) in enumerate(zip(pixels, colors, axx, twin_axx, labels)):
            ax.plot(delta_trace[:, x, y], color=color, label=lbl)

            if show_original and twin_y_axis:
                twinned_axis.plot(original[:, x, y], color=color, linestyle="--", alpha=alpha)
                twinned_axis.set_ylabel("Intensity")
            elif show_original:
                ax.plot(original[:, x, y], color=color, linestyle="--", alpha=alpha)

            # add labels
            ax.set_xlabel("frame")
            ax.set_ylabel(r'$\Delta$ Intensity')
            ax.legend()

        return fig


class XII:

    def __init__(self, file_path, dataset_name, num_channels=1, sampling_rate=None, channel_names=None):
        self.container = self.load_xii(file_path, dataset_name, num_channels, sampling_rate, channel_names)

    @staticmethod
    def load_xii(file_path, dataset_name, num_channels=1, sampling_rate=None, channel_names=None):

        # define sampling rate
        if sampling_rate is None:
            sampling_rate = 1

        elif isinstance(sampling_rate, (float, int)):
            sampling_rate = float(sampling_rate)

        elif isinstance(sampling_rate, str):

            units = OrderedDict(
                [("ps", 1e-12), ("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("s", 1), ("min", 60), ("h", 60 * 60)]
            )

            found_unit = False
            for key, value in units.items():

                if sampling_rate.endswith(key):
                    sampling_rate = sampling_rate.replace(key, "")
                    sampling_rate = float(sampling_rate) * value
                    found_unit = True

                    break

            if not found_unit:
                raise ValueError(
                    f"when providing the sampling_rate as string, the value has to end in one of these units: {units.keys()}"
                )

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
                idx = pd.RangeIndex(timestep * ch, timestep * ch + len(data_ch) * timestep, timestep)
            elif isinstance(timestep, float):
                idx = pd.Index(np.arange(timestep * ch, timestep * ch + len(data_ch) * timestep, timestep))
            else:
                raise ValueError(
                    f"sampling_rate should be able to be cast to int or float instead of: {type(timestep)}"
                )

            data_ch = pd.Series(data_ch, index=idx)
            if channel_names is None:
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

        idx = np.arange(offset_start + idx_channel, offset_start + len(video) * 2 - offset_stop, num_channels)

        if len(idx) != len(video):
            raise ValueError(
                f"video length and indices don't align: video ({len(video)}) vs. idx ({len(idx)}). \n{idx}"
            )

        mapping = timing.to_dict()
        idx = pd.Index([np.round(mapping[id_], decimals=3) for id_ in idx])

        return idx, mapping

    def show(
            self, dataset_name, mapping, viewer=None, viewer1d=None, down_sample=100, colormap=None, window=160,
            y_label="XII", x_label="step"
    ):

        # todo: test with Video

        import napari_plot
        from napari_plot._qt.qt_viewer import QtViewer
        from napari.utils.events import Event

        qt_viewer = None
        xii = self.container[dataset_name][::down_sample]

        if viewer1d is None:
            v1d = napari_plot.ViewerModel1D()
            qt_viewer = QtViewer(v1d)
        else:
            v1d = viewer1d

        v1d.axis.y_label = y_label
        v1d.axis.x_label = x_label
        v1d.text_overlay.visible = True
        v1d.text_overlay.position = "top_right"

        # create attachable qt viewer
        X, Y = xii.index, xii.values
        line = v1d.add_line(np.c_[X, Y], name=y_label, color=colormap)

        def update_line(event: Event):
            Z, _, _ = event.value
            z0, z1 = Z - window, Z

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
            viewer.window.add_dock_widget(qt_viewer, area="bottom", name=y_label)

        return viewer, v1d
