import itertools
import logging
import tempfile
from pathlib import Path

import czifile
import dask
import h5py
import numpy as np
import pandas as pd
import tifffile
import tiledb
import dask.array as da
from dask_image import imread
from dask.distributed import Client, LocalCluster
from skimage.transform import resize
from skimage.util import img_as_uint
from deprecated import deprecated
from scipy.ndimage import minimum_filter1d

class Input:

    def __init__(self):
        pass

    def run(self, input_path, output_path=None,
            sep="_", channels=1,
            subtract_background=None, subtract_func="mean",
            rescale=None, dtype=np.uint,
            in_memory=False, prefix="data", chunks=None, compression=None):

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

        if input_path.suffix in [".tiff", ".tif", ".TIFF", ".TIF"] or \
                (input_path.is_dir() and len(
                    [f for f in input_path.glob("*") if f.suffix in [".tif", ".tiff", ".TIF", ".TIFF"]]) > 0):
            data = self.load_tiff(input_path, sep=sep)

        elif input_path.suffix in [".czi"]:
            data = self.load_czi(input_path)

        else:
            raise TypeError(f"File format is not implemented: {input_path.suffix}")

        data = self.prepare_data(data, channels=channels, subtract_background=subtract_background,
                                 subtract_func=subtract_func, rescale=rescale, dtype=dtype, in_memory=in_memory)

        if output_path is None:
            return data

        self.save(output_path, data, prefix=prefix, chunks=chunks, compression=compression)

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

    def load_czi(self, path):

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

    def load_tiff(self, path, sep="_"):

        """
        Loads TIFF image data from the specified path and returns a Dask array.

        Args:
            path (str or pathlib.Path): The path to the TIFF file or directory containing TIFF files.
            sep (str): The separator used in sorting the filenames (default: "_").

        Returns:
            dask.array.core.Array: The loaded TIFF data as a Dask array.

        Raises:
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
            files = self.sort_alpha_numerical_names(file_names=files, sep=sep)

            # Read the TIFF files using dask.array and stack them
            stack = da.stack([imread.imread(f.as_posix()) for f in files])
            stack = np.squeeze(stack)

            if len(stack.shape) != 3:
                raise NotImplementedError(f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D")

        elif path.is_file():
            # If the path is a file, load a single TIFF file

            # TODO: Implement delayed loading from TIFF
            # with tifffile.TiffFile(path) as tif:
            #     num_frames = len(tif.pages)
            #     X, Y = tif.pages[0].shape
            #     dtype = tif.pages[0].dtype
            #
            # stack = da.stack([
            #     da.from_delayed(dask.delayed(tifffile.imread(path, key=i)), shape=(1, X, Y), dtype=dtype)
            #                     for i in range(num_frames)])
            # stack = np.squeeze(stack)

            # Read the TIFF file using tifffile and create a Dask array
            arr = tifffile.imread(path)
            stack = da.from_array(arr, chunks=(1, -1, -1))

        else:
            raise FileNotFoundError(f"cannot find directory or file: {path}")

        return stack

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
                lambda chunk: resize(chunk, (chunk.shape[0], rX, rY), anti_aliasing=True))

        return data

    def prepare_data(self, data, channels=1,
                     subtract_background=None, subtract_func="mean",
                     rescale=None, dtype=np.uint,
                     in_memory=False):

        # Convert data to a dask array if it's an ndarray, otherwise validate the input type
        stack = da.from_array(data, chunks=(1, -1, -1)) if isinstance(data, np.ndarray) else data
        if not isinstance(data, da.Array): raise TypeError("Please provide data as np.ndarray or dask.array.Array")

        # Check if the data has the correct dimensions
        if len(stack.shape) != 3: raise NotImplementedError(f"dimensions incorrect: {len(stack.shape)}. Currently not implemented for dim != 3D")

        # Validate the channels input and determine the number of channels
        if not isinstance(channels, (int, dict)): raise ValueError(f"please provide channels as int or dictionary.")

        num_channels = channels if isinstance(channels, int) else len(len(channels.keys()))

        if stack.shape[0] % num_channels != 0:
            logging.warning(f"cannot divide frames into channel number: {stack.shape[0]} % {num_channels} != 0. May lead to unexpacted behavior")

        channels = channels if isinstance(channels, dict) else {i: f"ch{i}" for i in range(num_channels)}

        # Split the data into channels based on the given channel indices or names
        data = {}
        for channel_key in channels.keys():
            data[channel_key] = stack[channel_key::num_channels, :, :]

        # Subtract background if specified
        if subtract_background is not None:
            data = self.subtract_background(data, channels, subtract_background, subtract_func)

        # Rescale the data if specified
        if (rescale is not None) and rescale != 1 and rescale != 1.0:
            self.rescale_data(data, rescale)

        # Convert the data type if specified
        if dtype is not None:

            for k in data.keys():
                data[k] = img_as_uint(data[k]) if dtype == np.uint else data[k].astype(dtype)

        # Load the data into memory if requested
        data = dask.compute(data)[0] if in_memory else data

        # Rename the channels in the output dictionary
        return {channels[i]: data[i] for i in data.keys()}

    @staticmethod
    def save(path, data, prefix="data", chunks=None, compression=None):
        """
        Save data to a specified file format.

        Args:
            path (str or pathlib.Path): The path to the output file.
            data (dict): A dictionary containing the data to be saved, with keys as channel names and values as arrays.
            prefix (str): The prefix to be used for naming datasets within the file (applicable only for HDF5 format).
            chunks (tuple or None): The chunk size to be used when saving Dask arrays (applicable only for HDF5 format).
            compression (str or None): The compression method to be used when saving Dask arrays (applicable only for HDF5 format).

        Raises:
            TypeError: If the provided path is not a string or pathlib.Path object.
            TypeError: If the provided data is not a dictionary.
            TypeError: If the provided data is not in a supported format.

        Returns:
            list: A list containing the paths of the saved files.
        """

        # Cast the path to a pathlib.Path object if it's provided as a string
        if isinstance(path, str):
            path = Path(path)

        # Check if the path is a pathlib.Path object, otherwise raise an error
        if not isinstance(path, Path):
            raise TypeError("please provide 'path' as str or pathlib.Path data type")

        # Check if the data is a dictionary, otherwise raise an error
        if not isinstance(data, dict):
            raise TypeError("please provide data as dict of {channel_name:array}")

        saved_paths = []  # Initialize an empty list to store the paths of the saved files
        for k in data.keys():
            channel = data[k]

            # Check if the channel is a numpy.ndarray or a dask.array.Array, otherwise raise an error
            if not isinstance(channel, (np.ndarray, da.Array)):
                raise TypeError("please provide data as either 'numpy.ndarray' or 'da.array.Array'")

            if path.suffix in [".h5", ".hdf5"]:
                # Save as HDF5 format

                fpath = path
                if isinstance(channel, da.Array):
                    # Save Dask array
                    da.to_hdf5(fpath, f"{prefix}/{k}", channel, chunks=chunks, compression=compression, shuffle=False)

                else:
                    # Save NumPy array
                    with h5py.File(fpath, "a") as f:
                        ds = f.create_dataset(f"{prefix}/{k}", shape=channel.shape, chunks=chunks,
                                              compression=compression, shuffle=False, dtype=channel.dtype)
                        ds[:] = channel

                logging.info(f"dataset saved to {fpath}::{prefix}/{k}")

            elif path.suffix == ".tdb":
                # Save as TileDB format

                if isinstance(channel, np.ndarray):
                    channel = da.from_array(channel, chunks=chunks if chunks is not None else "auto")

                fpath = path.with_suffix(f".{k}.tdb") if len(data.keys()) > 1 else path
                da.to_tiledb(channel, fpath.as_posix(), compute=True)
                logging.info(f"dataset saved to {fpath}")

            elif path.suffix in [".tiff", ".TIFF", ".tif", ".TIF"]:
                # Save as TIFF format

                fpath = path.with_suffix(f".{k}.tiff") if len(data.keys()) > 1 else path
                tifffile.imwrite(fpath, data=channel)
                logging.info(f"saved data to {fpath}")

            else:
                raise TypeError("please provide output format as .h5, .tdb, or .tiff file")

        return saved_paths  # Return the list of saved file paths

class Delta:

    def __init__(self, input_, loc=None, in_memory=True, parallel=False, ):

        self.input_ = Path(input_) if isinstance(input_, str) else input_
        self.dim, self.chunksize = self.get_data_dimensions(self.input_, loc=loc)
        self.in_memory = in_memory
        self.parallel = parallel
        self.loc = loc

    def run(self, method="background", window=None, overwrite_first_frame=True, use_dask=True):

        data = self.prepare_data(self.input_, in_memory=self.in_memory, shared=self.parallel, use_dask=use_dask)

        # sequential execution
        if isinstance(data, np.ndarray):
            res = self.calculate_delta_min_filter(data, window, method=method, inplace=False)

        # parallel from .tdb file
        elif isinstance(data, str) and data.startswith("tdb:"):

            logging.warning("this function will overwrite the provided .tdb file!")
            calculate_delta_min_filter = self.calculate_delta_min_filter

            def wrapper(path, ranges):

                (x0, x1), (y0, y1) = ranges

                # open tdb and load range
                with tiledb.open(path, mode="r") as tdb:
                    data = tdb[:, x0:x1, y0:y1]
                    res = calculate_delta_min_filter(data, window, method=method, inplace=False)

                with tiledb.open(path, mode="w") as tdb:
                    tdb[:, x0:x1, y0:y1] = calculate_delta_min_filter(data, window, method=method, inplace=False)

                # TODO implement not in-place version
                # return calculate_delta_min_filter(data, window, method=method, inplace=False)

            path = data[len("tdb:"):]

            # get chunk size
            (Z, X, Y), chunksize = self.get_data_dimensions(path, loc=None)
            assert chunksize is not None
            cz, cx, cy = chunksize

            with LocalCluster() as lc:
                with Client(lc) as client:

                    futures = []
                    for x0 in range(0, X, cx):
                        for y0 in range(0, X, cy):
                            range_ = ((x0, x0 + cx), (y0, y0 + cy))
                            futures.append(
                                client.submit(wrapper, path, range_)
                            )

                    client.gather(futures)

            res = self.load_to_memory(Path(path), loc=None)

        elif isinstance(data, da.core.Array):

            res = data.map_blocks(self.calculate_delta_min_filter,
                                  window=window, method=method, inplace=False,
                                  dtype=data.dtype)
            res = res.compute()

        elif isinstance(data, str) and data.startswith("smm:"):
            raise NotImplementedError("Implement shared memory")

        else:
            raise NotImplementedError

        if overwrite_first_frame:
            res[0, :, :] = res[1, :, :]

        return res

    @staticmethod
    def get_data_dimensions(input_, loc=None):
        """
        This function takes an input object and returns the shape and chunksize of the data it represents.

        If the input is a numpy ndarray, it returns the shape of the ndarray and None for chunksize.

        If the input is a Path to an HDF5 file (.h5 extension), it reads the data at the specified location
        and returns the shape of the data and its chunksize. The location should be specified using the 'loc'
        parameter. If the 'loc' parameter is not provided, the function raises an AssertionError.

        If the input is a Path to a TIFF file (.tiff or .tif extension), it returns the shape and None for chunksize.

        If the input is a Path to a TileDB array (.tdb extension), it returns the shape and chunksize of the
        TileDB array.

        If the input is of an unrecognized format, the function raises a TypeError.

        Args:
        - input_: An object representing the data whose dimensions are to be calculated.
        - loc: A string representing the location of the data in the HDF5 file. This parameter is optional
          and only applicable when input_ is a Path to an HDF5 file.

        Returns:
        - A tuple containing two elements:
          * The shape of the data represented by the input object.
          * The chunksize of the data represented by the input object. If the data is not chunked,
            this value will be None.
        """

        # Check if the input is a numpy ndarray
        if isinstance(input_, np.ndarray):
            # Return the shape of the ndarray and None for chunksize
            return input_.shape, None

        elif isinstance(input_, Path):
            path = input_

        elif isinstance(input_, str):
            path = Path(input_)

        else:
            raise TypeError(f"data type not recognized: {type(input_)}")

        # If the input is a Path to an HDF5 file, check if the file has the .h5 extension
        if path.suffix == ".h5":
            # If the 'loc' parameter is not provided, raise an AssertionError
            assert loc is not None, "please provide a dataset location as 'loc' parameter"
            # Open the HDF5 file and read the data at the specified location
            with h5py.File(path.as_posix()) as file:
                data = file[loc]
                shape = data.shape
                chunksize = data.chunks

        # If the input is a Path to a TIFF file, get the shape of the image data
        elif path.suffix in [".tiff", ".tif"]:

            # Open the TIFF file and read the data dimensions
            with tifffile.TiffFile(path.as_posix()) as tif:
                shape = (len(tif.pages), *tif.pages[0].shape)
                chunksize = None

        # If the input is not a Path to an HDF5 file, check if it is a Path to a TileDB array
        elif path.suffix == ".tdb":
            # Open the TileDB array and get its shape and chunksize
            with tiledb.open(path.as_posix()) as tdb:
                shape = tdb.shape
                chunksize = [int(tdb.schema.domain.dim(i).tile) for i in range(tdb.schema.domain.ndim)]

        # If the input is of an unrecognized format, raise a TypeError
        else:
            raise TypeError(f"data format not recognized: {type(path)}")

        return shape, chunksize

    @staticmethod
    def load_to_memory(path, loc=None):
        """
        This function loads data from the specified path into memory and returns it.

        Args:
        - path: A path to the data to be loaded.
        - loc: A string representing the location of the data in the HDF5 file. This parameter is optional
          and only applicable when path has the .h5 extension.

        Returns:
        - The data loaded from the specified path.

        Raises:
        - TypeError: If the file type is not recognized.

        """

        # Check if the input is already a numpy ndarray
        if isinstance(path, np.ndarray):
            return path

        # If the input is not a ndarray, check if it has the .tiff or .tif extension
        if path.suffix in [".tiff", ".tif"]:
            # Load the data from the file using tifffile
            data = tifffile.imread(path)

        # If the input has the .tdb extension, open the TileDB array and load its data
        elif path.suffix in [".tdb"]:
            with tiledb.open(path.as_posix(), "r") as tdb:
                data = tdb[:]

        # If the input has the .h5 extension, open the HDF5 file and load the data at the specified location
        elif path.suffix in [".h5"]:
            with h5py.File(path, "r") as h5:
                data = h5[loc][:]

        # If the input is of an unrecognized format, raise a TypeError
        else:
            raise TypeError(f"don't recognize file type: {path}")

        return data

    def save_to_tdb(self, arr: np.ndarray) -> str:
        """
        Save a numpy ndarray to a TileDB array and return the path to the TileDB array.

        Args:
        - arr: A numpy ndarray to be saved.

        Returns:
        - A string representing the path to the TileDB array where the ndarray was saved.
        """

        # Check if the input array is a numpy ndarray
        assert isinstance(arr, np.ndarray)

        # Convert the numpy ndarray to a dask array for lazy loading and rechunk it
        data = da.from_array(arr, chunks=(self.dim))
        data = da.rechunk(data, chunks=(-1, "auto", "auto"))

        # Create a temporary directory to store the TileDB array
        tileDBpath = tempfile.mkdtemp(suffix=".tdb")

        # Write the dask array to a TileDB array
        data.to_tiledb(tileDBpath)

        # Return the path to the TileDB array
        return f"tdb:{tileDBpath}"

    def prepare_data(self, input_, in_memory=True, shared=True, use_dask=True):

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

        # convert to .tdb
        if not in_memory:

            if isinstance(input_, Path) and input_.suffix == ".tdb":
                return f"tdb:{input_.as_posix()}"

            elif isinstance(input_, Path):
                # if the input is a file path, load it into memory and convert to TileDB array
                data = self.load_to_memory(input_, loc=self.loc)
                return self.save_to_tdb(data)

            elif isinstance(input_, np.ndarray):
                # if the input is a numpy ndarray, convert to TileDB array
                return self.save_to_tdb(input_)

            else:
                # if the input data type is not recognized, raise an error
                raise TypeError(f"do not recognize data type: {type(input_)}")

        # simple load
        elif in_memory and not shared:
            return self.load_to_memory(input_, loc=self.loc)

        # create dask.array
        elif in_memory and use_dask:
            # if 'in_memory' is True and 'shared' is True, load the data into memory as a Dask array
            # TODO this should be lazy loading through dask array instead
            arr = self.load_to_memory(input_, loc=self.loc)

            data = da.from_array(arr, chunks=(self.dim))
            data = da.rechunk(data, chunks=(-1, "auto", "auto"))

        elif in_memory and not use_dask:

            # TODO create VRAM shared memory
            # if not use_dask:
            #     shm_name = "my_shared_memory" # TODO dynamic
            #     shm_length = arr.nbytes # TODO dynamic
            #     shm_array = shm.SharedMemory(create=True, name=shm_name, size=shm_length)
            #     shm_array_np = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm_array.buf)
            #     shm_array_np[:] = arr[:]

            raise NotImplementedError("implement shared memory option")

        else:
            raise NotImplementedError

        return data

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

        return tr_max

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
