import glob
import logging
import pickle
import platform
import shutil
import tempfile
import time
import types
from pathlib import Path
from typing import Union, Tuple

import awkward as ak
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import py.path
import tifffile
import tiledb
import xxhash
import yaml
from skimage.util import img_as_uint


def closest_power_of_two(value):
    """
    Check if the value is a power of two, and if not, find the closest power of two.

    Args:
        value (int): The input value to check.

    Returns:
        int: The closest power of two.
    """
    # Check if the value is already a power of two
    if value & (value - 1) == 0 and value != 0:
        return value

    # Calculate the closest power of two
    closest_power = int(np.power(2, np.round(np.log2(value))))

    # Log a warning
    logging.warning(f"Input value {value} is not a power of 2. Using the closest power of 2: {closest_power}")

    return closest_power


def remove_temp_safe(tmp_dir: tempfile.TemporaryDirectory, wait_time: int = 20):
    # necessary to give Windows time to release files
    if platform.system() in ["Windows", "win32"]:

        time.sleep(wait_time)

        files = list(Path(tmp_dir.name).glob("*/*")) + list(Path(tmp_dir.name).glob("*"))
        for file in files:

            try:
                if file.is_file():
                    file.unlink(missing_ok=True)
                elif file.is_dir():
                    shutil.rmtree(file.as_posix())
            except PermissionError:
                logging.warning(f"Unable to delete locked file: {file}")

        logging.warning(f"Assuming to be on windows. Waiting for files to be released!")

        if len(list(Path(tmp_dir.name).glob("*"))) != 0:
            logging.error(f"temp dir not empty after cleanup: {tmp_dir.name}")

    tmp_dir.cleanup()

    if Path(tmp_dir.name).exists():
        logging.error(f"temp dir still exists after cleanup! {tmp_dir.name}")


def is_docker():
    path = Path('/proc/self/cgroup')
    return (
            path.joinpath('.dockerenv').exists() or
            path.is_file() and any('docker' in line for line in open(path.as_posix()))
    )


def wrapper_local_cache(f):
    """ Wrapper that creates a local save of the function call based on a hash of the arguments
    expects a function from a class with 'lc_path'::pathlib.Path and 'local_cache':bool attribute

    :param f:
    :return:
    """

    def hash_from_ndarray(v):
        h = xxhash.xxh64()
        h.update(v.flatten())

        return h.intdigest()

    def hash_arg(arg):

        from astrocast.analysis import Events, Video
        from astrocast.reduction import FeatureExtraction

        if isinstance(arg, np.ndarray):
            return hash_from_ndarray(arg)

        elif isinstance(arg, (pd.DataFrame, pd.Series)):
            df_hash = pd.util.hash_pandas_object(arg)
            return hash_from_ndarray(df_hash.values)

        elif isinstance(arg, dict):
            return get_hash_from_dict(arg)

        elif isinstance(arg, (Events, Video, FeatureExtraction)):
            return hash(arg)

        elif isinstance(arg, (bool, int, tuple)):
            return str(arg)

        elif isinstance(arg, str):

            if len(arg) < 10:
                return arg
            else:
                return hash(arg)

        elif isinstance(arg, list):

            arg = pd.Series(arg)
            df_hash = pd.util.hash_pandas_object(arg)
            return hash_from_ndarray(df_hash.values)

        elif callable(arg):
            return arg.__name__

        else:
            logging.warning(f"unknown argument type: {type(arg)}")

            try:
                h = hash(arg)
                return h

            except:
                logging.error(f"couldn't hash argument type: {type(arg)}")
                return arg

    def get_hash_from_dict(kwargs):

        # make sure keys are sorted to get same hash
        keys = list(kwargs.keys())
        keys.sort()

        # convert to ordered dict
        hash_string = ""
        for key in keys:

            if key in ["show_progress", "verbose", "verbosity", "cache_path", "n_jobs", "njobs"]:
                continue

            if key in ["in_place", "inplace"]:
                logging.warning(
                    f"cached value was loaded, which is incompatible with inplace option. "
                    f"Please overwrite value manually!"
                )
                continue

            # save key name
            hash_string += f"{hash_arg(key)}-"

            value = kwargs[key]
            hash_string += f"{hash_arg(value)}_"

        return hash_string

    def get_string_from_args(f, args, kwargs):

        hash_string = f"{f.__name__}_"

        args_ = [hash_arg(arg) for arg in args]
        for a in args_:
            hash_string += f"{a}_"

        hash_string += get_hash_from_dict(kwargs)

        logging.warning(f"hash_string: {hash_string}")
        return hash_string

    def save_value(path, value):

        # convert file path
        if isinstance(path, Path):
            path = path.as_posix()

        # convert pandas
        if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
            # value.to_csv(path+".csv", )
            with open(path + ".p", "wb") as f:
                pickle.dump(value, f)

        elif isinstance(value, np.ndarray) or isinstance(value, float) or isinstance(value, int):
            np.save(path + ".npy", value)

        else:

            try:
                # last saving attempt
                with open(path + ".p", "wb") as f:
                    pickle.dump(value, f)
            except:
                print("saving failed because datatype is unknown: ", type(value))
                return False

        return True

    def load_value(path):

        # convert file path
        if isinstance(path, Path):
            path = path.as_posix()

        # get suffix
        suffix = path.split(".")[-1]

        if suffix == "csv":
            result = pd.read_csv(path, index_col="Unnamed: 0")

        elif suffix == "npy":
            result = np.load(path)

        elif suffix == "p":
            with open(path, "rb") as f:
                result = pickle.load(f)

        else:
            print("loading failed because filetype not recognized: ", path)
            result = None

        return result

    def inner_function(*args, **kwargs):

        if isinstance(f, types.FunctionType) and "cache_path" in list(kwargs.keys()):
            cache_path = kwargs["cache_path"]

        else:

            try:
                self_ = args[0]
                cache_path = self_.cache_path

            except:
                logging.warning(f"trying to cache static method or class without 'cache_path': {f.__name__}")
                cache_path = None

        if cache_path is not None:

            hash_string = get_string_from_args(f, args, kwargs)
            cache_path = cache_path.joinpath(hash_string)

            # find file with regex matching from hash_value
            files = glob.glob(cache_path.as_posix() + ".*")

            # exists
            if len(files) == 1:

                result = load_value(files[0])

                if result is None:
                    logging.info("error during loading. recalculating value")
                    return f(*args, **kwargs)

                logging.info(f"loaded result of {f.__name__} from file")

            else:

                result = f(*args, **kwargs)

                if len(files) > 0:
                    logging.info(f"multiple saves found. files should be deleted: {files}")

                # save result
                logging.info(f"saving to: {cache_path}")
                save_value(cache_path, result)

        else:
            result = f(*args, **kwargs)

        return result

    return inner_function


def experimental(func):
    """
    Decorator to mark functions as experimental and log a warning upon their usage.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with a warning.
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        message = f"Warning: {func.__name__} is an experimental function and may be unstable."
        logger.warning(message)
        return func(*args, **kwargs)

    return wrapper


def get_data_dimensions(
        data: Union[np.ndarray, da.Array, str, Path], loc: str = None, return_dtype: bool = False
) -> Union[Tuple[Tuple, Tuple], Tuple[Tuple, Tuple, type]]:
    """ Takes an input object and returns the shape and chunksize of the data it represents. Optionally
        the chunksize can be returned as well.

    Args:
        data: An object representing the data whose dimensions are to be calculated.
        loc: A string representing the location of the data in the HDF5 file. This parameter is optional and only applicable when data is a Path to an HDF file.
        return_dtype: A boolean indicating whether to return the data type of the data.

    Raises:
        TypeError: If the input is not of a recognized type.
    """

    if isinstance(data, np.ndarray):
        shape = data.shape
        chunksize = np.array([])
        dtype = data.dtype

    elif isinstance(data, da.Array):
        shape = data.shape
        chunksize = data.chunksize
        dtype = data.dtype

    elif isinstance(data, (str, Path)):
        path = Path(data)

        # If the input is a Path to an HDF5 file, check if the file has the .h5 extension
        if path.suffix in [".h5", ".hdf5"]:
            # If the 'loc' parameter is not provided, raise an AssertionError
            assert loc is not None, "please provide a dataset location as 'loc' parameter"
            # Open the HDF5 file and read the data at the specified location
            with h5py.File(path.as_posix()) as file:
                data = file[loc]
                shape = data.shape
                chunksize = data.chunks
                dtype = data.dtype

        # If the input is a Path to a TIFF file, get the shape of the image data
        elif path.suffix in [".tiff", ".tif", ".TIFF", ".TIF"]:

            # Open the TIFF file and read the data dimensions
            with tifffile.TiffFile(path.as_posix()) as tif:
                shape = (len(tif.pages), *tif.pages[0].shape)
                chunksize = None
                dtype = tif.pages[0].dtype

        # If the input is not a Path to an HDF5 file, check if it is a Path to a TileDB array
        elif path.suffix == ".tdb":
            # Open the TileDB array and get its shape and chunksize
            with tiledb.open(path.as_posix()) as tdb:
                shape = tdb.shape
                chunksize = [int(tdb.schema.domain.dim(i).tile) for i in range(tdb.schema.domain.ndim)]
                dtype = tdb.schema.domain.dtype

        else:
            raise TypeError(f"data type not recognized: {path.suffix}")

    # If the input is of an unrecognized format, raise a TypeError
    else:
        raise TypeError(f"data type not recognized: {type(data)}")

    if return_dtype:
        return shape, chunksize, dtype
    else:
        return shape, chunksize


class DummyGenerator:

    def __init__(
            self, num_rows=25, trace_length=12, ragged=False, offset=0, min_length=2, n_groups=None, n_clusters=None
    ):

        self.data = self.get_data(
            num_rows=num_rows, trace_length=trace_length, ragged=ragged, offset=offset, min_length=min_length
        )

        self.groups = None if n_groups is None else np.random.randint(0, n_groups, size=len(self.data), dtype=int)
        self.clusters = None if n_clusters is None else np.random.randint(0, n_clusters, size=len(self.data), dtype=int)

    @staticmethod
    def get_data(num_rows, trace_length, ragged, offset, min_length):

        if isinstance(ragged, str):
            ragged = True if ragged == "ragged" else False

        if ragged:

            data = []
            for _ in range(num_rows):
                random_length = max(
                    min_length, trace_length + np.random.randint(low=-trace_length, high=trace_length) + offset
                )
                data.append(np.random.random(size=(random_length)))

        else:
            data = np.random.random(size=(num_rows, trace_length)) + offset

        return data

    def get_dataframe(self):

        data = self.data

        if type(data) == list:
            df = pd.DataFrame(dict(trace=data))

        elif type(data) == np.ndarray:
            df = pd.DataFrame(dict(trace=data.tolist()))
        else:
            raise TypeError

        # create dz, z0 and z1
        df["dz"] = df.trace.apply(lambda x: len(x))

        dz_sum = int(df.dz.sum() / 2)
        df["z0"] = [np.random.randint(low=0, high=max(dz_sum, 1)) for _ in range(len(df))]
        df["z1"] = df.z0 + df.dz

        # create fake index
        df["idx"] = df.index

        return df

    def get_list(self):

        data = self.data

        if type(data) == list:
            return data

        elif type(data) == np.ndarray:
            return data.tolist()

        else:
            raise TypeError

    def get_array(self):

        data = self.data

        if type(data) == list:
            return np.array(data, dtype='object')

        elif type(data) == np.ndarray:
            return data

        else:
            raise TypeError

    def get_dask(self, chunks=None):

        data = self.get_array()

        if isinstance(data.dtype, object):

            if chunks is None:

                if len(data.shape) == 1:
                    chunks = (1)
                elif len(data.shape) == 2:
                    chunks = (1, -1)
                else:
                    raise ValueError("unable to infer chunks for da. Please provide 'chunks' flag.")

                chunks = (1, -1) if chunks is None else chunks

                return da.from_array(data, chunks=chunks)

        else:
            return da.from_array(data, chunks="auto")

    def get_events(self):

        from astrocast.analysis import Events

        ev = Events(event_dir=None)
        df = self.get_dataframe()

        if self.groups is not None:
            df["group"] = self.groups

        if self.clusters is not None:
            df["clusters"] = self.clusters

        ev.events = df
        ev.seed = 1

        return ev

    def get_by_name(self, name, param={}):

        options = {"numpy": self.get_array(**param), "dask": self.get_dask(**param), "list": self.get_list(**param),
                   "pandas": self.get_dataframe(**param), "events": self.get_events(**param)}

        if name not in options.keys():
            raise ValueError(f"unknown attribute: {name}")

        return options[name]


class EventSim:

    def __init__(self):
        pass

    @staticmethod
    def split_3d_array_indices(arr, cz, cx, cy, skip_n):
        """
        Split a 3D array into sections based on the given segment lengths while skipping initial and trailing frames in z-dimension.

        Args:
            arr (numpy.ndarray): The 3D array to split.
            cz (int): The length of each section along the depth dimension.
            cx (int): The length of each section along the rows dimension.
            cy (int): The length of each section along the columns dimension.
            skip_n (int): Number of initial and trailing frames to skip in z-dimension.

        Returns:
            list: A list of tuples representing the start and end indices for each section.
                  Each tuple has the format (start_z, end_z, start_x, end_x, start_y, end_y).

        Raises:
            None

        Note:
            This function assumes that the segment lengths evenly divide the array dimensions.
            If the segment lengths do not evenly divide the array dimensions, a warning message is logged.
        """

        # Get the dimensions of the array
        depth, rows, cols = arr.shape

        # Define the segment lengths
        section_size_z = cz
        section_size_x = cx
        section_size_y = cy

        # Make sure the segment lengths evenly divide the array dimensions
        if (depth - 2 * skip_n) % cz != 0 or rows % cx != 0 or cols % cy != 0:
            logging.warning("Segment lengths do not evenly divide the adjusted array dimensions.")

        # Calculate the number of sections in each dimension
        num_sections_z = (depth - 2 * skip_n) // cz
        num_sections_x = rows // cx
        num_sections_y = cols // cy

        # Calculate the indices for each section
        indices = []
        for i in range(num_sections_z):
            for j in range(num_sections_x):
                for k in range(num_sections_y):
                    start_z = i * section_size_z + skip_n
                    end_z = (i + 1) * section_size_z + skip_n
                    start_x = j * section_size_x
                    end_x = (j + 1) * section_size_x
                    start_y = k * section_size_y
                    end_y = (k + 1) * section_size_y
                    indices.append((start_z, end_z, start_x, end_x, start_y, end_y))

        return indices

    @staticmethod
    def create_random_blob(section: np.ndarray, min_gap: int = 1, blob_size_fraction: float = 0.2,
                           event_num: int = 1) -> np.ndarray:
        """
        Generate a random blob of connected shape in a given array.

        Args:
            section: array to populate with blobs
            min_gap: The minimum distance of the blob to the edge of the array.
            blob_size_fraction: The average size of the blob as a fraction of the total array size.
            event_num: The value to assign to the blob pixels.

        Returns:
            numpy.ndarray: The array with the generated random blob.

        """

        # Get the dimensions of the array
        depth, rows, cols = section.shape

        # Calculate the maximum size of the blob based on the fraction of the total array size
        max_blob_size = int(blob_size_fraction * (depth * rows * cols))

        # Generate random coordinates for the starting point of the blob
        start_z = np.random.randint(min_gap, depth - min_gap)
        start_x = np.random.randint(min_gap, rows - min_gap)
        start_y = np.random.randint(min_gap, cols - min_gap)

        # Create a queue to store the coordinates of the blob
        queue = [(start_z, start_x, start_y)]

        # Create a set to keep track of visited coordinates
        visited = set()

        # Run the blob generation process
        while queue and len(visited) < max_blob_size:
            z, x, y = queue.pop(0)

            # Check if the current coordinate is already visited
            if (z, x, y) in visited:
                continue

            # Set the current coordinate to event_num in the array
            section[z, x, y] = event_num

            # Add the current coordinate to the visited set
            visited.add((z, x, y))

            # Generate random neighbors within the min_gap distance
            neighbors = [(z + dz, x + dx, y + dy) for dz in range(-min_gap, min_gap + 1) for dx in
                         range(-min_gap, min_gap + 1) for dy in range(-min_gap, min_gap + 1) if abs(dz) + abs(dx) + abs(
                    dy
                ) <= min_gap and 0 <= z + dz < depth and 0 <= x + dx < rows and 0 <= y + dy < cols]

            # Add the neighbors to the queue
            queue.extend(neighbors)

        return section

    def simulate(self, shape: Tuple[int, int, int], z_fraction: float = 0.2, xy_fraction: float = 0.1,
                 gap_space: int = 5, gap_time: int = 3, event_intensity: Union[str, int, float] = "incr",
                 background_noise: float = None, blob_size_fraction: float = 0.05,
                 event_probability: float = 0.2, skip_n: int = 5) -> Tuple[np.ndarray, int]:
        """ Simulates the generation of random events (blobs) in a 3D array.

        This function creates a 3D numpy array of the given shape and populates it with randomly generated blobs.
        The blobs' distribution and characteristics are determined by the specified parameters, allowing customization
        of the simulation. The method is useful for generating synthetic data for testing or algorithm development in
        image analysis.

        Args:
            shape: The shape of the 3D array (Z, X, Y).
            z_fraction: Fraction of the depth dimension to be covered by the blobs.
            xy_fraction: Fraction of the rows and columns dimensions to be covered by the blobs.
            gap_space: Minimum distance between blobs along rows and columns.
            gap_time: Minimum distance between blobs along the depth dimension.
            event_intensity: Determines intensity of the events. Can be 'incr' for incremental, or a specific int/float value.
            background_noise: Background noise level. None for no noise.
            blob_size_fraction: Average size of a blob as a fraction of the total array size.
            event_probability: Probability of generating a blob in each section.
            skip_n: Number of sections to skip for blob placement.

        Returns:
            A tuple containing the 3D array with generated blobs and the number of created events.

        Raises:
            ValueError: If `event_intensity` is neither 'incr', int, nor float.
        """

        # Create empty array
        if background_noise is None:
            event_map = np.zeros(shape, dtype=int)
        else:
            event_map = np.abs(np.random.random(shape) * background_noise)

        Z, X, Y = shape

        # Get indices for splitting the array into sections
        indices = self.split_3d_array_indices(
            event_map, int(Z * z_fraction), int(X * xy_fraction), int(Y * xy_fraction), skip_n=skip_n
        )

        # Fill with blobs
        num_events = 0
        for num, ind in enumerate(indices):
            # Skip section based on event_probability
            if np.random.random() > event_probability:
                continue

            z0, z1, x0, x1, y0, y1 = ind

            # Adjust indices to account for gap_time and gap_space
            z0 += int(gap_time / 2)
            z1 -= int(gap_time / 2)
            x0 += int(gap_space / 2)
            x1 -= int(gap_space / 2)
            y0 += int(gap_space / 2)
            y1 -= int(gap_space / 2)

            if event_intensity == "incr":
                event_num = num_events + 1
            elif isinstance(event_intensity, (int, float)):
                event_num = event_intensity
            else:
                raise ValueError(
                    f"event_intensity must be 'infer' or int/float; not {event_intensity}:{event_intensity.dtype}"
                )

            section = event_map[z0:z1, x0:x1, y0:y1]
            event_map[z0:z1, x0:x1, y0:y1] = self.create_random_blob(
                section, event_num=event_num, blob_size_fraction=blob_size_fraction
            )

            num_events += 1

        # Convert to TIFF compatible format
        if event_map.dtype == int:
            event_map = img_as_uint(event_map)

        return event_map, num_events

    def create_dataset(
            self, h5_path, loc="dff/ch0", debug=False, shape=(50, 100, 100), z_fraction=0.2, xy_fraction=0.1,
            gap_space=5, gap_time=3, event_intensity=100, background_noise=1, blob_size_fraction=0.05,
            event_probability=0.2
    ):

        from astrocast.analysis import IO
        from astrocast.detection import Detector

        h5_path = Path(h5_path)

        data, num_events = self.simulate(
            shape=shape, z_fraction=z_fraction, xy_fraction=xy_fraction, event_intensity=event_intensity,
            background_noise=background_noise, gap_space=gap_space, gap_time=gap_time,
            blob_size_fraction=blob_size_fraction, event_probability=event_probability
        )

        io = IO()
        io.save(path=h5_path, data=data, loc=loc)

        det = Detector(h5_path.as_posix(), output=None)
        det.run(loc=loc, lazy=True, debug=debug)

        return det.output_directory


class SampleInput:

    def __init__(self, test_data_dir="./testdata/", tmp_dir=None):
        self.test_data_dir = Path(test_data_dir)

        if tmp_dir is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
        else:
            self.tmp_dir = Path(tmp_dir.strpath)

        self.sample_path = None

    def get_dir(self):

        if isinstance(self.tmp_dir, tempfile.TemporaryDirectory):
            return Path(self.tmp_dir.name)

        elif isinstance(self.tmp_dir, py.path.LocalPath):
            return Path(self.tmp_dir.strpath)

        elif isinstance(self.tmp_dir, Path):
            return self.tmp_dir

        else:
            raise ValueError(f"tmp_dir must be of type tempfile.TemporaryDirectory or py.path.LocalPath, "
                             f"not {type(self.tmp_dir)}")

    def get_test_data(self, extension=".h5"):

        tmp_dir = self.get_dir()

        # collect sample file
        samples = list(self.test_data_dir.glob(f"sample_*{extension}"))
        assert len(samples) > 0, f"cannot find sample with extension: {extension}"
        sample = samples[0]

        # copy to temporary directory
        new_path = tmp_dir.joinpath(sample.name)
        shutil.copy(sample, new_path)
        assert new_path.exists()

        self.sample_path = new_path

        return new_path

    def get_loc(self, ref=None):

        if self.sample_path is None:
            raise FileNotFoundError("please run 'get_test_data()' first")

        if self.sample_path.suffix in [".h5", ".hdf5"]:

            with h5py.File(self.sample_path.as_posix(), "r") as f:

                # make sure reference dataset exists in sample file
                if ref is not None and ref not in f:
                    raise ValueError(f"cannot find {ref}")
                elif ref is not None:
                    return ref

                # get dataset
                def recursive_get_dataset(f_, loc):

                    if loc is None:
                        # choose first location if none is provided
                        locs = list(f.keys())
                        loc = locs[0]

                    if isinstance(f_[loc], h5py.Group):

                        locs = list(f_[loc].keys())
                        if len(loc) < 1:
                            raise ValueError(f"cannot find any datasets in sample file: {self.sample_path}")

                        loc = f"{loc}/{locs[0]}"
                        return recursive_get_dataset(f_, loc)

                    if isinstance(f_[loc], h5py.Dataset):
                        return loc

                return recursive_get_dataset(f, None)

    def clean_up(self):

        tmp_dir = self.get_dir()

        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def is_ragged(data):
    # check if ragged and convert to appropriate type
    ragged = False
    if isinstance(data, list):

        if not isinstance(data[0], (list, np.ndarray)):
            ragged = False

        else:

            last_len = len(data[0])
            for dat in data[1:]:
                cur_len = len(dat)

                if cur_len != last_len:
                    ragged = True
                    break

                last_len = cur_len

    elif isinstance(data, pd.Series):

        if len(data.apply(lambda x: len(x)).unique()) > 1:
            ragged = True

    elif isinstance(data, (np.ndarray, da.Array)):

        if isinstance(data.dtype, object) and isinstance(data[0], (np.ndarray, da.Array)):

            item0 = data[0] if isinstance(data[0], np.ndarray) else data[0].compute()
            last_len = len(item0)

            for i in range(1, data.shape[0]):

                item = data[i] if isinstance(data[i], np.ndarray) else data[i].compute()

                cur_len = len(item)

                if cur_len != last_len:
                    ragged = True
                    break

                last_len = cur_len

    else:
        raise TypeError(f"datatype not recognized: {type(data)}")

    return ragged


class Normalization:

    def __init__(self, data, inplace=True):

        if not inplace:
            data = data.copy()

        if not isinstance(data, (list, np.ndarray, pd.Series)):
            raise TypeError(f"datatype not recognized: {type(data)}")

        if isinstance(data, (pd.Series, np.ndarray)):
            data = data.tolist()

        data = ak.Array(data) if is_ragged(data) else np.array(data)

        # enforce minimum of two dimensions
        if isinstance(data, np.ndarray) and len(data.shape) < 2:
            data = [data]

        self.data = data

    def run(self, instructions):

        assert isinstance(
            instructions, dict
        ), "please provide 'instructions' as {0: 'func_name'} or {0: ['func_name', params]}"

        data = self.data

        keys = np.sort(list(instructions.keys()))
        for key in keys:

            instruct = instructions[key]
            if isinstance(instruct, str):
                func = self.__getattribute__(instruct)
                data = func(data)

            elif isinstance(instruct, list):
                func, param = instruct
                func = self.__getattribute__(func)

                data = func(data, **param)

        return data

    def min_max(self):

        instructions = {0: ["subtract", {"mode": "min"}], 1: ["divide", {"mode": "max_abs"}]}
        return self.run(instructions)

    def mean_std(self):

        instructions = {0: ["subtract", {"mode": "mean"}], 1: ["divide", {"mode": "std"}]}
        return self.run(instructions)

    @staticmethod
    def get_value(data, mode, population_wide=False, axis=1):

        summary_axis = None if population_wide else axis

        mode_options = {
            "first": lambda x: np.mean(x[:, 0] if axis else x[0, :]) if population_wide else x[:, 0] if axis else x[0,
                                                                                                                  :],
            "mean": lambda x: np.mean(x, axis=summary_axis), "min": lambda x: np.min(x, axis=summary_axis),
            "min_abs": lambda x: np.min(np.abs(x), axis=summary_axis), "max": lambda x: np.max(x, axis=summary_axis),
            "max_abs": lambda x: np.max(np.abs(x), axis=summary_axis), "std": lambda x: np.std(x, axis=summary_axis)}
        assert mode in mode_options.keys(), f"please provide valid mode: {mode_options.keys()}"

        ret = mode_options[mode](data)
        return ret if population_wide else ret[:, None]  # broadcasting for downstream calculation

    def subtract(self, data, mode="min", population_wide=False, rows=True):

        value = self.get_value(data, mode, population_wide, axis=int(rows))

        # transpose result if subtracting by columns
        if not rows:
            value = value.tranpose()

        return data - value

    def divide(self, data, mode="max", population_wide=False, rows=True):

        divisor = self.get_value(data, mode, population_wide, axis=int(rows))

        # deal with ZeroDivisonError
        if population_wide and divisor == 0:
            logging.warning("Encountered '0' in divisor, returning data untouched.")
            return data

        # row by row
        else:

            # check if there are zeros in any rows
            idx = np.where(divisor == 0)[0]
            if len(idx) > 0:
                logging.warning("Encountered '0' in divisor, returning those rows untouched.")

                if isinstance(data, ak.Array):

                    if not rows:
                        raise ValueError("column wise normalization cannot be performed for ragged arrays.")

                    # recreate array, since modifications cannot be done inplace
                    data = ak.Array([data[i] / divisor[i] if i not in idx else data[i] for i in range(len(data))])

                else:

                    mask = np.ones(data.shape[0], bool) if rows else np.ones(data.shape[1], bool)
                    mask[idx] = 0

                    if rows:
                        data[mask, :] = data[mask, :] / divisor[mask]
                    else:
                        data[:, mask] = np.squeeze(data[:, mask]) / np.squeeze(divisor[mask])

                return data

            # all rows healthy
            else:

                if rows:
                    return data / divisor
                else:
                    return data / np.squeeze(divisor)

    @staticmethod
    def impute_nan(data, fixed_value=None):

        if len(data) == 0:
            return data

        if isinstance(data, np.ndarray):

            if fixed_value is not None:
                return np.nan_to_num(data, copy=True, nan=fixed_value)

            else:

                for r in range(data.shape[0]):
                    trace = data[r, :]

                    mask = np.isnan(trace)
                    logging.debug(f"mask: {mask}")
                    trace[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), trace[~mask])

                    data[r, :] = trace

        elif isinstance(data, ak.Array):

            if fixed_value is not None:
                data = ak.fill_none(data, fixed_value)  # this does not deal with np.nan

            container = []
            for r in range(len(data)):

                trace = data[r].to_numpy(allow_missing=True)

                mask = np.isnan(trace)
                if fixed_value is not None:
                    trace[mask] = fixed_value
                else:
                    trace = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), trace[~mask])

                container.append(trace)

            data = ak.Array(container)

        else:
            raise TypeError("please provide np.ndarray or ak.Array")

        return data

    @staticmethod
    def diff(data):

        if isinstance(data, ak.Array):

            arr = []
            zero = np.zeros([1])
            for i in range(len(data)):
                row = np.concatenate([zero, np.diff(data[i], axis=0)])
                arr.append(row)

            return ak.Array(arr)

        else:

            x = np.diff(data, axis=1)

            if len(x.shape) > 1:
                zero = np.zeros((x.shape[0], 1), dtype=x.dtype)  # Reshape zero to match a single column of x
            else:
                zero = np.zeros([1], dtype=x.dtype)

            return np.concatenate([zero, x], axis=1)


class CachedClass:

    def __init__(self, cache_path=None, logging_level=logging.INFO):

        if cache_path is not None:

            if isinstance(cache_path, str):
                cache_path = Path(cache_path)

            if not cache_path.is_dir():
                cache_path.mkdir()

        self.cache_path = cache_path

        # set logging level
        logging.basicConfig(level=logging_level)

    @wrapper_local_cache
    def print_cache_path(self):
        logging.warning(f"cache_path: {self.cache_path}")
        time.sleep(0.5)
        return np.random.random(1)


def load_yaml_defaults(yaml_file_path):
    """Load default values from a YAML file."""

    logging.warning(
        "loading configuration from yaml file. "
        "Be advised that command line parameters take priority over configurations in the yaml."
    )

    with open(yaml_file_path, 'r') as file:
        params = yaml.safe_load(file)

        for key, value in params.items():
            logging.info(f"yaml parameter >> {key}:{value}")

        return params


def download_sample_data(save_path, public_datasets=True, custom_datasets=True):
    import gdown

    save_path = Path(save_path)

    if public_datasets:
        folder_url = "https://drive.google.com/drive/u/0/folders/10hhWg4XdVGlPmqmSXy4devqfjs2xE6A6"
        gdown.download_folder(
            folder_url, output=save_path.joinpath("public_data").as_posix(), quiet=False, use_cookies=False
        )

    if custom_datasets:
        folder_url = "https://drive.google.com/drive/u/0/folders/13I_1q3osfIGlLhjEiAnLBoJSfPux688g"
        gdown.download_folder(
            folder_url, output=save_path.joinpath("custom_data").as_posix(), quiet=False, use_cookies=False,
            remaining_ok=True
        )

    logging.info(f"Downloaded sample datasets to: {save_path}")


def download_pretrained_models(save_path):
    import gdown

    save_path = Path(save_path)

    folder_url = "https://drive.google.com/drive/u/0/folders/1RJU-JjQIpoRJOqxivOVo44Q3irs88YX8"
    gdown.download_folder(
        folder_url, output=save_path.joinpath("models").as_posix(), quiet=False, use_cookies=False, remaining_ok=True
    )

    logging.info(f"Downloaded sample datasets to: {save_path}")
