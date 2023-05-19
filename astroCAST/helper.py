import glob
import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
import awkward as ak
import h5py
from skimage.util import img_as_uint

import numpy as np
import pandas as pd
import tifffile
import tiledb
import xxhash


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

        if isinstance(arg, np.ndarray):
            return hash_from_ndarray(arg)

        elif isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series):
            df_hash = pd.util.hash_pandas_object(arg)
            return hash_from_ndarray(df_hash.values)

        elif isinstance(arg, dict):
            return sort(arg)

        elif callable(arg):
            return arg.__name__
        else:
            return arg

    def sort(kwargs):
        sorted_dict = OrderedDict()

        # make sure keys are sorted to get same hash
        keys = list(kwargs.keys())
        keys.sort()

        # convert to ordered dict
        for key in keys:

            value = kwargs[key]

            if isinstance(value, dict):
                sorted_dict[key] = sort(value)
            else:
                # hash arguments if necessary
                sorted_dict[key] = hash_arg(value)
        return sorted_dict

    def get_hash_from_args(f, args, kwargs):

        cache_key = json.dumps(
            [
                f.__name__,
                [hash_arg(arg) for arg in args[1:]],
                sort(kwargs)
            ],
            separators=(',', ':')
        )

        print(f"\t{f.__name__}: {cache_key}")

        h = xxhash.xxh64()
        h.update(cache_key)
        hash_val = h.intdigest()
        h.reset()

        return hash_val

    def get_string_from_args(f, args, kwargs):

        name_ = f.__name__
        args_ = [hash_arg(arg) for arg in args[1:]]
        kwargs_ = sort(kwargs)

        hash_string = f"{name_}_"

        for a in args_:
            hash_string += f"{a}_"

        for k in kwargs.keys():
            hash_string += f"{k}-{kwargs_[k]}_"

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

        # what happens if we call it on a function without self??
        self_ = args[0]

        if self_.local_cache:

            # get hash from arguments
            # hash_value = get_hash_from_args(f, args, kwargs)
            # print("\thas_value: ", hash_value)
            # cache_path = self_.lc_path.joinpath(f"{f.__name__}_{hash_value}")
            # print("\tcache_path: ", cache_path)

            hash_string = get_string_from_args(f, args, kwargs)
            cache_path = self_.lc_path.joinpath(hash_string)
            # print("\tcache_path: ", cache_path)

            # find file with regex matching from hash_value
            files = glob.glob(cache_path.as_posix() + ".*")
            # print("\tfiles: ", files)

            # exists
            if len(files) == 1:

                result = load_value(files[0])

                if result is None:
                    logging.info("error during loading. recacalculating value")
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

# TODO update docstring for byte_num
def get_data_dimensions(input_, loc=None, return_dtype=False):
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

    # If the input is of an unrecognized format, raise a TypeError
    else:
        raise TypeError(f"data format not recognized: {type(path)}")

    if return_dtype:
        return (shape, chunksize, dtype)
    else:
        return (shape, chunksize)

class DummyGenerator:

    def __init__(self, num_rows=25, trace_length=12, ragged=False, offset=0):

        self.data = self.get_data(num_rows=num_rows, trace_length=trace_length, ragged=ragged, offset=offset)

    @staticmethod
    def get_data(num_rows, trace_length, ragged, offset):

        if isinstance(ragged, str):
            ragged = True if ragged == "ragged" else False

        if ragged:
            data = [np.random.random(
                size=trace_length + np.random.randint(low=-trace_length + 1, high=trace_length - 1)) + offset for _ in
                    range(num_rows)]
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

class EventSim:

    def __init__(self):
        pass

    @staticmethod
    def split_3d_array_indices(arr, cz, cx, cy):
        """
        Split a 3D array into sections based on the given segment lengths.

        Args:
            arr (numpy.ndarray): The 3D array to split.
            cz (int): The length of each section along the depth dimension.
            cx (int): The length of each section along the rows dimension.
            cy (int): The length of each section along the columns dimension.

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
        if depth % cz != 0 or rows % cx != 0 or cols % cy != 0:
            logging.warning("Segment lengths do not evenly divide the array dimensions.")

        # Calculate the number of sections in each dimension
        num_sections_z = depth // cz
        num_sections_x = rows // cx
        num_sections_y = cols // cy

        # Calculate the indices for each section
        indices = []
        for i in range(num_sections_z):
            for j in range(num_sections_x):
                for k in range(num_sections_y):
                    start_z = i * section_size_z
                    end_z = (i + 1) * section_size_z
                    start_x = j * section_size_x
                    end_x = (j + 1) * section_size_x
                    start_y = k * section_size_y
                    end_y = (k + 1) * section_size_y
                    indices.append((start_z, end_z, start_x, end_x, start_y, end_y))

        return indices

    @staticmethod
    def create_random_blob(shape, min_gap=1, blob_size_fraction=0.2, event_num=1):
        """
        Generate a random blob of connected shape in a given array.

        Args:
            shape (tuple): The shape of the array (depth, rows, cols).
            min_gap (int, optional): The minimum distance of the blob to the edge of the array. Default is 1.
            blob_size_fraction (float, optional): The average size of the blob as a fraction of the total array size.
                                                  Default is 0.2.
            event_num (int, optional): The value to assign to the blob pixels. Default is 1.

        Returns:
            numpy.ndarray: The array with the generated random blob.

        Raises:
            None
        """

        array = np.zeros(shape, dtype=int)

        # Get the dimensions of the array
        depth, rows, cols = shape

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
            array[z, x, y] = event_num

            # Add the current coordinate to the visited set
            visited.add((z, x, y))

            # Generate random neighbors within the min_gap distance
            neighbors = [(z + dz, x + dx, y + dy)
                         for dz in range(-min_gap, min_gap + 1)
                         for dx in range(-min_gap, min_gap + 1)
                         for dy in range(-min_gap, min_gap + 1)
                         if abs(dz) + abs(dx) + abs(dy) <= min_gap
                         and 0 <= z + dz < depth
                         and 0 <= x + dx < rows
                         and 0 <= y + dy < cols]

            # Add the neighbors to the queue
            queue.extend(neighbors)

        return array

    def simulate(self, shape, z_fraction=0.2, xy_fraction=0.1, gap_space=1, gap_time=1,
                 blob_size_fraction=0.05, event_probability=0.2):

        """
        Simulate the generation of random blobs in a 3D array.

        Args:
            shape (tuple): The shape of the 3D array (depth, rows, cols).
            z_fraction (float, optional): The fraction of the depth dimension to be covered by the blobs. Default is 0.2.
            xy_fraction (float, optional): The fraction of the rows and columns dimensions to be covered by the blobs.
                                           Default is 0.1.
            gap_space (int, optional): The minimum distance between blobs along the rows and columns. Default is 1.
            gap_time (int, optional): The minimum distance between blobs along the depth dimension. Default is 1.
            blob_size_fraction (float, optional): The average size of the blob as a fraction of the total array size.
                                                  Default is 0.05.
            event_probability (float, optional): The probability of generating a blob in each section. Default is 0.2.

        Returns:
            numpy.ndarray: The 3D array with the generated random blobs.
            int: The number of created events.

        Raises:
            None
        """

        # Create empty array
        event_map = np.zeros(shape, dtype=int)
        Z, X, Y = shape

        # Get indices for splitting the array into sections
        indices = self.split_3d_array_indices(event_map, int(Z*z_fraction), int(X*xy_fraction), int(Y*xy_fraction))

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

            shape = (z1 - z0, x1 - x0, y1 - y0)

            blob = self.create_random_blob(shape, event_num=num + 1, blob_size_fraction=blob_size_fraction)
            event_map[z0:z1, x0:x1, y0:y1] = blob

            num_events += 1

        # Convert to TIFF compatible format
        event_map = img_as_uint(event_map)

        return event_map, num_events

class Normalization:

    def __init__(self, data, inplace=True):

        if not inplace:
            data = data.copy()

        # check if ragged and convert to appropriate type
        ragged = False
        if isinstance(data, list):

            last_len = len(data[0])
            for dat in data[1:]:
                cur_len = len(dat)

                if cur_len != last_len:
                    ragged = True
                    self.data = ak.Array(data)
                    break

                last_len = cur_len

            if not ragged:
                self.data = np.array(data)

        elif isinstance(data, pd.Series):

            if len(data.apply(lambda x: len(x)).unique()) > 1:
                self.data = ak.Array(data.tolist())
            else:
                self.data = np.array(data.tolist())

        elif isinstance(data, np.ndarray):

            if isinstance(data.dtype, object):
                last_len = len(data[0])
                for i in range(1, data.shape[0]):
                    cur_len = len(data[i])

                    if cur_len != last_len:
                        ragged = True
                        self.data = ak.Array(data.tolist())
                        break

                    last_len = cur_len

                if not ragged:
                    self.data = np.array(data, dtype=type(data[0][0]))

            else:
                self.data = data
        else:
            raise TypeError(f"datatype not recognized: {type(data)}")

    def run(self, instructions):

        assert isinstance(instructions,
                          dict), "please provide 'instructions' as {0: 'func_name'} or {0: ['func_name', params]}"

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

        instructions = {
            0: ["subtract", {"mode": "min"}],
            1: ["divide", {"mode": "max_abs"}]
        }
        return self.run(instructions)

    @staticmethod
    def get_value(data, mode, population_wide):

        axis = None if population_wide else 1

        mode_options = {
            "first": lambda x: np.mean(x[:, 0]) if population_wide else x[:, 0],
            "mean": lambda x: np.mean(x, axis=axis),
            "min": lambda x: np.min(x, axis=axis),
            "min_abs": lambda x: np.min(np.abs(x), axis=axis),
            "max": lambda x: np.max(x, axis=axis),
            "max_abs": lambda x: np.max(np.abs(x), axis=axis),
            "std": lambda x: np.std(x, axis=axis)
        }
        assert mode in mode_options.keys(), f"please provide valid mode: {mode_options.keys()}"

        logging.warning(f"data: {data}")
        logging.warning(f"type: {type(data)}")
        # logging.warning(f"shape: {data.shape}")
        # logging.warning(f"std: {np.std(data, axis=axis)}")

        ret = mode_options[mode](data)
        return ret if population_wide else ret[:, None]  # broadcasting for downstream calculation

    def subtract(self, data, mode="min", population_wide=False):
        return data - self.get_value(data, mode, population_wide)

    def divide(self, data, mode="max", population_wide=False):

        divisor = self.get_value(data, mode, population_wide)
        print("type: ", type(data))

        # deal with ZeroDivisonError
        if population_wide and divisor == 0:
            logging.warning("Encountered '0' in divisor, returning data untouched.")
            return data

        else:
            idx = np.where(divisor == 0)[0]
            print("idx: ", idx)
            if len(idx) > 0:
                logging.warning("Encountered '0' in divisor, returning those rows untouched.")

                if isinstance(data, ak.Array):

                    # recreate array, since modifications cannot be done inplace
                    data = ak.Array([data[i] / divisor[i] if i not in idx else data[i] for i in range(len(data))])

                else:
                    mask = np.ones(len(data), bool)
                    mask[idx] = 0

                    print("divisor_mask: ", divisor[mask])
                    data[mask, :] = data[mask, :] / divisor[mask]

                return data

            else:
                return data / divisor

    @staticmethod
    def enforce_length(data, min_length=None, pad_mode="edge", max_length=None):

        if min_length is None and max_length is None:
            return data

        if isinstance(data, np.ndarray):

            logging.warning(data.shape)
            if min_length is not None and data.shape[1] < min_length:
                data = np.pad(data, pad_width=min_length - data.shape[1], mode=pad_mode)
                data = data[:, :min_length]

            elif max_length is not None and data.shape[1] > max_length:
                data = data[:, :max_length]

            return data

        elif isinstance(data, ak.Array):

            if min_length is not None and max_length is None:
                data = ak.pad_none(data, min_length)

            elif max_length is not None and min_length is None:
                data = data[:, :max_length]

            else:
                assert max_length == min_length, "when providing 'max_length' and 'min_length', both have to be equal"
                data = ak.pad_none(data, max_length, clip=True)

        else:
            raise TypeError(f"datatype {type(data)} not recognized.")

        return data

    @staticmethod
    def impute_nan(data, fixed_value=None):

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
            return ak.Array([np.diff(data[i]) for i in range(len(data))])

        else:
            return np.diff(data, axis=1)
