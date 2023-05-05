import glob
import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
import awkward as ak

import numpy as np
import pandas as pd
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
            with open(path+".p", "wb") as f:
                    pickle.dump(value, f)

        elif isinstance(value, np.ndarray) or isinstance(value, float) or isinstance(value, int):
            np.save(path+".npy", value)

        else:

            try:
                # last saving attempt
                with open(path+".p", "wb") as f:
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
            files = glob.glob(cache_path.as_posix()+".*")
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

class DummyGenerator():

    def __init__(self, num_rows=25, trace_length=12, ragged=False):

        self.data = self.get_data(num_rows=num_rows, trace_length=trace_length, ragged=ragged)

    def get_data(self, num_rows, trace_length, ragged):

        if ragged:
            data = [np.random.random(size=trace_length+np.random.randint(low=-trace_length+1, high=trace_length-1)) for _ in range(num_rows)]
        else:
            data = np.random.random(size=(num_rows, trace_length))

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

        dz_sum = int(df.dz.sum()/2)
        df["z0"] = [np.random.randint(low=0, high=dz_sum) for _ in range(len(df))]
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

class Normalization:

    # TODO parallelization
    def __init__(self, data, approach):

        # check if ragged and convert to appropriate type
        ragged = False
        if isinstance(data, list):

            last_len = len(data[0])
            for dat in data[1:]:
                cur_len = len(dat)

                if cur_len != last_len:
                    ragged = True
                    self.data = ak.Array(data)
                    self.library = ak
                    break

                last_len = cur_len

            if not ragged:
                self.data = np.array(data)
                self.library = np

        elif isinstance(data, pd.Series):

            if len(data.apply(lambda x: len(x)).unique()) > 1:
                self.data = ak.Array(data.tolist())
                self.library = ak
            else:
                self.data = np.array(data.tolist())
                self.library = np

        elif isinstance(data, np.ndarray):

            if isinstance(data.dtype, object):
                last_len = len(data[0, :])
                for i in range(1, data.shape[0]):
                    cur_len = len(data[i, :])

                    if cur_len != last_len:
                        ragged = True
                        self.data = ak.Array(data)
                        self.library = ak
                        break

                    last_len = cur_len

                if not ragged:
                    self.data = np.array(data)
                    self.library = np

            else:
                self.data = data
                self.library = np
        else:
            raise TypeError(f"datatype not recognized: {type(data)}")

        # choose normalization approach
        func = getattr(self, approach, lambda: None)
        if func is not None:
            self.func = func
        else:
            raise AttributeError(f"please provide a valid approach instead of {approach}")

    def run(self):
        return self.func(self.data, self.library)

    @staticmethod
    def min_max(arr, library):

        """ subtract minimum and divide by new maximum

        :returns array between 0 and 1
        """

        arr = arr - np.expand_dims(library.min(arr, axis=1), 1)
        arr = arr / np.expand_dims(library.max(np.abs(arr), axis=1), 1)

        return arr.tolist()

    @staticmethod
    def sub0_max(arr, library):

        """ subtract start value and divide by new maximum

        :returns array between 0 and 1
        """

        arr = arr - np.expand_dims(arr[:, 0], 1)
        arr = arr / np.expand_dims(library.max(np.abs(arr), axis=1), 1)

        return arr.tolist()

    @staticmethod
    def standardize(arr, library):

        """ subtract minimum and divide by new maximum

        :returns array between 0 and 1
        """

        arr = arr - np.expand_dims(library.mean(arr, axis=1), 1)
        arr = arr / np.expand_dims(library.std(np.abs(arr), axis=1), 1)

        return arr.tolist()
