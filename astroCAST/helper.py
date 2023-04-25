import glob
import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

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
        h.update(v)

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
                    logging.INFO("error during loading. recacalculating value", 0)
                    return f(*args, **kwargs)

                logging.INFO(f"loaded result of {f.__name__} from file", 1)

            else:

                result = f(*args, **kwargs)

                if len(files) > 0:
                    logging.INFO(f"multiple saves found. files should be deleted: {files}", 0)

                # save result
                logging.INFO(f"saving to: {cache_path}", 2)
                save_value(cache_path, result)

        else:

            result = f(*args, **kwargs)

        return result

    return inner_function
