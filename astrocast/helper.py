import glob
import inspect
import logging
import pickle
import platform
import random
import shutil
import tempfile
import time
import types
from pathlib import Path
from typing import List, Tuple, Union

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
from tqdm import tqdm


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
        custom_classes = [Events, Video, FeatureExtraction]
        
        try:
            from astrocast.denoising import SubFrameDataset
            custom_classes += [SubFrameDataset]
        except ImportError as err:
            logging.warning(f"Could not import package: {err}")
        
        if isinstance(arg, np.ndarray):
            return hash_from_ndarray(arg)
        
        elif isinstance(arg, (pd.DataFrame, pd.Series)):
            df_hash = pd.util.hash_pandas_object(arg)
            return hash_from_ndarray(df_hash.values)
        
        elif isinstance(arg, dict):
            return get_hash_from_dict(arg)
        
        elif isinstance(arg, tuple(custom_classes)):
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
        
        if cache_path is not None and isinstance(cache_path, (str, Path)):
            
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


class SignalGenerator:
    
    def __init__(self, parameter_fluctuations: float = 0,
                 signal_amplitude: float = None, noise_amplitude: float = 0.05, abort_amplitude=None,
                 allow_negative_values: bool = False, trace_length: Union[int, List[int], Tuple[int, int]] = None,
                 offset: Union[int, Tuple[int, int], Tuple[float, float], Tuple[None]] = None,
                 ragged_allowed: bool = True,
                 oscillation_frequency: int = 4, oscillation_amplitude: float = 1, plateau_duration: int = 1,
                 a: float = 0, k: float = 1, b: float = 1, v: float = 1, m_0: float = 0,
                 leaky_k: float = 0.1, leaky_n: float = 1,
                 ):
        """
        Initializes the SignalGenerator with parameters for the signal phases and noise level.

        Args:
            a: Left horizontal asymptote, representing the curve's minimum value.
            k: Right horizontal asymptote, representing the curve's maximum value.
            b: Growth rate; negative for inverted behavior.
            v: Shape parameter, biases the slope of the maximum growth rate.
            m_0: The value of `t` at which the maximum growth rate occurs; adjusts the curve's position along the x-axis.
            plateau_duration: Duration of the plateau phase in arbitrary units.
            oscillation_frequency: Frequency of oscillation within the plateau.
            oscillation_amplitude: Amplitude of the oscillations within the plateau.
            leaky_k: A constant controlling the rate of the leak.
            leaky_n: The exponent controlling how the leak rate scales with the state's value.
        """
        self.signal_amplitude = signal_amplitude
        self.noise_amplitude = noise_amplitude
        self.abort_amplitude = abort_amplitude
        self.allow_negative_values = allow_negative_values
        
        self.oscillation_frequency = oscillation_frequency
        self.oscillation_amplitude = oscillation_amplitude
        self.plateau_duration = plateau_duration
        
        self.leaky_k = leaky_k
        self.leaky_n = leaky_n
        
        self.trace_length = trace_length
        self.offset = offset
        self.ragged_allowed = ragged_allowed
        
        self.m_0 = m_0
        self.v = v
        self.b = b
        self.c = 1
        self.k = k
        self.a = a
        
        self.parameter_fluctuations = parameter_fluctuations
    
    @staticmethod
    def _richards_curve(t: Union[float, np.ndarray, int] = None, a: float = 0, k: float = 1,
                        c: float = 1, b: float = 1, v: float = 1, m_0: float = 6):
        """
        Calculate the first derivative of the Richards curve at time t or returns the derivative function of the Richards curve.

        The derivative of the Richards curve represents the rate of change of the growth process at time t.

        Args:
            t: Time point(s) at which the curve's derivative is evaluated. If None, returns the derivative function itself.
            a: Left horizontal asymptote, representing the curve's minimum value.
            k: Right horizontal asymptote, representing the curve's maximum value.
            c: Affects the rate of growth; often set to 1.
            b: Growth rate; negative for inverted behavior.
            v: Shape parameter, biases the slope of the maximum growth rate.
            m_0: The value of `t` at which the maximum growth rate occurs; adjusts the curve's position along the x-axis.

        Raises:
            ValueError: If `v` is not greater than 0.

        Returns:
            The first derivative of the Richards curve value(s) at time t, or the derivative function if t is None.
        """
        
        if v <= 0:
            raise ValueError(f"v={v} must be greater than 0.")
        
        def func(x):
            exponent = np.exp(-b * (x - m_0))
            return ((k - a) * b * exponent) / (v * (c + exponent) ** ((1 / v) + 1))
        
        if t is None:
            return func
        
        elif isinstance(t, (float, int, np.ndarray)):
            return func(t)
        
        else:
            raise ValueError(f"t must be float, int, np.ndarray or None")
    
    @staticmethod
    def _oscillatory_plateau(t: Union[float, np.ndarray, int] = None, a=1, plateau_duration: float = 4,
                             oscillation_frequency: float = 1.0, oscillation_amplitude: float = 0.1):
        """
        Generates an oscillatory plateau based on given parameters.
    
        Args:
            t: Time point(s) to calculate the oscillation. If None, returns the oscillatory function itself.
            plateau_duration: Duration of the plateau phase in arbitrary units.
            oscillation_frequency: Frequency of oscillation within the plateau.
            oscillation_amplitude: Amplitude of the oscillations within the plateau.
    
        Returns:
            The oscillatory plateau value(s) at time t, or the oscillatory function if t is None.
        """
        
        def func(x):
            
            if x > plateau_duration or plateau_duration == 0:
                return 0
            else:
                plateau = a + np.cos(2 * np.pi * oscillation_frequency * x) * oscillation_amplitude
                return plateau
        
        if t is None:
            return func
        
        elif isinstance(t, (float, int, np.ndarray)):
            return func(t)
        
        else:
            raise ValueError(f"t must be float, int, np.ndarray or None")
    
    @staticmethod
    def leaky_integrator(k: float = 0.1, n: float = 1):
        """
        Calculate the next state of a leaky integrator system where the leak rate increases with the state's value.

        Args:
            y: The current state of the system.
            k: A constant controlling the rate of the leak.
            n: The exponent controlling how the leak rate scales with the state's value.

        Returns:
            dy: the change in signal

        """
        
        # Calculate the rate of change of Y
        def func(x, reduce: float = 1):
            return -k * x ** n * reduce
        
        return func
    
    def noise_floor(self):
        noise = np.random.normal() * self.noise_amplitude
        return noise
    
    def _fluctuate_parameter(self, param):
        
        if param in [0, 1]:
            return param
        else:
            return np.random.normal(loc=param, scale=self.parameter_fluctuations)
    
    def generate_signal(self) -> Union[np.ndarray, None]:
        """
        Generates a calcium burst signal with added Gaussian noise.

        Returns:
          A numpy array representing the generated signal.
        """
        
        # rise parameters
        a = self._fluctuate_parameter(self.a)
        k = self._fluctuate_parameter(self.k)
        c = self._fluctuate_parameter(self.c)
        b = self._fluctuate_parameter(self.b)
        v = self._fluctuate_parameter(self.v)
        m_0 = self._fluctuate_parameter(self.m_0)
        
        # plateau parameters
        plateau_duration = self._fluctuate_parameter(self.plateau_duration)
        oscillation_frequency = self._fluctuate_parameter(self.oscillation_frequency)
        oscillation_amplitude = self._fluctuate_parameter(self.oscillation_amplitude)
        
        # define functions
        rise = self._richards_curve(a=a, k=k, c=c, b=b, v=v, m_0=m_0)
        leaky_integrator = self.leaky_integrator(k=self._fluctuate_parameter(self.leaky_k),
                                                 n=self._fluctuate_parameter(self.leaky_n))
        plateau = self._oscillatory_plateau(plateau_duration=plateau_duration,
                                            oscillation_frequency=oscillation_frequency,
                                            oscillation_amplitude=oscillation_amplitude)
        
        # define trace length
        length = self.trace_length
        if self.ragged_allowed:
            min_length, max_length = length if isinstance(length, (tuple, list)) else (1, length)
        else:
            min_length, max_length = length if isinstance(length, (tuple, list)) else (length, length)
            
            if max_length is not None and min_length != max_length:
                raise ValueError(f"min_length != max_length although ragged is not allowed:"
                                 f" {min_length} vs. {max_length}")
        
        if min_length is None:
            min_length = 1
        
        abort_amplitude = self.abort_amplitude if self.abort_amplitude is not None else self.noise_amplitude * 4
        
        # define offset
        offset_0, offset_1 = self.offset if isinstance(self.offset, (tuple, list)) else (self.offset, self.offset)
        
        # generate signal
        signal = []
        event_occurred = False
        v = 0
        t = 0
        plateau_start = 0
        phase = 0
        while True:
            
            # add noise
            v += self.noise_floor()
            
            # burst up
            r = rise(t)
            v += r
            
            # plateau
            if phase == 2 and plateau_duration > 0:
                v += plateau(t - plateau_start)
                v += leaky_integrator(v, reduce=abs(t - plateau_duration) / plateau_duration)
            
            # subtract leak
            else:
                v += leaky_integrator(v)
            
            # deal with negative values
            if not self.allow_negative_values and v < 0:
                v = 0
            
            # wait for burst to pick up speed
            if phase == 0 and v > k / 4:
                phase += 1
            
            # wait for burst to lose speed
            if phase == 1 and len(signal) > 1 and signal[-1] > v + self.noise_amplitude:
                phase += 1
                plateau_start = t
            
            # wait for plateau to end
            if phase == 2 and t > plateau_start + plateau_duration:
                phase += 1
            
            # abort conditions
            if max_length is not None and len(signal) >= max_length:
                break
            
            if phase >= 3 and abs(v) < abort_amplitude:
                break
            
            # save value
            signal.append(v)
            t += 1
        
        signal = np.array(signal)
        
        # add offset
        if offset_0 is not None:
            offset_0 = np.array([abs(self.noise_floor()) for _ in range(offset_0)])
            signal = np.concatenate((offset_0, signal), axis=0)
        
        if offset_1 is not None:
            offset_1 = np.array([abs(self.noise_floor()) for _ in range(offset_1)])
            signal = np.concatenate((signal, offset_1), axis=0)
        
        # enforce signal length
        if len(signal) < min_length:
            diff = min_length - len(signal)
            diff_0, diff_1 = int(diff / 2) + diff % 2, int(diff / 2)
            
            diff_0 = np.array([abs(self.noise_floor()) for _ in range(diff_0)])
            diff_1 = np.array([abs(self.noise_floor()) for _ in range(diff_1)])
            signal = np.concatenate((diff_0, signal, diff_1), axis=0)
        
        if max_length is not None and len(signal) > max_length:
            logging.warning(f"Generated signal is too long: {len(signal)} !<=  {max_length}")
            return signal[:max_length]
        
        # scale to signal amplitude
        if self.signal_amplitude is not None:
            signal = (signal / np.max(signal) * self._fluctuate_parameter(self.signal_amplitude))
        
        return signal
    
    def generate_signal_population(self, num_signals: int = 1) -> List[np.ndarray]:
        """
        Generates a population of signals with unique identifiers.

        Args:
          num_signals: Number of signals to generate.

        Returns:
          A list of generated signals.
        """
        signals = []
        for _ in tqdm(range(num_signals)):
            signal = self.generate_signal()
            
            if signal is not None:
                
                if self.trace_length is None:
                    signals.append(signal)
                
                elif isinstance(self.trace_length, (int, float)):
                    if len(signal) <= self.trace_length:
                        signals.append(signal)
                
                elif isinstance(self.trace_length, (tuple, list)):
                    min_, max_ = self.trace_length
                    if len(signal) >= min_ and len(signal) <= max_:
                        signals.append(signal)
        
        return signals


class DummyTensorFlow:
    class keras:
        class utils:
            class Sequence:
                def __init__(self, *args, **kwargs):
                    pass


class DummyGenerator:
    
    def __init__(
            self, num_rows=25, trace_length=12, ragged=False, offset=0, min_length=2,
            n_groups: int = 1, n_clusters=None,
            timings: List[List[int]] = None, timing_jitter: Union[int, Tuple[float, float]] = None,
            timing_offset: int = 0,
            z_range: Tuple[int, int] = None,
            generators: Union[SignalGenerator, List[SignalGenerator]] = None, ratios: List[float] = None
            ):
        
        self.generators = [generators] if isinstance(generators, SignalGenerator) else generators
        self.ratios = ratios
        self.ragged = True if (isinstance(ragged, str) and ragged == "ragged") else ragged
        self.groups = n_groups
        self.clusters = n_clusters
        
        self.timings = timings
        self.z_range = z_range
        self.timing_jitter = timing_jitter
        self.timing_offset = timing_offset
        
        # random traces
        if self.generators is None:
            self.data = self._get_random_data(num_rows=num_rows, trace_length=trace_length,
                                              ragged=self.ragged, min_length=min_length, offset=offset)
            
            if self.groups is not None:
                self.groups = np.random.randint(0, n_groups, size=len(self.data), dtype=int)
            
            if self.clusters is not None:
                self.clusters = np.random.randint(0, n_clusters, size=len(self.data), dtype=int)
        
        # signal populations
        else:
            self.data, self.groups = self._get_signal_population(generators=self.generators,
                                                                 num_rows=num_rows,
                                                                 ratios=self.ratios)
            
            # todo: implement differences between groups and clusters
    
    @staticmethod
    def _get_random_data(num_rows, trace_length, ragged, min_length, offset):
        
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
    
    @staticmethod
    def _get_signal_population(generators, num_rows, ratios: List[Tuple] = None,
                               shuffle: bool = True):
        """
        Generates a population of signals according to specified ratios for each generator.

        Args:
          num_rows: Total size of the population to generate.
          ratios: List of ratios for each generator. Must sum to 1.
          shuffle: randomize the population after generation

        Returns:
          A tuple of two lists: the generated signals and their identifiers.
        """
        
        # calculate ratios
        if ratios is None:
            equal_ratio = 1 / len(generators)
            ratios = [equal_ratio for _ in range(len(generators))]
        
        elif sum(ratios) != 1:
            diff = 1 - sum(ratios)
            ratios = list(ratios)
            ratios[-1] += diff
            logging.warning(f"ratios did not add up to 1; adjusted last entry to {ratios[-1]:.2f} by {diff:.2f}.")
        
        assert np.allclose(sum(ratios), 1, rtol=0.01), f"Ratios must sum to 1, not {sum(ratios)} > {ratios}."
        
        # generate population
        population = []
        identifiers = []
        for gen_idx, gen in enumerate(generators):
            
            num_signals = int(num_rows * ratios[gen_idx])
            signals = gen.generate_signal_population(num_signals)
            population.extend(signals)
            identifiers.extend([gen_idx] * num_signals)
        
        # shuffle
        if shuffle:
            indices = list(range(len(population)))
            random.shuffle(indices)
            
            population = [population[i] for i in indices]
            identifiers = [identifiers[i] for i in indices]
        
        return population, identifiers
    
    def _create_z_boundaries(self, df: pd.DataFrame) -> pd.DataFrame:
        
        timings = self.timings
        timing_jitter = self.timing_jitter
        timing_offset = self.timing_offset
        
        # create event length column
        df["dz"] = df.trace.apply(lambda x: len(x))
        
        # calculate z_range
        if self.z_range is None:
            z_range = (0, max(df.dz.sum() / 2, 1))
        else:
            z_range = self.z_range
        
        if timings is None:
            logging.info(f"Creating random z boundaries.")
            
            # create random boundaries
            dz_sum = int(df.dz.sum() / 2)
            df["z0"] = [np.random.randint(low=z_range[0], high=z_range[1]) for _ in range(len(df))]
            df["z1"] = df.z0 + df.dz
        
        else:
            
            if len(timings) != len(df.group.unique()):
                raise ValueError(f"Length of timings ({len(timings)}) does not equal "
                                 f"unique group ({len(df.group.unique())})")
            
            def apply_timings(row):
                
                # select timings associated with group
                group = int(row.group)
                timing = timings[group]
                
                if timing is None:
                    z0 = np.random.randint(low=z_range[0], high=z_range[1])
                else:
                    
                    if isinstance(timing_jitter, int):
                        jitter = np.random.randint(low=0, high=timing_jitter)
                    elif isinstance(timing_jitter, Tuple):
                        jitter = int(np.random.normal(loc=timing_jitter[0], scale=timing_jitter[1]))
                    else:
                        jitter = 0
                    
                    z0 = np.random.choice(timing) + jitter + timing_offset
                
                return z0
            
            # create boundaries
            df["z0"] = df.apply(func=apply_timings, axis='columns')
            df["z1"] = df.z0 + df.dz
        
        # create fake index
        df["idx"] = df.index
        
        return df
    
    def get_dataframe(self):
        
        data = self.data
        
        if type(data) == list:
            df = pd.DataFrame(dict(trace=data))
        
        elif type(data) == np.ndarray:
            df = pd.DataFrame(dict(trace=data.tolist()))
        else:
            raise TypeError
        
        if self.groups is not None:
            df["group"] = self.groups
        else:
            df["group"] = 0
        
        if self.clusters is not None:
            df["clusters"] = self.clusters
        else:
            df["cluster"] = 0
        
        df = self._create_z_boundaries(df=df)
        
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
        
        ev.events = df
        ev.seed = 1
        
        return ev
    
    def get_by_name(self, name, param={}):
        
        options = {"numpy":  self.get_array(**param), "dask": self.get_dask(**param), "list": self.get_list(**param),
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
    def _first(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the first value along the specified axis."""
        if population_wide:
            values = np.mean(x[:, 0] if summary_axis else x[0, :])
        else:
            values = x[:, 0] if summary_axis else x[0, :]
        
        if not population_wide:
            values = values[:, None]  # broadcasting for downstream calculation
        
        return values
    
    @staticmethod
    def _mean(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the mean value along the specified axis."""
        values = np.mean(x, axis=summary_axis)
        
        if not population_wide:
            values = values[:, None]  # broadcasting for downstream calculation
        
        return values
    
    @staticmethod
    def _median(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the median value along the specified axis."""
        values = np.median(x, axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
        return values
    
    @staticmethod
    def _min(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the minimum value along the specified axis."""
        values = np.min(x, axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
        return values
    
    @staticmethod
    def _min_abs(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the minimum of absolute values along the specified axis."""
        values = np.min(np.abs(x), axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
        return values
    
    @staticmethod
    def _max(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the maximum value along the specified axis."""
        values = np.max(x, axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
        return values
    
    @staticmethod
    def _max_abs(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the maximum of absolute values along the specified axis."""
        values = np.max(np.abs(x), axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
        return values
    
    @staticmethod
    def _std(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the standard deviation along the specified axis."""
        values = np.std(x, axis=summary_axis)
        if not population_wide:
            values = values[:, None]  # Broadcasting for downstream calculation
    
    @staticmethod
    def _std_mean(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the standard deviation along the specified axis."""
        
        if not population_wide:
            logging.warning(f"'_std_mean' can only be calculated for population wide characteristic; ignoring flag.")
        
        if summary_axis is None:
            summary_axis = 1
        
        values = np.std(x, axis=summary_axis)
        values = np.mean(values)
        
        return values
    
    @staticmethod
    def _std_median(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the standard deviation along the specified axis."""
        
        if not population_wide:
            logging.warning(f"'_std_mean' can only be calculated for population wide characteristic; ignoring flag.")
        
        if summary_axis is None:
            summary_axis = 1
        
        values = np.std(x, axis=summary_axis)
        values = np.median(values)
        
        return values
    
    @staticmethod
    def _std_max(x: Union[da.Array, np.ndarray, ak.Array], population_wide: bool = False, summary_axis: int = None):
        """Calculate the standard deviation along the specified axis."""
        
        if not population_wide:
            logging.warning(f"'_std_mean' can only be calculated for population wide characteristic; ignoring flag.")
        
        if summary_axis is None:
            summary_axis = 1
        
        values = np.std(x, axis=summary_axis)
        values = np.max(values)
        
        return values
    
    @staticmethod
    def get_value(data, mode, population_wide=False, axis=1):
        summary_axis = None if population_wide else axis
        
        # Dynamically create the mode_options dictionary from methods
        mode_options = {
            method_name[1:]: method
            for method_name, method in inspect.getmembers(Normalization, predicate=inspect.isfunction)
            if method_name.startswith("_")
            }
        
        assert mode in mode_options.keys(), f"please provide valid mode: {list(mode_options.keys())}"
        
        ret = mode_options[mode](data, population_wide, summary_axis)
        
        return ret
    
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
            folder_url, output=save_path.joinpath("models").as_posix(), quiet=False, use_cookies=False,
            remaining_ok=True
            )
    
    logging.info(f"Downloaded sample datasets to: {save_path}")
