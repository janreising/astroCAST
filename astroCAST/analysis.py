import copy
import logging
from functools import lru_cache
from pathlib import Path

import dask.array as da
import deprecation
import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
import napari

import astroCAST.detection
from astroCAST import helper
from astroCAST.helper import get_data_dimensions, wrapper_local_cache, is_ragged
from astroCAST.preparation import IO


class Events:

    def __init__(self, event_dir, meta_path=None, in_memory=False,
                 data=None, h5_loc=None,
                 z_slice=None, index_prefix=None, custom_columns=("area_norm", "cx", "cy"),
                 frame_to_time_mapping=None, frame_to_time_function=None):

        if event_dir is None:
            return None

        if not isinstance(event_dir, list):

            event_dir = Path(event_dir)
            if not event_dir.is_dir():
                raise FileNotFoundError(f"cannot find provided event directory: {event_dir}")

            if meta_path is not None:
                logging.debug("I should not be called at all")
                self.get_meta_info(meta_path)

            # get data
            if isinstance(data, (str, Path)):
                self.data = Video(data, z_slice=z_slice, h5_loc=h5_loc, lazy=False)

            if isinstance(data, (np.ndarray, da.Array)):

                if z_slice is not None:
                    logging.warning("'data'::array > Please ensure array was not sliced before providing data flag")

                self.data = Video(data, z_slice=z_slice, lazy=False)

            elif isinstance(data, Video):

                if z_slice is not None:
                    logging.warning("'data'::Video > Slice manually during Video object initialization")

                self.data = data

            else:
                self.data = data

            # load event map
            event_map, event_map_shape, event_map_dtype = self.get_event_map(event_dir, in_memory=in_memory) # todo slicing
            self.event_map = event_map
            self.num_frames, self.X, self.Y = event_map_shape

            # create time map
            time_map, events_start_frame, events_end_frame = self.get_time_map(event_dir)

            # load events
            self.events = self.load_events(event_dir, z_slice=z_slice, index_prefix=index_prefix, custom_columns=custom_columns)

            # z slicing
            self.z_slice = z_slice
            if z_slice is not None:
                z_min, z_max = z_slice
                self.events = self.events[(self.events.z0 >= z_min) & (self.events.z1 <= z_max)]

                # TODO how does this effect:
                #   - time_map, events_start_frame, events_end_frame
                #   - data
                #   - indices in the self.events dataframe

            self.z_range = (self.events.z0.min(), self.events.z1.max())

            # align time
            if frame_to_time_mapping is not None or frame_to_time_function is not None:
                self.events["t0"] = self.convert_frame_to_time(self.events.z0.tolist(),
                                                            mapping=frame_to_time_mapping,
                                                            function=frame_to_time_function)

                self.events["t1"] = self.convert_frame_to_time(self.events.z1.tolist(),
                                                            mapping=frame_to_time_mapping,
                                                            function=frame_to_time_function)

                self.events.dt = self.events.t1 - self.events.t0

        else:
            # multi file support

            event_objects = []
            for i in range(len(event_dir)):

                event = Events(event_dir[i],
                               meta_path=None if meta_path is None else meta_path[i],
                               data=None if data is None else data[i],
                               h5_loc=None if h5_loc is None else h5_loc[i],
                               z_slice=None if z_slice is None else z_slice[i],
                               in_memory=in_memory,
                               index_prefix=f"{i}x",
                               custom_columns=custom_columns,
                               frame_to_time_mapping=frame_to_time_mapping,
                               frame_to_time_function=frame_to_time_function)

                event_objects.append(event)

            self.event_objects = event_objects
            self.events = pd.concat([ev.events for ev in event_objects])
            self.z_slice = z_slice

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events.iloc[item]

    def copy(self):
        return copy.deepcopy(self)

    def filter(self, filters={}, inplace=True):

        events = self.events
        L1 = len(events)

        for column in filters:

            min_, max_ = filters[column]

            if min_ in [-1, None]:
                min_ = events[column].min() + 1

            if max_ in [-1, None]:
                max_ = events[column].max() + 1

            events = events[events[column].between(min_, max_, inclusive="both")]

        if inplace:
            self.events = events

        L2 = len(events)
        logging.info(f"#events: {L1} > {L2} ({L1/L2*100:.1f}%)")

        return events

    @staticmethod
    def get_event_map(event_dir, in_memory=False):

        """
        Retrieve the event map from the specified directory.

        Args:
            event_dir (str): The directory path where the event map is located.
            in_memory (bool, optional): Specifies whether to load the event map into memory. Defaults to False.

        Returns:
            tuple: A tuple containing the event map, its shape, and data type.

        """

        # Check if the event map is stored as a directory with 'event_map.tdb' file
        if Path(event_dir).joinpath("event_map.tdb").is_dir():
            path = Path(event_dir).joinpath("event_map.tdb")
            shape, chunksize, dtype = get_data_dimensions(path, return_dtype=True)

        # Check if the event map is stored as a file with 'event_map.tiff' extension
        elif Path(event_dir).joinpath("event_map.tiff").is_file():
            path = Path(event_dir).joinpath("event_map.tiff")
            shape, chunksize, dtype = get_data_dimensions(path, return_dtype=True)

        else:

            # Neither 'event_map.tdb' directory nor 'event_map.tiff' file found
            logging.warning(f"Cannot find 'event_map.tdb' or 'event_map.tiff'."
                            f"Consider recreating the file with 'create_event_map()', "
                            f"otherwise errors downstream might occur'.")
            shape, chunksize, dtype = (None, None, None), None, None
            event_map = None

            return event_map, shape, dtype

        # Load the event map from the specified path
        io = IO()
        event_map = io.load(path, lazy=not in_memory)

        return event_map, shape, dtype

    @staticmethod
    def create_event_map(events, video_dim, dtype=int, show_progress=True, save_path=None):
        """
        Create an event map from the events DataFrame.

        Args:
            events (DataFrame): The events DataFrame containing the 'mask' column.
            video_dim (tuple): The dimensions of the video in the format (num_frames, width, height).
            dtype (type, optional): The data type of the event map. Defaults to int.
            show_progress (bool, optional): Specifies whether to show a progress bar. Defaults to True.
            save_path (str, optional): The file path to save the event map. Defaults to None.

        Returns:
            ndarray: The created event map.

        Raises:
            ValueError: If 'mask' column is not present in the events DataFrame.

        """
        num_frames, width, height = video_dim
        event_map = np.zeros((num_frames, width, height), dtype=dtype)

        if "mask" not in events.columns:
            raise ValueError("Cannot recreate event_map without 'mask' column in events dataframe.")

        event_id = 1

        # Iterate over each event in the DataFrame
        iterator = tqdm(events.iterrows(), total=len(events)) if show_progress else events.iterrows()
        for _, event in iterator:
            # Extract the mask and reshape it to match event dimensions
            mask = np.reshape(event["mask"], (event.dz, event.dx, event.dy))

            # Find the indices where the mask is 1
            indices_z, indices_x, indices_y = np.where(mask == 1)

            # Adjust the indices to match the event_map dimensions
            indices_z += event.z0
            indices_x += event.x0
            indices_y += event.y0

            # Set the corresponding event_id at the calculated indices in event_map
            event_map[indices_z, indices_x, indices_y] = event_id
            event_id += 1

        if save_path is not None:
            # Save the event map to the specified path using IO()
            io = IO()
            io.save(save_path, data={"0": event_map.astype(float)})

        return event_map

    # @notimplemented("implement get_time_map()")
    @staticmethod
    def get_time_map(event_dir=None, event_map=None, chunk=100):
        """
        Creates a binary array representing the duration of events.

        Args:
            event_dir (Path): The directory containing the event data.
            event_map (ndarray): The event map data.
            chunk (int): The chunk size for processing events.

        Returns:
            Tuple: A tuple containing the time map, events' start frames, and events' end frames.
                time_map > binary array (num_events x num_frames) where 1 denotes event is active during that frame
                events_start_frame > 1D array (num_events x num_frames) of event start
                events_end_frame > 1D array (num_events x num_frames) of event end

        Raises:
            ValueError: If neither 'event_dir' nor 'event_map' is provided.

        """

        if event_dir is not None:

            if not event_dir.is_dir():
                raise FileNotFoundError(f"cannot find event_dir: {event_dir}")

            time_map_path = Path(event_dir).joinpath("time_map.npy")

            if time_map_path.is_file():
                time_map = np.load(time_map_path.as_posix(), allow_pickle=True)[()]

            else:
                raise ValueError(f"cannot find {time_map_path}. Please provide the event_map argument instead.")

        elif event_map is not None:

            if not isinstance(event_map, (np.ndarray, da.Array)):
                raise ValueError(f"please provide 'event_map' as np.ndarray or da")

            time_map = astroCAST.detection.Detector.get_time_map(event_map=event_map, chunk=chunk)

        else:
            raise ValueError("Please provide either 'event_dir' or 'event_map'.")

        # 1D array (num_events x frames) of event start
        events_start_frame = np.argmax(time_map, axis=0)

        # 1D array (num_events x frames) of event stop
        events_end_frame = time_map.shape[0] - np.argmax(time_map[::-1, :], axis=0)

        return time_map, events_start_frame, events_end_frame

    @staticmethod
    def load_events(event_dir, z_slice=None, index_prefix=None, custom_columns=["area_norm", "cx", "cy"]):

        """
        Load events from the specified directory and perform optional preprocessing.

        Args:
            event_dir (str): The directory containing the events.npy file.
            z_slice (tuple, optional): A tuple specifying the z-slice range to filter events.
            index_prefix (str, optional): A prefix to add to the event index.
            custom_columns (list, optional): A list of custom columns to compute for the events DataFrame.

        Returns:
            DataFrame: The loaded events DataFrame.

        Raises:
            FileNotFoundError: If 'events.npy' is not found in the specified directory.
            ValueError: If the custom_columns value is invalid.

        """

        path = Path(event_dir).joinpath("events.npy")
        if not path.is_file():
            raise FileNotFoundError(f"Did not find 'events.npy' in {event_dir}")

        events = np.load(path, allow_pickle=True)[()]
        logging.info(f"Number of events: {len(events)}")

        events = pd.DataFrame(events).transpose()
        events.sort_index(inplace=True)

        # Dictionary of custom column functions
        custom_column_functions = {
            "area_norm": lambda events: events.area / events.dz,
            # "pix_num_norm": lambda events: events.pix_num / events.dz,
            "area_footprint": lambda events: events.footprint.apply(sum),
            "cx": lambda events: events.x0 + events.dx * events["fp_centroid_local-0"],
            "cy": lambda events: events.y0 + events.dy * events["fp_centroid_local-1"]
        }

        if custom_columns is not None:

            if isinstance(custom_columns, str):
                custom_columns = [custom_columns]

            # Compute custom columns for the events DataFrame
            for custom_column in custom_columns:

                if isinstance(custom_column, dict):
                    column_name = list(custom_column.keys())[0]
                    func = custom_column[column_name]

                    events[column_name] = func(events)

                elif custom_column in custom_column_functions.keys():
                    func = custom_column_functions[custom_column]
                    events[custom_column] = func(events)
                else:
                    raise ValueError(f"Could not find 'custom_columns' value {custom_column}. "
                                     f"Please provide one of {list(custom_column_functions.keys())} or dict('column_name'=lambda events: ...)")

        if index_prefix is not None:
            events.index = ["{}{}".format(index_prefix, i) for i in events.index]

        if z_slice is not None:
            z0, z1 = z_slice
            events = events[(events.z0 >= z0) & (events.z1 <= z1)]

        return events

    # @wrapper_local_cache
    @staticmethod
    def get_extended_events(events, video, dtype=np.half, extend=-1,
                       normalization_instructions=None, show_progress=True,
                       memmap_path=None, save_path=None, save_param={}):

        """ takes the footprint of each individual event and extends it over the whole z-range

        example standardizing:

        normalization_instructions =
            0: ["subtract", {"mode": "mean"}],
            1: ["divide", {"mode": "std"}]
        }

        """

        n_events = len(events)
        n_frames, X, Y = video.shape

        # create array to save events in
        if memmap_path:
            memmap_path = Path(memmap_path).with_suffix(f".dtype_{np.dtype(dtype).name}_shape_{n_events}x{n_frames}.mmap")
            arr = np.memmap(memmap_path.as_posix(), dtype=dtype, mode='w+', shape=(n_events, n_frames))
        else:
            arr = np.zeros((n_events, n_frames), dtype=dtype)

        arr_size = arr.itemsize*n_events*n_frames
        ram_size = psutil.virtual_memory().total
        if arr_size > 0.9 * ram_size:
            logging.warning(f"array size ({n_events}, {n_frames}) is larger than 90% RAM size ({arr_size*1e-9:.2f}GB, {arr_size/ram_size*100}%). Consider using smaller dtype or 'lazy=True'")

        # extract footprints
        c = 0
        iterator = tqdm(events.iterrows(), total=len(events), desc="gathering footprints") if show_progress else events.iterrows()
        for i, event in iterator:

            # get z extend
            if extend == -1:
                z0, z1 = 0, n_frames
            elif isinstance(extend, (int)):
                dz0 = dz1 = extend
                z0, z1 = max(0, event.z0-dz0), min(event.z1+dz1, n_frames)
            elif isinstance(extend, (list, tuple)):

                if len(extend) != 2:
                    raise ValueError("provide 'extend' flag as int or tuple (ext_left, ext_right")

                dz0, dz1 = extend

                dz0 = n_frames if dz0 == -1 else dz0
                dz1 = n_frames if dz1 == -1 else dz1

                z0, z1 = max(0, event.z0-dz0), min(event.z1+dz1, n_frames)
            else:
                raise ValueError("provide 'extend' flag as int or tuple (ext_left, ext_right")

            # extract requested volume
            event_trace = video[z0:z1, event.x0:event.x1, event.y0:event.y1]

            # select event pixel
            footprint = np.invert(np.reshape(event["footprint"], (event.dx, event.dy)))
            mask = np.broadcast_to(footprint, event_trace.shape)

            # project to trace
            projection = np.ma.masked_array(data=event_trace, mask=mask)
            p = np.nanmean(projection, axis=(1, 2))

            if normalization_instructions is not None:
                norm = helper.Normalization(data=p, inplace=True)
                norm.run(normalization_instructions)

            arr[c, z0:z1] = p
            c += 1

        # TODO add trimming of empty left and right borders

        if save_path is not None and memmap_path is None:
            io = IO()
            io.save(path=save_path, data=arr, **save_param)

        return arr

    # @wrapper_local_cache
    def to_numpy(self, events=None, empty_as_nan=True):

        """
        Convert events DataFrame to a numpy array.

        Args:
            events (pd.DataFrame): The DataFrame containing event data with columns 'z0', 'z1', and 'trace'.
            empty_as_nan (bool): Flag to represent empty values as NaN.

        Returns:
            np.ndarray: The resulting numpy array.

        """

        if events is None:
            events = self.events

        num_frames = events.z1.max() + 1
        arr = np.zeros((len(events), num_frames))

        for i, (z0, z1, trace) in enumerate(zip(events.z0, events.z1, events.trace)):
            arr[i, z0:z1] = trace

        # todo this should actually be a mask instead then; np.nan creates weird behavior
        if empty_as_nan:
            arr[arr == 0] = np.nan

        return arr

    # @wrapper_local_cache
    def get_average_event_trace(self, events: pd.DataFrame = None, empty_as_nan: bool = True,
                                agg_func: callable = np.nanmean, index: list = None,
                                gradient: bool = False, smooth: int = None) -> pd.Series:
        """
        Calculate the average event trace.

        Args:
            events (pd.DataFrame): The DataFrame containing event data.
            empty_as_nan (bool): Flag to represent empty values as NaN.
            agg_func (callable): The function to aggregate the event traces.
            index (list): The index values for the resulting series.
            gradient (bool): Flag to calculate the gradient of the average trace.
            smooth (int): The window size for smoothing the average trace.

        Returns:
            pd.Series: The resulting average event trace.

        Raises:
            ValueError: If the provided 'agg_func' is not callable.

        """

        # Convert events DataFrame to a numpy array representation
        arr = self.to_numpy(events=events, empty_as_nan=empty_as_nan)

        # Check if agg_func is callable
        if not callable(agg_func):
            raise ValueError("Please provide a callable function for the 'agg_func' argument.")

        # Calculate the average event trace using the provided agg_func
        avg_trace = agg_func(arr, axis=0)

        if index is None:
            index = range(len(avg_trace))

        if smooth is not None:
            # Smooth the average trace using rolling mean
            avg_trace = pd.Series(avg_trace, index=index)
            avg_trace = avg_trace.rolling(smooth, center=True).mean()

        if gradient:
            # Calculate the gradient of the average trace
            avg_trace = np.gradient(avg_trace)

        avg_trace = pd.Series(avg_trace, index=index)

        return avg_trace

    @staticmethod
    def convert_frame_to_time(z, mapping=None, function=None):

        """
        Convert frame numbers to absolute time using a mapping or a function.

        Args:
            z (int or list): Frame number(s) to convert.
            mapping (dict): Dictionary mapping frame numbers to absolute time.
            function (callable): Function that converts a frame number to absolute time.

        Returns:
            float or list: Absolute time corresponding to the frame number(s).

        Raises:
            ValueError: If neither mapping nor function is provided.

        """

        if mapping is not None:

            if function is not None:
                logging.warning("function argument ignored, since mapping has priority.")

            if isinstance(z, int):
                return mapping[z]
            elif isinstance(z, list):
                return [mapping[frame] for frame in z]
            else:
                raise ValueError("Invalid 'z' value. Expected int or list.")

        elif function is not None:
            if isinstance(z, int):
                return function(z)
            elif isinstance(z, list):
                return [function(frame) for frame in z]
            else:
                raise ValueError("Invalid 'z' value. Expected int or list.")

        else:
            raise ValueError("Please provide either a mapping or a function.")

    def show_event_map(self, video=None, h5_loc=None, z_slice=None, lazy=True):

        viewer = napari.Viewer()

        if video is not None:
            io = IO()
            data = io.load(path=video, h5_loc=h5_loc, z_slice=z_slice, lazy=lazy)

            viewer.add_image(data, )

        event_map = self.event_map
        if z_slice is not None:
            event_map = event_map[z_slice[0]:z_slice[1], :, :]

        viewer.add_labels(event_map)

        return viewer

    def get_summary_statistics(self, decimals=2, groupby=None,
        columns_excluded=('z0', 'z1', 'x0', 'x1', 'y0', 'y1', 'dz', 'dx', 'dy', 'mask', 'contours', 'footprint', 'fp_cx', 'fp_cy', 'trace', 'error',  'cx', 'cy')):

        events = self.events

        # select columns
        if columns_excluded is not None:
            cols = [c for c in events.columns if c not in columns_excluded ]
            ev = events[cols]
        else:
            ev = events.copy()

        # cast to numbers
        ev = ev.astype(float)

        # grouping
        if groupby is not None:
            ev = ev.groupby(groupby)

        # calculate summary statistics
        mean, std = ev.mean(), ev.std()

        # combine mean and std
        val = mean.round(decimals).astype(str) + u" \u00B1 " + std.round(decimals).astype(str)

        if groupby is not None:
            val = val.transpose()

        return val

class Correlation:
    """
    A class for computing correlation matrices and histograms.
    """

    #todo local cache
    # @wrapper_local_cache
    @staticmethod
    def get_correlation_matrix(events, dtype=np.single):
        """
        Computes the correlation matrix of events.

        Args:
            events (np.ndarray or da.Array or pd.DataFrame): Input events data.
            dtype (np.dtype, optional): Data type of the correlation matrix. Defaults to np.single.
            mmap (bool, optional): Flag indicating whether to use memory-mapped arrays. Defaults to False.

        Returns:
            np.ndarray: Correlation matrix.

        Raises:
            ValueError: If events is not one of (np.ndarray, da.Array, pd.DataFrame).
            ValueError: If events DataFrame does not have a 'trace' column.
        """

        if not isinstance(events, (np.ndarray, pd.DataFrame, da.Array, Events)):
            raise ValueError(f"Please provide events as one of (np.ndarray, pd.DataFrame, Events) instead of {type(events)}.")

        if isinstance(events, Events):
            events = events.events

        if isinstance(events, pd.DataFrame):
            if "trace" not in events.columns:
                raise ValueError("Events DataFrame is expected to have a 'trace' column.")
            events = np.array(events["trace"].tolist())

        if is_ragged(events):

            logging.warning(f"Events are ragged (unequal length), default to slow correlation calculation.")

            N = len(events)
            corr = np.zeros((N, N), dtype=dtype)
            for x in tqdm(range(N)):
                for y in range(N):

                    if corr[y, x] == 0:

                        ex = events[x]
                        ey = events[y]

                        ex = ex - np.mean(ex)
                        ey = ey - np.mean(ey)

                        c = np.correlate(ex, ey, mode="valid")

                        # ensure result between -1 and 1
                        c = np.max(c)
                        c = c / (max(len(ex), len(ey) * np.std(ex) * np.std(ey)))

                        corr[x, y] = c

                    else:
                        corr[x, y] = corr[y, x]
        else:
            corr = np.corrcoef(events).astype(dtype)
            corr = np.tril(corr)

        return corr

    def _get_correlation_histogram(self, corr=None, events=None, start=-1, stop=1, num_bins=1000, density=False):
        """
        Computes the correlation histogram.

        Args:
            corr (np.ndarray, optional): Precomputed correlation matrix. If not provided, events will be used.
            events (np.ndarray or pd.DataFrame, optional): Input events data. Required if corr is not provided.
            start (float, optional): Start value of the histogram range. Defaults to -1.
            stop (float, optional): Stop value of the histogram range. Defaults to 1.
            num_bins (int, optional): Number of histogram bins. Defaults to 1000.
            density (bool, optional): Flag indicating whether to compute the histogram density. Defaults to False.

        Returns:
            np.ndarray: Correlation histogram counts.

        Raises:
            ValueError: If neither corr nor events is provided.
        """

        if corr is None:
            if events is None:
                raise ValueError("Please provide either 'corr' or 'events' flag.")
            corr = self.get_correlation_matrix(events)

        counts, _ = np.histogram(corr, bins=num_bins, range=(start, stop), density=density)

        return counts

    def plot_correlation_characteristics(self, corr=None, events=None, ax=None,
                                         perc=[5e-5, 5e-4, 1e-3, 1e-2, 0.05], bin_num=50, log_y=True,
                                         figsize=(10, 3)):
        """
        Plots the correlation characteristics.

        Args:
            corr (np.ndarray, optional): Precomputed correlation matrix. If not provided, footprint correlation is used.
            ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes, optional): Subplots axes to plot the figure.
            perc (list, optional): Percentiles to plot vertical lines on the cumulative plot. Defaults to [5e-5, 5e-4, 1e-3, 1e-2, 0.05].
            bin_num (int, optional): Number of histogram bins. Defaults to 50.
            log_y (bool, optional): Flag indicating whether to use log scale on the y-axis. Defaults to True.
            figsize (tuple, optional): Figure size. Defaults to (10, 3).

        Returns:
            matplotlib.figure.Figure: Plotted figure.

        Raises:
            ValueError: If ax is provided but is not a tuple of (ax0, ax1).
        """

        if corr is None:
            if events is None:
                raise ValueError("Please provide either 'corr' or 'events' flag.")
            corr = self.get_correlation_matrix(events)

        if ax is None:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
        else:
            if not isinstance(ax, (tuple, list, np.ndarray)) or len(ax) != 2:
                raise ValueError("'ax' argument expects a tuple/list/np.ndarray of (ax0, ax1)")

            ax0, ax1 = ax
            fig = ax0.get_figure()

        # Plot histogram
        bins = ax0.hist(corr.flatten(), bins=bin_num)
        if log_y:
            ax0.set_yscale("log")
        ax0.set_ylabel("Counts")
        ax0.set_xlabel("Correlation")

        # Plot cumulative distribution
        counts, xaxis, _ = bins
        counts = np.flip(counts)
        xaxis = np.flip(xaxis)
        cumm = np.cumsum(counts)
        cumm = cumm / np.sum(counts)

        ax1.plot(xaxis[1:], cumm)
        if log_y:
            ax1.set_yscale("log")
        ax1.invert_xaxis()
        ax1.set_ylabel("Fraction")
        ax1.set_xlabel("Correlation")

        # Plot vertical lines at percentiles
        pos = [np.argmin(abs(cumm - p)) for p in perc]
        vlines = [xaxis[p] for p in pos]
        for v in vlines:
            ax1.axvline(v, color="gray", linestyle="--")

        return fig

    def plot_compare_correlated_events(self, corr, events, event_ids=None,
                                   event_index_range=(0, -1), z_range=None,
                                   corr_mask=None, corr_range=None,
                                   ev0_color="blue", ev1_color="red", ev_alpha=0.5, spine_linewidth=3,
                                   ax=None, figsize=(20, 3), title=None):
        """
        Plot and compare correlated events.

        Args:
            corr (np.ndarray): Correlation matrix.
            events (pd.DataFrame, np.ndarray or Events): Events data.
            event_ids (tuple, optional): Tuple of event IDs to plot.
            event_index_range (tuple, optional): Range of event indices to consider.
            z_range (tuple, optional): Range of z values to plot.
            corr_mask (np.ndarray, optional): Correlation mask.
            corr_range (tuple, optional): Range of correlations to consider.
            ev0_color (str, optional): Color for the first event plot.
            ev1_color (str, optional): Color for the second event plot.
            ev_alpha (float, optional): Alpha value for event plots.
            spine_linewidth (float, optional): Linewidth for spines.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on.
            figsize (tuple, optional): Figure size.
            title (str, optional): Plot title.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if isinstance(events, Events):
            events = events.events

        # Validate event_index_range
        if not isinstance(event_index_range, (tuple, list)) or len(event_index_range) != 2:
            raise ValueError("Please provide event_index_range as a tuple of (start, stop)")

        # Convert events to numpy array if it is a DataFrame
        if isinstance(events, pd.DataFrame):
            if "trace" not in events.columns:
                raise ValueError("'events' dataframe is expected to have a 'trace' column.")

            events = np.array(events.trace.tolist())

        ind_min, ind_max = event_index_range
        if ind_max == -1:
            ind_max = len(events)

        # Choose events
        if event_ids is None:
            # Randomly choose two events if corr_mask and corr_range are not provided
            if corr_mask is None and corr_range is None:
                ev0, ev1 = np.random.randint(ind_min, ind_max, size=2)

            # Choose events based on corr_mask
            elif corr_mask is not None:
                # Warn if corr_range is provided and ignore it
                if corr_range is not None:
                    logging.warning("Prioritizing 'corr_mask'; ignoring 'corr_range' argument.")

                if isinstance(corr_mask, (list, tuple)):
                    corr_mask = np.array(corr_mask)

                    if corr_mask.shape[0] != 2:
                        raise ValueError(f"corr_mask should have a shape of (2xN) instead of {corr_mask.shape}")

                rand_index = np.random.randint(0, corr_mask.shape[1])
                ev0, ev1 = corr_mask[:, rand_index]

            # Choose events based on corr_range
            elif corr_range is not None:
                # Validate corr_range
                if len(corr_range) != 2:
                    raise ValueError("Please provide corr_range as a tuple of (min_corr, max_corr)")

                corr_min, corr_max = corr_range

                # Create corr_mask based on corr_range
                corr_mask = np.array(np.where(np.logical_and(corr >= corr_min, corr <= corr_max)))
                logging.warning("Thresholding the correlation array may take a long time. Consider precalculating the 'corr_mask' with eg. 'np.where(np.logical_and(corr >= corr_min, corr <= corr_max))'")

                rand_index = np.random.randint(0, corr_mask.shape[1])
                ev0, ev1 = corr_mask[:, rand_index]

        else:
            ev0, ev1 = event_ids

        if isinstance(ev0, np.ndarray):
            ev0 = ev0[0]
            ev1 = ev1[0]

        # Choose z range
        trace_0 = np.squeeze(events[ev0]).astype(float)
        trace_1 = np.squeeze(events[ev1]).astype(float)

        if isinstance(trace_0, da.Array):
            trace_0 = trace_0.compute()
            trace_1 = trace_1.compute()

        if z_range is not None:
            z0, z1 = z_range

            if (z0 > len(trace_0)) or (z0 > len(trace_1)):
                raise ValueError(f"Left bound z0 larger than event length: {z0} > {len(trace_0)} or {len(trace_1)}")

            trace_0 = trace_0[z0: min(z1, len(trace_0))]
            trace_1 = trace_1[z0: min(z1, len(trace_1))]

        ax.plot(trace_0, color=ev0_color, alpha=ev_alpha)
        ax.plot(trace_1, color=ev1_color, alpha=ev_alpha)

        if title is None:
            if isinstance(ev0, np.ndarray):
                ev0 = ev0[0]
                ev1 = ev1[0]
            ax.set_title("{:,d} x {:,d} > corr: {:.4f}".format(ev0, ev1, corr[ev0, ev1]))

        def correlation_color_map(colors=None):
            """
            Create a correlation color map.

            Args:
                colors (list, optional): List of colors.

            Returns:
                function: Color map function.
            """
            if colors is None:
                neg_color = (0, "#ff0000")
                neu_color = (0.5, "#ffffff")
                pos_color = (1, "#0a700e")

                colors = [neg_color, neu_color, pos_color]

            cm = LinearSegmentedColormap.from_list("Custom", colors, N=200)

            def lsc(v):
                assert np.abs(v) <= 1, "Value must be between -1 and 1: {}".format(v)

                if v == 0:
                    return cm(100)
                if v < 0:
                    return cm(100 - int(abs(v) * 100))
                elif v > 0:
                    return cm(int(v * 100 + 100))

            return lsc

        lsc = correlation_color_map()
        for spine in ax.spines.values():
            spine.set_edgecolor(lsc(corr[ev0, ev1]))
            spine.set_linewidth(spine_linewidth)

        return fig

class Video:

    def __init__(self, data, z_slice=None, h5_loc=None, lazy=False):

        if isinstance(data, (np.ndarray, da.Array)):
            self.data = data

            if z_slice is not None:
                z0, z1 = z_slice
                self.data = self.data[z0:z1, :, :]

        elif isinstance(data, (str, Path)):

            io = IO()
            self.data = io.load(data, h5_loc=h5_loc, lazy=lazy, z_slice=z_slice)

        self.z_slice = z_slice
        self.Z, self.X, self.Y = self.data.shape

    def get_data(self, in_memory=False):

        if in_memory and isinstance(self.data, da.Array):
            return self.data.compute()

        else:
            return self.data

    # @lru_cache(maxsize=None)
    # @wrapper_local_cache
    def get_image_project(self, agg_func=np.mean, window=None, window_agg=np.sum,
                          show_progress=True):

        img = self.data

        # calculate projection
        if window is None:
            proj = agg_func(img, axis=0)

        else:

            from numpy.lib.stride_tricks import sliding_window_view

            Z, X, Y = img.shape
            proj = np.zeros((X, Y))

            z_step = int(window/2)
            for x in tqdm(range(X)) if show_progress else range(X):
                for y in range(Y):

                    slide_ = sliding_window_view(img[:, x, y], axis=0, window_shape = window) # sliding trick
                    slide_ = slide_[::z_step, :] # skip most steps
                    agg = agg_func(slide_, axis=1) # aggregate
                    proj[x, y] = window_agg(agg) # window aggregate

        return proj

    def show(self):
        return napari.view_image(self.data)

class Plotting:

    def __init__(self, events):
        self.events = events.events

    @staticmethod
    def _get_factorials(nr):
        """
        Returns the factors of a number.

        Args:
            nr (int): Number.

        Returns:
            list: List of factors.

        """
        i = 2
        factors = []
        while i <= nr:
            if (nr % i) == 0:
                factors.append(i)
                nr = nr / i
            else:
                i = i + 1
        return factors

    def _get_square_grid(self, N, figsize=(4, 4), figsize_multiply=4, sharex=False, sharey=False, max_n=5, switch_dim=False):
        """
        Returns a square grid of subplots in a matplotlib figure.

        Args:
            N (int): Number of subplots.
            figsize (tuple, optional): Figure size in inches. Defaults to (4, 4).
            figsize_multiply (int, optional): Factor to multiply figsize by when figsize='auto'. Defaults to 4.
            sharex (bool, optional): Whether to share the x-axis among subplots. Defaults to False.
            sharey (bool, optional): Whether to share the y-axis among subplots. Defaults to False.
            max_n (int, optional): Maximum number of subplots per row when there is only one factor. Defaults to 5.
            switch_dim (bool, optional): Whether to switch the dimensions of the grid. Defaults to False.

        Returns:
            tuple: A tuple containing the matplotlib figure and a list of axes.

        """

        # Get the factors of N
        f = self._get_factorials(N)

        if len(f) < 1:
            # If no factors found, set grid dimensions to 1x1
            nx = ny = 1

        elif len(f) == 1:

            if f[0] > max_n:
                # If only one factor and it exceeds max_n, set grid dimensions to ceil(sqrt(N))
                nx = ny = int(np.ceil(np.sqrt(N)))

            else:
                # If only one factor and it doesn't exceed max_n, set grid dimensions to that factor x 1
                nx = f[0]
                ny = 1

        elif len(f) == 2:
            # If two factors, set grid dimensions to those factors
            nx, ny = f

        elif len(f) == 3:
            # If three factors, set grid dimensions to factor1 x factor2 and factor3
            nx = f[0] * f[1]
            ny = f[2]

        elif len(f) == 4:
            # If four factors, set grid dimensions to factor1 x factor2 and factor3 x factor4
            nx = f[0] * f[1]
            ny = f[2] * f[3]

        else:
            # For more than four factors, set grid dimensions to ceil(sqrt(N))
            nx = ny = int(np.ceil(np.sqrt(N)))

        if figsize == "auto":
            # If figsize is set to "auto", calculate figsize based on the dimensions of the grid
            figsize = (ny * figsize_multiply, nx * figsize_multiply)

        # Switch dimensions if necessary
        if switch_dim:
            nx, ny = ny, nx

        # Create the figure and axes grid
        fig, axx = plt.subplots(nx, ny, figsize=figsize, sharex=sharex, sharey=sharey)

        # Convert axx to a list if N is 1, otherwise flatten the axx array and convert to a list
        axx = [axx] if N == 1 else list(axx.flatten())

        new_axx = []
        for i, ax in enumerate(axx):
            # Remove excess axes if N is less than the total number of axes created
            if i >= N:
                fig.delaxes(ax)
            else:
                new_axx.append(ax)

        # Adjust the spacing between subplots
        fig.tight_layout()

        return fig, new_axx

    def _get_random_sample(self, num_samples):
        """
        Get a random sample of traces from the events.

        Args:
            num_samples (int): Number of samples to retrieve.

        Returns:
            list: List of sampled traces.

        Raises:
            ValueError: If the events data type is not one of pandas.DataFrame, numpy.ndarray, or list.

        """

        events = self.events

        if num_samples == -1:
            return events

        if isinstance(events, pd.DataFrame):
            # If events is a pandas DataFrame, sample num_samples rows and retrieve the trace values
            sel = events.sample(num_samples)
            traces = sel.trace.values

        elif isinstance(events, np.ndarray):
            # If events is a numpy ndarray, generate random indices and retrieve the corresponding trace values
            idx = np.random.randint(0, len(events), size=num_samples)
            traces = events[idx, :, 0]

        elif isinstance(events, list):
            # If events is a list, generate random indices and retrieve the corresponding events
            idx = np.random.randint(0, len(events), size=num_samples)
            traces = [events[id_] for id_ in idx]

        else:
            # If events is neither a pandas DataFrame, numpy ndarray, nor list, raise a ValueError
            raise ValueError("Please provide one of the following data types: pandas.DataFrame, numpy.ndarray, or list. "
                             f"Instead of {type(events)}")

        return traces

    # todo clustering
    def plot_traces(self, num_samples=-1, ax=None, figsize=(5, 5)):

        traces = self._get_random_sample(num_samples=num_samples)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        for i, trace in enumerate(traces):
            ax.plot(trace, label=i)

        plt.tight_layout()

        return fig

    def plot_distribution(self, column, plot_func=sns.violinplot, outlier_deviation=None,
                          axx=None, figsize=(8, 3), title=None):

        values = self.events[column]

        # filter outliers
        if outlier_deviation is not None:

            mean, std = values.mean(), values.std()

            df_low = values[values < mean - outlier_deviation * std]
            df_high = values[values > mean + outlier_deviation * std]
            df_mid = values[values.between(mean - outlier_deviation * std, mean + outlier_deviation * std)]

            num_panels = 3

        else:
            df_low = df_high = None
            df_mid = values
            num_panels = 1

        # create figure if necessary
        if axx is None:
            _, axx = self._get_square_grid(num_panels, figsize=figsize, switch_dim=True)

        # make sure axx can be indexed
        if not isinstance(axx, list):
            axx = [axx]

        # plot distribution
        plot_func(df_mid.values, ax=axx[0])
        axx[0].set_title(f"Distribution {column}")

        if outlier_deviation is not None:

            # plot outlier number
            if len(axx) != 3:
                raise ValueError(f"when providing outlier_deviation, len(axx) is expected to be 3 (not: {len(axx)}")

            count = pd.DataFrame({"count": [len(df_low), len(df_mid), len(df_high)], "type":["low", "mid", "high"]})
            sns.barplot(data=count, y="count", x="type", ax=axx[1])
            axx[1].set_title("Outlier count")

            # plot swarm plot
            sns.swarmplot(data=pd.concat((df_low, df_high)),
                          marker="x", linewidth=2, color="red", ax=axx[2])
            axx[2].set_title("Outliers")

        # figure title
        if title is not None:
            axx[0].get_figure().suptitle(title)

        plt.tight_layout()

        return axx[0].get_figure()
