import copy
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import dask.array as da
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
import numpy as np
import pandas as pd
import psutil
import xxhash
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import napari
from napari.utils.events import Event
import awkward as ak

import astroCAST.detection
from astroCAST import helper
from astroCAST.helper import get_data_dimensions, is_ragged, CachedClass, Normalization, wrapper_local_cache
from astroCAST.preparation import IO


class Events(CachedClass):

    def __init__(self, event_dir, lazy=True, data=None, h5_loc=None, group=None, subject_id=None,
                 z_slice=None, index_prefix=None, custom_columns=("area_norm", "cx", "cy"), frame_to_time_mapping=None,
                 frame_to_time_function=None, cache_path=None, seed=1):

        super().__init__(cache_path=cache_path)

        self.seed = seed
        self.z_slice = z_slice

        if not isinstance(event_dir, list): # single file support

            if event_dir is None:
                logging.warning("event_dir is None. Creating empty Events instance!")
                self.event_map = None
                self.num_frames, self.X, self.Y = None, None, None
                self.events = None

            else:

                event_dir = Path(event_dir)
                if not event_dir.is_dir():
                    raise FileNotFoundError(f"cannot find provided event directory: {event_dir}")

                # load event map
                event_map, event_map_shape, event_map_dtype = self.get_event_map(event_dir, lazy=lazy) # todo slicing
                self.event_map = event_map
                self.num_frames, self.X, self.Y = event_map_shape

                # create time map
                # time_map, events_start_frame, events_end_frame = self.get_time_map(event_dir=event_dir, event_map=event_map)

                # load events
                self.events = self.load_events(event_dir, z_slice=z_slice, index_prefix=index_prefix, custom_columns=custom_columns)

                # z slicing
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

                # add group columns
                self.events["group"] = group
                self.events["subject_id"] = subject_id
                self.events["name"] = event_dir.stem

            # get data
            if isinstance(data, (str, Path)):
                self.data = Video(data, z_slice=z_slice, h5_loc=h5_loc, lazy=False)

            elif isinstance(data, (np.ndarray, da.Array)):

                if z_slice is not None:
                    logging.warning("'data'::array > Please ensure array was not sliced before providing data flag")

                self.data = Video(data, z_slice=z_slice, lazy=False)

            elif isinstance(data, Video):

                if z_slice is not None:
                    logging.warning("'data'::Video > Slice manually during Video object initialization")

                self.data = data

            else:
                self.data = data

        else: # multi file support

            event_objects = []
            for i in range(len(event_dir)):

                event = Events(event_dir[i],
                               data=data if not isinstance(data, list) else data[i],
                               h5_loc=h5_loc if not isinstance(h5_loc, list) else h5_loc[i],
                               z_slice=z_slice if not isinstance(z_slice, list) else z_slice[i],
                               group=group if not isinstance(group, list) else group[i],
                               lazy=lazy,
                               index_prefix=f"{i}x", subject_id=i,
                               custom_columns=custom_columns,
                               frame_to_time_mapping=frame_to_time_mapping if not isinstance(frame_to_time_mapping, list) else frame_to_time_mapping[i],
                               frame_to_time_function=frame_to_time_function if not isinstance(frame_to_time_function, list) else frame_to_time_function[i])

                event_objects.append(event)

            self.event_objects = event_objects
            self.events = pd.concat([ev.events for ev in event_objects])
            self.events.reset_index(drop=False, inplace=True, names="idx")
            self.z_slice = z_slice

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events.iloc[item]

    def __hash__(self):

        traces = self.events.trace

        hashed_traces = traces.apply(lambda x: xxhash.xxh64_intdigest(np.array(x), seed=self.seed))
        hash_ = xxhash.xxh64_intdigest(hashed_traces.values, seed=self.seed)

        return hash_

    def _repr_html_(self):
        return self.events._repr_html_()

    def is_multi_subject(self):
        if len(self.events.subject_id.unique()) > 1:
            return True
        else:
            return False

    def is_ragged(self):
        return is_ragged(self.events.trace.tolist())

    def add_clustering(self, cluster_lookup_table, column_name="cluster"):

        events = self.events

        if column_name in events.columns:
            logging.warning(f"column_name ({column_name}) already exists in events table > overwriting column. "
                            f"Please provide a different column_name if this is not the expected behavior.")

        events[column_name] = events.index.map(cluster_lookup_table)

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
        logging.info(f"#events: {L1} > {L2} ({L2/L1*100:.1f}%)")

        return events

    @staticmethod
    def get_event_map(event_dir:Path, lazy=True):

        """
        Retrieve the event map from the specified directory.

        Args:
            event_dir (Path): The directory path where the event map is located.
            in_memory (bool, optional): Specifies whether to load the event map into memory. Defaults to False.

        Returns:
            tuple: A tuple containing the event map, its shape, and data type.

        """

        # Check if the event map is stored as a directory with 'event_map.tdb' file
        if event_dir.joinpath("event_map.tdb").is_dir():
            path = event_dir.joinpath("event_map.tdb")
            shape, chunksize, dtype = get_data_dimensions(path, return_dtype=True)

        # Check if the event map is stored as a file with 'event_map.tiff' extension
        elif event_dir.joinpath("event_map.tiff").is_file():
            path = event_dir.joinpath("event_map.tiff")
            shape, chunksize, dtype = get_data_dimensions(path, return_dtype=True)

        else: # Neither 'event_map.tdb' directory nor 'event_map.tiff' file found

            logging.warning(f"Cannot find 'event_map.tdb' or 'event_map.tiff'."
                            f"Consider recreating the file with 'create_event_map()', "
                            f"otherwise errors downstream might occur'.")
            shape, chunksize, dtype = (None, None, None), None, None
            event_map = None

            return event_map, shape, dtype

        # Load the event map from the specified path
        io = IO()
        event_map = io.load(path, lazy=lazy)

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

    @wrapper_local_cache
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

            elif event_map is not None:
                time_map = astroCAST.detection.Detector.get_time_map(event_map=event_map, chunk=chunk)
                np.save(time_map_path.as_posix(), time_map)

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
    def load_events(event_dir:Path, z_slice=None, index_prefix=None, custom_columns=("area_norm", "cx", "cy")):

        """
        Load events from the specified directory and perform optional preprocessing.

        Args:
            event_dir (Path): The directory containing the events.npy file.
            z_slice (tuple, optional): A tuple specifying the z-slice range to filter events.
            index_prefix (str, optional): A prefix to add to the event index.
            custom_columns (list, optional): A list of custom columns to compute for the events DataFrame.

        Returns:
            DataFrame: The loaded events DataFrame.

        Raises:
            FileNotFoundError: If 'events.npy' is not found in the specified directory.
            ValueError: If the custom_columns value is invalid.

        """

        path = event_dir.joinpath("events.npy")
        if not path.is_file():
            raise FileNotFoundError(f"Did not find 'events.npy' in {event_dir}")

        events = np.load(path.as_posix(), allow_pickle=True)[()]
        logging.info(f"Number of events: {len(events)}")

        events = pd.DataFrame(events).transpose()
        events.sort_index(inplace=True)

        # Dictionary of custom column functions
        custom_column_functions = {
            "area_norm": lambda events: events.area / events.dz,
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

    @wrapper_local_cache
    def get_extended_events(self, video=None, dtype=np.half, extend=-1,
                            return_array=False, in_place=False,
                            normalization_instructions=None, show_progress=True,
                            memmap_path=None, save_path=None, save_param={}):

        """ takes the footprint of each individual event and extends it over the whole z-range

        example standardizing:

        normalization_instructions =
            0: ["subtract", {"mode": "mean"}],
            1: ["divide", {"mode": "std"}]
        }

        """

        events = self.events if in_place else self.events.copy()
        n_events = len(events)

        # load data
        if video is not None:
            video = video.get_data()
        elif self.data is not None:
            video = self.data.get_data()
        else:
            raise ValueError("to extend the event traces you either have to provide the 'video' argument "
                             "when calling this function or the 'data' argument during Event creation.")

        # get video dimensions
        n_frames, X, Y = video.shape

        # create container to save extended events in
        arr_ext, extended = None, None
        if return_array:

            # create array
            if memmap_path:
                memmap_path = Path(memmap_path).with_suffix(f".dtype_{np.dtype(dtype).name}_shape_{n_events}x{n_frames}.mmap")
                arr_ext = np.memmap(memmap_path.as_posix(), dtype=dtype, mode='w+', shape=(n_events, n_frames))
            else:
                arr_ext = np.zeros((n_events, n_frames), dtype=dtype)

            arr_size = arr_ext.itemsize*n_events*n_frames
            ram_size = psutil.virtual_memory().total
            if arr_size > 0.9 * ram_size:
                logging.warning(f"array size ({n_events}, {n_frames}) is larger than 90% RAM size ({arr_size*1e-9:.2f}GB, {arr_size/ram_size*100}%). Consider using smaller dtype or providing a 'mmemap_path'")

        else:
            extended = list()

        z0_container, z1_container = list(), list()

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

            if return_array:
                arr_ext[c, z0:z1] = p
            else:
                extended.append(p)

            z0_container.append(z0)
            z1_container.append(z1)

            c += 1

        if return_array:

            if memmap_path is not None:
                logging.info(f"'save_path' ignored. Extended array saved as memmap to :{memmap_path}")

            elif save_path is not None:
                io = IO()
                io.save(path=save_path, data=arr_ext, **save_param)

            return arr_ext, z0_container, z1_container

        else:

            events.trace = extended

            # save a copy of original z frames
            events["z0_orig"] = events.z0
            events["z1_orig"] = events.z1
            events["dz_orig"] = events.dz

            # update current z frames
            events.z0 = z0_container
            events.z1 = z1_container
            events.dz = events["z1"] - events["z0"]

            return events

    def to_numpy(self, events=None, empty_as_nan=True, simple=False):

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

        if simple:
            return np.array(events.trace.tolist())

        num_frames = events.z1.max() + 1
        arr = np.zeros((len(events), num_frames))

        for i, (z0, z1, trace) in enumerate(zip(events.z0, events.z1, events.trace)):
            arr[i, z0:z1] = trace

        # todo this should actually be a mask instead then; np.nan creates weird behavior
        if empty_as_nan:
            arr[arr == 0] = np.nan

        return arr

    @lru_cache
    def to_tsfresh(self, show_progress=False):

        iterator = self.events.trace.items()
        iterator = tqdm(iterator, total=len(self.events)) if show_progress else iterator

        logging.info("creating tsfresh dataset ...")
        ids, times, dim_0s = [], [], []
        for id_, trace in iterator:

            if type(trace) != np.ndarray:
                trace = np.array(trace)

            # take care of NaN
            trace = np.nan_to_num(trace)

            ids = ids + [id_]*len(trace)
            times = times + list(range(len(trace)))
            dim_0s = dim_0s + list(trace)

        X = pd.DataFrame({"id":ids, "time":times, "dim_0":dim_0s})
        return X

    @wrapper_local_cache
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

    @wrapper_local_cache
    def get_summary_statistics(self, decimals=2, groupby=None,
        columns_excluded=('name', 'subject_id', 'group', 'z0', 'z1', 'x0', 'x1', 'y0', 'y1', 'mask', 'contours', 'footprint', 'fp_cx', 'fp_cy', 'trace', 'error',  'cx', 'cy')):

        events = self.events

        # select columns
        if columns_excluded is not None:
            cols = [c for c in events.columns if c not in columns_excluded ] + [] if groupby is None else [groupby]
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

    @wrapper_local_cache
    def get_trials(self, trial_timings, trial_length=30, multi_timing_behavior="first", format="array"):

        if format not in ["array", "dataframe"]:
            raise ValueError(f"'format' attribute has to be one of ['array', 'dataframe'] not: {format}")

        if multi_timing_behavior not in ["first", "expand", "exclude"]:
            raise ValueError("'multi_timing_behavior' has to be one of ['first', 'expand', 'exclude']")

        events = self.events.copy()

        # convert timings to np.ndarray
        if not isinstance(trial_timings, np.ndarray):
            trial_timings = np.array(trial_timings)

        # split trial_length in pre and post
        leading = trailing = int(trial_length / 2)
        leading += trial_length - (leading + trailing)

        # get contained timings per event
        def find_contained_timings(row):
            mask = np.logical_and(trial_timings >= row.z0, trial_timings <= row.z1)

            num_timings = np.sum(mask)
            contained_timings = trial_timings[mask]
            return tuple(contained_timings)

        events["timings"] = events.apply(find_contained_timings, axis=1)
        events["num_timings"] = events.timings.apply(lambda x: len(x))

        # decide what happens if multiple timings happen during a single event
        if multi_timing_behavior == "first":
            events = events[events.num_timings > 0]

            # print(f"num_timings:\n{events.num_timings}")
            # print(f"timings:\n{events.timings}")

            events.timings = events.timings.apply(lambda x: [x[0]])
            num_rows = len(events)

        elif multi_timing_behavior == "expand":
            events = events[events.num_timings > 0]
            num_rows = events.num_timings.sum()

        elif multi_timing_behavior == "exclude":
            events = events[events.num_timings == 1]
            num_rows = len(events)

        # print(f"num_timings:\n{events.num_timings}")
        # print(f"timings:\n{events.timings}")

        #create trial matrix
        array = np.empty((num_rows, trial_length))

        # fill array
        i = 0
        for ev_idx, row in events.iterrows():
            for t in row.timings:

                # get boundaries
                z0, z1 = row.z0, row.z1
                t0, t1 = t - leading, t + trailing
                delta_left, delta_right = t0-z0, t1-z1

                # calculate offsets
                eve_idx_left = max(0, delta_left)
                eve_idx_right = delta_right if delta_right < 0 else None

                # arr_idx_left = min(0, delta_right)
                # arr_idx_right = delta_left if delta_left < 0 else None

                arr_idx_left = max(0, -delta_left)
                arr_idx_right = -delta_right if -delta_right < 0 else None

                # print(f"\n{ev_idx} (#{len(row.trace)}) - stimulus_t: {t}")
                # print(f"z: {z0}-{z1}, t:{t0}-{t1}, d:{delta_left}-{delta_right}")
                # print(f"event_idx: {eve_idx_left}:{eve_idx_right} (#{len(np.array(row.trace)[eve_idx_left:eve_idx_right])})")
                # print(f"arr_idx: {arr_idx_left}:{arr_idx_right} (#{len(array[i, arr_idx_left:arr_idx_right])})")
                # print(np.array(row.trace))

                # splice event into array
                array[i, arr_idx_left:arr_idx_right] = np.array(row.trace)[eve_idx_left:eve_idx_right]

                i += 1

        if format == "dataframe":

            t_range = list(range(-leading, trailing))

            trial_ids = []
            values = []
            timepoints = []
            for row in range(len(array)):

                trial_ids += [row]*trial_length
                values += array[row, :].tolist()
                timepoints += t_range

            res = pd.DataFrame({"trial_ids":trial_ids, "timepoint":timepoints, "value":values})

        else:
            res = array

        return res

    def enforce_length(self, min_length=None, pad_mode="edge", max_length=None, inplace=False):

        if inplace:
            events = self.events
        else:
            events = self.events.copy()

        data = events.trace.tolist()

        if min_length is not None and max_length is not None:

            if is_ragged(data):

                # # todo this implementation would be more efficient, but somehow doesn't work
                # data = ak.Array(data)
                #
                # if min_length is not None and max_length is None:
                #     data = ak.pad_none(data, min_length)
                #
                # elif max_length is not None and min_length is None:
                #     data = data[:, :max_length]
                #
                # else:
                #     assert max_length == min_length, "when providing 'max_length' and 'min_length', both have to be equal"
                #     data = ak.pad_none(data, max_length, clip=True)
                #
                # # impute missing values
                # data = data.to_numpy(allow_missing=True)
                # for i in range(len(data)):
                #
                #     trace = data[i]
                #     mask = np.isnan(trace)
                #     trace = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), trace[~mask])
                #
                #     data[i] = trace

                for i in range(len(data)):

                    trace = np.array(data[i])

                    if min_length is not None and len(trace) < min_length:
                        # todo not elegant to just add values at the end
                        trace = np.pad(trace, pad_width=(min_length - len(trace), 0), mode=pad_mode)

                        data[i] = trace

                    elif max_length is not None and len(trace) > max_length:
                        data[i] = trace[:max_length]

                data = np.array(data)

            else:

                data = np.array(data)

                if min_length is not None and data.shape[1] < min_length:
                    data = np.pad(data, pad_width=min_length - data.shape[1], mode=pad_mode)
                    data = data[:, :min_length]

                elif max_length is not None and data.shape[1] > max_length:
                    data = data[:, :max_length]

        # update events dataframe
        print(data.shape)
        events.trace = data.tolist()
        events.dz = events.trace.apply(lambda x: len(x))
        logging.warning("z0 and z1 values do not correspond to the adjusted event boundaries")

        if inplace:
            self.events = events

        return events

    @wrapper_local_cache
    def get_frequency(self, grouping_column, cluster_column, normalization_instructions=None):

        events = self.events

        grouped = events[[grouping_column, cluster_column, "dz"]].groupby([grouping_column, cluster_column]).count()
        grouped.reset_index(inplace=True)

        pivot = grouped.pivot(index=cluster_column, columns=grouping_column, values="dz")
        pivot = pivot.fillna(0)

        if normalization_instructions is not None:
            norm = Normalization(pivot.values, inplace=True)
            norm_arr = norm.run(normalization_instructions)
            pivot = pd.DataFrame(norm_arr, index=pivot.index, columns=pivot.columns)

        return pivot

    def normalize(self, normalize_instructions, inplace=True):

        traces = self.events.trace

        norm = Normalization(traces)
        norm_traces = norm.run(normalize_instructions)

        # update events
        if inplace:
            self.events.trace = norm_traces.tolist()
        else:
            return norm_traces

    def create_lookup_table(self, labels, default_cluster=-1):

        cluster_lookup_table = defaultdict(lambda: default_cluster)
        cluster_lookup_table.update({k: label for k, label in list(zip(self.events.index.tolist(), labels.tolist()))})

        return cluster_lookup_table


class Video:

    def __init__(self, data, z_slice=None, h5_loc=None, lazy=False, name=None):

        if isinstance(data, (Path, str)):

            io = IO()
            self.data = io.load(data, h5_loc=h5_loc, lazy=lazy, z_slice=z_slice)

        elif isinstance(data, (np.ndarray, da.Array)):
            z0, z1 = z_slice
            self.data = data[z0:z1, :, :]

        self.z_slice = z_slice
        self.Z, self.X, self.Y = self.data.shape
        self.name = name

    def __hash__(self):

        if isinstance(self.data, da.Array):
            logging.warning(f"da.Array.name: {self.data.name}")
            return hash(self.data.name)

        elif isinstance(self.data, np.ndarray):
            return xxhash.xxh128_intdigest(self.data.data)

        else:
            raise ValueError(f"please provide data as either np.ndarray or dask.array")

    def get_data(self, in_memory=False):

        if in_memory and isinstance(self.data, da.Array):
            return self.data.compute()

        else:
            return self.data

    @lru_cache(maxsize=None)
    def get_image_project(self, agg_func=np.mean, window=None, window_agg=np.sum, axis=0,
                          show_progress=True):

        img = self.data

        # calculate projection
        if window is None:
            proj = agg_func(img, axis=axis)

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

    def show(self, viewer=None, colormap=None,
             show_trace=False, window=160, indices=None, viewer1d=None, xlabel="frames", ylabel="Intensity", reset_y=False):

        if viewer is None:
            viewer = napari.view_image(self.data, name=self.name, colormap=colormap)

        else:
            viewer.add_image(self.data, name=self.name, colormap=colormap)

        if show_trace:

            # get trace
            Y = self.get_image_project(agg_func=np.mean, axis=(1, 2))
            X = range(len(Y)) if indices is None else indices

            # create 1D viewer
            # v1d = None
            # if attach:
            #
            #     for widget in viewer.window._dock_widgets.values():
            #         potential_widget = widget.widget()
            #         print(potential_widget)
            #         print(potential_widget.children())
            #         if isinstance(potential_widget, napari_plot._qt.qt_viewer.QtViewer):
            #             v1d = potential_widget
            #             break

            if viewer1d is None:
                v1d = napari_plot.ViewerModel1D()
                qt_viewer = QtViewer(v1d)
            else:
                v1d = viewer1d

            v1d.axis.y_label = ylabel
            v1d.axis.x_label = xlabel
            v1d.text_overlay.visible = True
            v1d.text_overlay.position = "top_right"

            v1d.set_y_view(np.min(Y) * 0.9, np.max(Y) * 1.1)

            # create attachable qtviewer
            # qt_viewer = QtViewer(v1d)
            line = v1d.add_line(np.c_[X, Y], name=self.name, color=colormap)

            def update_line(event: Event):
                Z, _, _ = event.value
                z0, z1 = Z-window, Z

                if z0 < 0:
                    z0 = 0

                y_ = Y[z0:z1]
                x_ = X[z0:z1]
                # y1 = np.pad(y1, (-z0, 0), 'constant', constant_values=0)

                line.data = np.c_[x_, y_]

                v1d.reset_x_view()

                if reset_y:
                    v1d.reset_y_view()

            viewer.dims.events.current_step.connect(update_line)

            if viewer1d is None:
                viewer.window.add_dock_widget(qt_viewer, area="bottom", name=self.name)

            return viewer, v1d

        return viewer


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
