import logging
from pathlib import Path

import dask.array
import deprecation
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

import astroCAST.detection
from astroCAST.helper import get_data_dimensions, notimplemented, wrapper_local_cache
from astroCAST.preparation import IO


class Events:

    def __init__(self, event_dir, data_path=None, meta_path=None, in_memory=False,
                 z_slice=None, index_prefix=None, custom_columns=["area_norm", "cx", "cy"]):

        if isinstance(event_dir, list):
            raise NotImplementedError("multi file support not implemented yet.")

        if not Path(event_dir).is_dir():
            raise FileNotFoundError(f"cannot find provided event directory: {event_dir}")

        if meta_path is not None:
            logging.debug("I should not be called at all")
            self.get_meta_info(meta_path)

        # get data
        if data_path is not None:
            self.get_data() # todo slicing

        # load event map
        event_map, event_map_shape, event_map_dtype = self.get_event_map(event_dir, in_memory=in_memory) # todo slicing

        # create time map
        time_map, events_start_frame, events_end_frame = self.get_time_map(event_dir)

        # z slicing
        if z_slice is not None:
            self.z_slice = z_slice
            self.num_frames = z_slice[1] - z_slice[0]

            raise NotImplementedError("z_slice not implemented for Events")

        # load events
        events = self.load_events(event_dir, z_slice=z_slice, index_prefix=index_prefix, custom_columns=custom_columns)

        # align time
        # TODO align time

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

            if not isinstance(event_map, (np.ndarray, dask.array.Array)):
                raise ValueError(f"please provide 'event_map' as np.ndarray or dask.Array")

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
                       normalize=None, lazy=False, show_progress=True,
                       save_path=None, save_param={}):

        """ takes the footprint of each individual event and extends it over the whole z-range

        """

        if normalize is not None:
            # TODO
            raise NotImplementedError("implement normalize flag")

        if lazy:
            # TODO
            raise NotImplementedError("implement lazy flag")

        if extend != -1:
            # TODO
            raise NotImplementedError("implement extend flag")

        n_events = len(events)
        n_frames, X, Y = video.shape

        arr = np.zeros((n_events, n_frames), dtype=dtype)

        arr_size = arr.itemsize*n_events*n_frames
        ram_size = psutil.virtual_memory().total
        if arr_size > 0.9 * ram_size:
            logging.warning(f"array size ({n_events}, {n_frames}) is larger than 90% RAM size ({arr_size*1e-9:.2f}GB, {arr_size/ram_size*100}%). Consider using smaller dtype or 'lazy=True'")

        # extract footprints
        c = 0
        iterator = tqdm(events.iterrows(), total=len(events), desc="gathering footprints") if show_progress else events.iterrows()
        for i, event in iterator:

            event_trace = video[:, event.x0:event.x1, event.y0:event.y1]

            footprint = np.invert(np.reshape(event["footprint"], (event.dx, event.dy)))
            mask = np.broadcast_to(footprint, event_trace.shape)

            projection = np.ma.masked_array(data=event_trace, mask=mask)
            p = np.nanmean(projection, axis=(1, 2))

            # if standardize:
            #     p = (p-np.mean(p)) / np.std(p)

            arr[c, :] = p

            c += 1

        if save_path is not None:
            io = IO()
            io.save(path=save_path, data=arr, **save_param)

        return arr


class Correlation:

    # @wrapper_local_cache
    @staticmethod
    def get_correlation_matrix(events, dtype=np.single, mmap=False):

        if mmap:
            raise NotImplementedError("mmap flag currently not implemented")

        if not isinstance(events, (np.ndarray, dask.array.Array, pd.DataFrame)):
            raise ValueError("please provide events as one of (np.ndarray, dask.array.Array, pd.DataFrame)")

        if isinstance(events, pd.DataFrame):

            if "trace" not in events.columns:
                raise ValueError("events dataframe expected to have trace column")

            events = events.trace.values

        corr = np.corrcoef(events, dtype=dtype)
        corr = np.tril(corr)

        return corr

class Video:

    def __init__(self, data, z_slice=None, h5_loc=None, lazy=False):

        if isinstance(data, (np.ndarray, dask.array.Array)):
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

        if in_memory and isinstance(self.data, dask.array.Array):
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
