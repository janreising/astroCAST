import logging
from pathlib import Path

import dask.array
import deprecation
import numpy as np
import pandas as pd
from tqdm import tqdm

import astroCAST.detection
from astroCAST.helper import get_data_dimensions, notimplemented, wrapper_local_cache
from astroCAST.preparation import IO


class Events:

    def __init__(self, event_dir, data_path=None, meta_path=None, in_memory=False):

        # todo multi file

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
        # TODO @Ana your module has a function to do this. Maybe we should move that method here

        # add slicing
        # self.z_slice = z_slice
        # if z_slice is not None:
        #     self.num_frames = z_slice[1] - z_slice[0]

        # load events
        # self.load_events(z_slice=self.z_slice, index_prefix=index_prefix)

        # align time

    # @notimplemented("implement meta loading function")
    def get_meta_info(self, meta_path):

        if not Path(meta_path).is_file():
                raise FileNotFoundError(f"cannot find provided meta file: {meta_path}")

        self.info = json.load(file)

        if fix_errors:

            #     if "xii" not in info.keys() or info["xii"] is None:
            #
            #     info["xii"] = {
            #         "sampling_rate": 25,
            #         "units": "us",
            #         "channels": ["xii", "camera_timing"],
            #         "folder": "/A",
            #         "burst_prominence": 0.02,
            #         "burst_distance": 20000,
            #         "camera_prominence": 0.1
            #     }
            #
            #     self.vprint("cannot find 'xii' settings in json config. Assuming values: \n {}".format(info["xii"]), urgency=0)
            #
            #
            # if "alignment" not in info.keys() or info["alignment"] is None:
            #
            #     info["alignment"] = {
            #         "num_channels": len(info["preprocessing"]["channel"]),
            #         "z_skip": 1,
            #         "skip_first": 10,
            #         "skip_last": 11
            #     }
            #
            #     self.vprint("cannot find 'alignment' settings in json config. Assuming values: \n {}".format(info["alignment"]), urgency=0)
            #

            pass

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
            io.save(save_path, data=event_map)

        return event_map

    # @notimplemented("implement get_time_map()")
    @staticmethod
    def get_time_map(event_dir=None, event_map=None, chunk=100):
        """
        Creates a binary array representing the time map of events.

        Args:
            event_dir (Path): The directory containing the event data.
            event_map (ndarray): The event map data.
            chunk (int): The chunk size for processing events.

        Returns:
            Tuple: A tuple containing the time map, events' start frames, and events' end frames.
                time_map > binary array of size (num_events x num_frames) where 1 denotes an active event
                events_start_frame > 1D array (num_events x num_frames) of event start
                events_end_frame > 1D array (num_events x num_frames) of event end

        Raises:
            ValueError: If neither 'event_dir' nor 'event_map' is provided.

        """

        if event_dir is not None:

            if not event_dir.is_dir():
                raise FileNotFoundError(f"cannot find event_dir: {event_dir}")

            time_map_path = Path(event_dir).joinpath("time_map.npy")
            time_map = np.load(time_map_path.as_posix(), allow_pickle=True)[()]

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

class Footprints:

    def __init__(self):
        pass

    def get_footprints(self, events=None, dff=None, dtype=np.half, standardize=True):

        """ takes the footprint of each individual event and extends it over the whole z-range

        """

        if self.cache and self.footprints is not None:
            return self.footprints

        # load if possible
        if self.local_cache:

            cache_path = self.lc_path.joinpath("fp_traces.npy")

            if cache_path.is_file():
                self.vprint("loading footprints from: {}".format(cache_path), 2)
                footprints = np.load(cache_path.as_posix())

                if self.cache:
                    self.footprints = footprints

                return footprints

        n_events = len(self.events)
        n_frames = self.num_frames
        events = events if events is not None else self.events

        arr = np.zeros((n_events, n_frames), dtype=dtype)
        self.vprint("Proposed shape: ({}, {}) [{:.2f}GB]".format(n_events, n_frames, arr.itemsize*n_events*n_frames*1e-9), 2)

        # load dff
        if dff is None:
            dff = self.get_channel()

        # extract footprints
        c = 0
        for i, event in tqdm(events.iterrows(), total=len(events), desc="gathering footprints"):

            data = dff[:, event.x0:event.x1, event.y0:event.y1]

            footprint = np.invert(np.reshape(event["footprint"], (event.dx, event.dy)))
            mask = np.broadcast_to(footprint, data.shape)

            projection = np.ma.masked_array(data=data, mask=mask)
            p = np.nanmean(projection, axis=(1, 2))

            if standardize:
                p = (p-np.mean(p)) / np.std(p)

            arr[c, :] = p

            c += 1

        self.footprints = arr

        if self.local_cache:
            np.save(cache_path.as_posix(), arr)

        return arr

    # todo decide whether to move this to a dedicated Correlation Module
    def get_footprint_correlation(self, footprints=None, dtype=np.single, mmap=False):

        """ correlate the footprints of all events with each other

        """

        if self.cache and self.corr is not None:
            return self.corr

        if self.local_cache:

            cache_path = self.lc_path.joinpath("fp_corr.npy")
            if cache_path.is_file():
                self.vprint("loading footprint correlation from: {}".format(cache_path), 2)

                if not mmap:
                    corr = np.load(cache_path.as_posix())
                else:
                    corr = np.load(cache_path.as_posix(), mmap_mode="r")

                if self.cache:
                    self.corr = corr

                return corr

        if footprints is None:
            footprints = self.get_footprints()

        corr = np.corrcoef(footprints, dtype=dtype)
        corr = np.tril(corr)

        self.corr = corr

        if self.local_cache:
            np.save(cache_path.as_posix(), corr)

            if mmap:
                corr = np.load(cache_path.as_posix(), mmap_mode="r")

        return corr


class Video:

    def __init__(self):
        pass

    def get_channel(self, channel="dff"):

        if self.cache and (channel in self.channels.keys()):
            return self.channels[channel]

        # guess file locations
        if channel == "event_map":
            h5path = self.dir_.joinpath("event_map.h5")
            tiffpath = self.dir_.joinpath("event_map.tiff")

        else:

            h5path = self.dir_.parent.joinpath("{}.h5".format(self.name_))

            # example: ./22A1x2.dff.ast.tiff
            typ = self.dir_.suffixes[0].replace(".", "")
            tiffpath = self.dir_.parent.joinpath("{}.{}.{}.tiff".format(self.name_, channel, typ))

        # load files
        if tiffpath.is_file():
            if self.z_slice is None:
                img = tiff.imread(tiffpath.as_posix())
            else:
                img = tiff.imread(tiffpath.as_posix(), key=range(self.z_slice[0], self.z_slice[1]))

        elif h5path.is_file():
            with h5py.File(h5path.as_posix(), "r") as file:
                if self.z_slice is None:
                    img = file[f"{channel}/{typ}"][:]
                else:
                    img = file[f"{channel}/{typ}"][self.z_slice[0]:self.z_slice[1], :, :]
        else:
            # self.vprint("Couldn't find {} or {}".format(h5path, tiffpath), 0)
            raise FileNotFoundError("Couldn't find {} or {}".format(h5path, tiffpath), 0)

        if self.cache:
            self.channels[channel] = img

        return img

    @deprecation.deprecated(details="function label suggests different functonality. Use 'get_channel()' instead")
    def get_image(self):

        if self.img is None:
            self.img = da.from_tiledb(self.event_map_.as_posix())

        return self.img

    # @lru_cache(maxsize=None)
    @wrapper_local_cache
    def get_image_project(self, agg_func=np.mean, window=None, window_agg=np.sum, channel="dff", z_slice=None,
                          show_progress=True):

        img = self.get_channel(channel)

        # select Z axis region
        if (self.z_slice is not None) or (z_slice is not None):

            if (self.z_slice is not None) and (z_slice is not None):
                z_slice = (max(self.z_slice[0], z_slice[0]), min(self.z_slice[1], z_slice[1]))
            elif self.z_slice is not None:
                z_slice = self.z_slice

            img = img[z_slice[0]:z_slice[1]]

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
