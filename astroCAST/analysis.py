import logging
from pathlib import Path
from astroCAST.helper import get_data_dimensions
from astroCAST.preparation import IO


class Events:

    def __init__(self, event_dir, data_path=None, meta_path=None):

        # todo multi file

        if not Path(event_dir).is_dir():
            raise FileNotFoundError(f"cannot find provided event directory: {event_dir}")


        if meta_path is not None:
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

    @staticmethod
    def get_meta_info(meta_path):

        if not Path(meta_path).is_file():
                raise FileNotFoundError(f"cannot find provided meta file: {meta_path}")

        raise NotImplementedError("implement meta loading function")

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

        if Path(event_dir).joinpath("event_map.tdb").is_file():

            path = Path(event_dir).joinpath("event_map.tdb")
            shape, dtype = get_data_dimensions(path, return_dtype=True)

        elif Path(event_dir).joinpath("event_map.tiff").is_file():

            path = Path(event_dir).joinpath("event_map.tiff")
            shape, dtype = get_data_dimensions(path, return_dtype=True)

        else:
            logging.warning(f"cannot find 'event_map.tdb' or 'event_map.tiff'. Might lead to problems in downstream processing.")
            shape, dtype = (None, None, None), None
            event_map = None

            return event_map, shape, dtype

        event_map = IO.load(path, lazy=not in_memory)

        return event_map, shape, dtype

    @staticmethod
    def get_time_map():
        raise NotImplementedError("implement get_time_map() function")

        if time_map_path.is_file():
            self.time_map = np.load(time_map_path.as_posix(), allow_pickle=True)[()]
            self.e_start = np.argmax(self.time_map, axis=0)  # 1D array of frame number when event occurs
            self.e_stop = self.time_map.shape[0]-np.argmax(self.time_map[::-1, :], axis=0)  # 1D array last event frame

    def load_events(self, z_slice=None, index_prefix=None):

        events_path = list(self.dir_.glob("events.npy"))[0]
        raw_events = np.load(events_path, allow_pickle=True)[()]
        self.vprint("#num events: {}".format(len(raw_events.keys())), urgency=1)

        events = pd.DataFrame(raw_events).transpose()
        events.sort_index(inplace=True)

        # legacy version check
        if "dz" not in events.columns:
            warnings.warn("Events dataframe is missing crucial columns. You are probably trying to load a legacy file. We advise to rerun your analysis with the newest version.")
            events[["dz", "dx", "dy"]] = pd.DataFrame(events["dim"].tolist())
            # events[["z0", "z1", "x0", "x1", "y0", "y1"]] = pd.DataFrame(events["bbox"].tolist())
            # print(events.columns)

        # calculate extra characteristics
        events["area_norm"] = events.area / events.dz
        # events["pix_num_norm"] = events.pix_num / events.dz
        # events["area_footprint"] = events.footprint.apply(sum)

        if "cx" not in events.columns:
            events["cx"] = events.x0 + events.dx * events["fp_centroid_local-0"]
            events["cy"] = events.y0 + events.dy * events["fp_centroid_local-1"]

        if index_prefix is not None:
            events.index = ["{}{}".format(index_prefix, i) for i in events.index]

        if z_slice is not None:
            z0, z1 = z_slice
            events = events[(events.z0 >= z0) & (events.z1 <= z1)]

        # save results
        self.events = events
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
    @local_cache
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
