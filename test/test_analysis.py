import platform
import tempfile
import time

import dask.array
import matplotlib
import pytest

from astrocast.analysis import *
from astrocast.helper import EventSim

matplotlib.use('Agg')  # Use the Agg backend


class TestEvents:

    def setup_method(self):

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def cleanup_method(self):

        # necessary to give Windows time to release files
        if platform.system() == "win32":
            logging.warning(f"Assuming to be on windows. Waiting for files to be released!")
            time.sleep(20)

        self.tmpdir.cleanup()

    def test_load_events_minimal(self):

        path = self.tmp_path.joinpath("sim_ev_min.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        events = Events(event_dir)
        result = events._load_events(event_dir, custom_columns=None)

        assert isinstance(result, pd.DataFrame)

    def test_load_events_custom_columns(self):

        path = self.tmp_path.joinpath("sim_ev_cust_col.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        custom_columns = ["v_area_norm", "cx", "cy", "v_area_footprint",  # "pix_num_norm",
                          {"add_one": lambda events_: events_.z0 + 1}]

        # Call the function and assert the expected result
        events = Events(event_dir)
        result = events._load_events(event_dir, custom_columns=custom_columns)
        assert isinstance(result, pd.DataFrame)

        for col in custom_columns:

            if isinstance(col, str):
                assert col in result.columns
            elif isinstance(col, dict):
                assert list(col.keys())[0] in result.columns
            else:
                raise ValueError(f"please provide valid custom_colum instead of: {col}")

    def test_load_events_z_slice(self, z_slice=(10, 25)):

        path = self.tmp_path.joinpath("sim_ev_z_slice.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        # Call the function and assert the expected result
        events = Events(event_dir)
        result = events._load_events(event_dir, z_slice=z_slice)
        assert isinstance(result, pd.DataFrame)

        assert z_slice[0] <= result.z0.min(), f"slicing unsuccessful; {z_slice[0]} !<= {result.z0.min()} "
        assert z_slice[1] >= result.z1.max(), f"slicing unsuccessful; {z_slice[1]} !>= {result.z1.max()} "

    def test_load_events_prefix(self, prefix="prefix_"):

        path = self.tmp_path.joinpath("sim_ev_prefix.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        # Call the function and assert the expected result
        events = Events(event_dir)
        result = events._load_events(event_dir, index_prefix=prefix)
        assert isinstance(result, pd.DataFrame)

        for ind in result.index.tolist():
            assert ind.startswith(prefix)

    @pytest.mark.parametrize("input_type", ["dir", "event_map"])
    def test_get_time_map(self, input_type, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("sim_ev_time_map.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        # Call the function and assert the expected result
        events = Events(event_dir)

        if input_type == "dir":
            time_map, event_start_frame, event_stop_frame = events.get_time_map(event_dir=event_dir)
            assert time_map.shape[0] == shape[
                0], f"wrong second dimension: {time_map.shape[0]} instead of {shape[0]}"

        else:
            event_map = np.zeros(shape=(4, 3, 3), dtype=int)
            event_map[3:, 0, 0] = 1
            event_map[0:2, 1, 1] = 2
            event_map[2:3, 2, 2] = 3

            time_map, event_start_frame, event_stop_frame = events.get_time_map(event_map=event_map)
            assert time_map.shape == (4, 4), f"wrong dimensions: {time_map.shape} vs. (4, 3)"
            assert np.allclose(event_start_frame, np.array((0, 3, 0, 2)))
            assert np.allclose(event_stop_frame, np.array((4, 4, 2, 3)))

    def test_get_event_map(self):

        path = self.tmp_path.joinpath("sim_ev_map.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path)

        # Call the function and assert the expected result
        events = Events(event_dir)

        # get event map
        event_map, shape, dtype = events.get_event_map(event_dir)

    def test_create_event_map(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("sim_ev_create_ev_map.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path,
                                       shape=shape, event_intensity=100, background_noise=1, gap_time=3,
                                       gap_space=5, blob_size_fraction=0.05, event_probability=0.2
                                       )

        # Call the function and assert the expected result
        events = Events(event_dir)

        # get event map
        event_map, shape, dtype = events.get_event_map(event_dir)
        if isinstance(event_map, da.Array):
            event_map = event_map.compute()

        df = events._load_events(event_dir)
        event_map_recreated = events.create_event_map(df, video_dim=shape)

        assert event_map.shape == event_map_recreated.shape

        orig_masked = event_map > 0
        recr_masked = event_map_recreated > 0

        if not np.allclose(orig_masked, recr_masked):

            log_string = "orig vs. recr"
            for i in range(len(orig_masked)):

                so, sr = np.sum(orig_masked[i, :, :]), np.sum(recr_masked[i, :, :])

                if so != sr:
                    log_string += f"\n{i}: {so} vs. {sr}"

            logging.warning(log_string)

            raise ValueError(
                f"original and recreated event_map is not equivalend: {np.sum(orig_masked)} vs. {np.sum(recr_masked)}"
            )

    @pytest.mark.parametrize(
        "param", [dict(memmap_path=False), dict(
            normalization_instructions={0: ["subtract", {"mode": "mean"}], 1: ["divide", {"mode": "std"}]},
            memmap_path=False
        ), dict(
            normalization_instructions={0: ["subtract", {"mode": "mean"}], 1: ["divide", {"mode": "std"}]},
            memmap_path=True
        )]
    )
    @pytest.mark.parametrize("extend", [-1, 4, (3, 2), (-1, 2), (2, -1)])
    def test_extension(self, param, extend, shape=(50, 100, 100), num_events=3, event_length=3):

        if extend == -1:
            e0, e1 = 0, 0
        elif isinstance(extend, int):
            e0, e1 = extend, extend
        else:
            e0, e1 = extend

        Z, X, Y = shape

        arr = np.random.random(shape) * 1000
        arr = np.abs(arr.astype(int))

        events = {k: [] for k in
                  ["z0", "z1", "x0", "x1", "y0", "y1", "dz", "dx", "dy", "trace", "full_trace", "mask", "fp_mask"]}
        for i in range(num_events):

            z0 = np.random.randint(event_length + 1 + abs(e0), Z - event_length - 1 - abs(e1))
            x0 = np.random.randint(event_length + 1 + abs(e0), X - event_length - 1 - abs(e1))
            y0 = np.random.randint(event_length + 1 + abs(e0), Y - event_length - 1 - abs(e1))

            events["z0"].append(z0)
            events["x0"].append(x0)
            events["y0"].append(y0)

            z1, x1, y1 = z0 + event_length, x0 + event_length, y0 + event_length

            events["z1"].append(z1)
            events["x1"].append(x1)
            events["y1"].append(y1)

            events["dz"].append(event_length)
            events["dx"].append(event_length)
            events["dy"].append(event_length)

            trace = np.squeeze(np.mean(arr[z0:z1, x0:x1, y0:y1], axis=(1, 2)))
            logging.warning(f"trace.shape: {trace.shape}")

            if (extend == -1) or ((e0 == -1) and (e1 == -1)):
                full_trace = np.squeeze(np.mean(arr[:, x0:x1, y0:y1], axis=(1, 2)))

            elif (e0 == -1) and (e1 != -1):
                full_trace = np.squeeze(np.mean(arr[0:z1 + e1, x0:x1, y0:y1], axis=(1, 2)))

            elif (e0 != -1) and (e1 == -1):
                full_trace = np.squeeze(np.mean(arr[z0 - e0:, x0:x1, y0:y1], axis=(1, 2)))

            else:
                full_trace = np.squeeze(np.mean(arr[z0 - e0:z1 + e1, x0:x1, y0:y1], axis=(1, 2)))

            events["trace"].append(trace)
            events["full_trace"].append(full_trace)

            events["mask"].append(np.ones(shape=(event_length, event_length, event_length), dtype=bool).flatten())
            events["fp_mask"].append(np.ones(shape=(event_length, event_length), dtype=bool).flatten())

        # create Events instance
        ev = Events(None)
        ev.events = pd.DataFrame(events)
        ev.num_frames, ev.X, ev.Y = shape

        # extend events
        path = self.tmp_path.joinpath("ext_arr.mmap")
        ext_events = ev.get_extended_events(
            video=Video(arr), return_array=False, extend=extend,
            memmap_path=path if param["memmap_path"] else None
        )

        # check result
        for i in range(len(ext_events)):
            t = np.squeeze(np.array(ext_events.iloc[i].trace))
            f = np.squeeze(np.array(ev.events.iloc[i].full_trace))

            assert t.shape == f.shape
            assert np.allclose(t, f)

    def test_extension_cache(self, lazy=False, shape=(50, 100, 100)):

        sim_path = self.tmp_path.joinpath("ev_ext_cache.h5")
        cache_path = self.tmp_path.joinpath("ev_ext_cache_dir/")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(sim_path, shape=shape)

        # load events - 1
        events_1 = Events(event_dir, lazy=lazy, cache_path=cache_path)
        video = Video(events_1.event_map)

        t0 = time.time()
        trace = events_1.get_extended_events(video=video, extend=(2, 2), in_place=True)
        d1 = time.time() - t0

        # load events - 2
        events_2 = Events(event_dir, lazy=lazy, cache_path=cache_path)
        video = Video(events_2.event_map)

        t0 = time.time()
        trace_2 = events_2.get_extended_events(video=video, extend=(2, 2))
        events_2.events = trace_2
        d2 = time.time() - t0

        assert d2 < d1, f"caching is taking too long: {d2:.2f}s >= {d1:.2f}s"
        assert hash(events_1) == hash(events_2)

    def test_extension_save(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("ext_save_sim_2.h5")
        save_path = self.tmp_path.joinpath("ext_save_footprints.npy")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path,
                                       shape=shape, event_intensity=100, background_noise=1, gap_space=5,
                                       gap_time=2
                                       )

        events = Events(event_dir)
        df = events._load_events(event_dir)
        arr, shape, dtype = events.get_event_map(event_dir, lazy=True)

        arr = arr.astype(np.int32)
        trace, _, _ = events.get_extended_events(video=Video(arr), save_path=save_path, return_array=True)

        io = IO()
        trace_loaded = io.load(save_path)

        assert np.allclose(trace, trace_loaded)

    def test_z_slice(self, z_slice=(10, 40), shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)
            events = Events(event_dir, z_slice=z_slice)

            # TODO there should be some kind of assert here, no?

    def test_frame_to_time_conversion(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("frame_to_time_conv.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        # test dictionary mapping
        frame_to_time_mapping = {i: i * 2 for i in range(1000)}
        events = Events(event_dir, frame_to_time_mapping=frame_to_time_mapping)
        assert "t0" in events.events.columns
        assert "t1" in events.events.columns

        def frame_to_time_function(x):
            return x * 2

        events = Events(event_dir, frame_to_time_function=frame_to_time_function)
        assert "t0" in events.events.columns
        assert "t1" in events.events.columns

    def test_load_data(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("test_load_data_sim.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        # test path
        events = Events(event_dir, data=event_dir.joinpath("event_map.tiff"))

        # test video
        events = Events(event_dir, data=Video(event_dir.joinpath("event_map.tiff")))

        # test array
        events = Events(event_dir, data=np.zeros((5, 5, 5)))

        # test object
        events = Events(event_dir, data=bool)

    def test_multi_file_support(self, shape=(50, 100, 100)):

        path_1 = self.tmp_path.joinpath("multi_file_support_1.h5")
        path_2 = self.tmp_path.joinpath("multi_file_support_2.h5")

        sim = EventSim()
        event_dir_1 = sim.create_dataset(path_1, shape=shape)
        event_dir_2 = sim.create_dataset(path_2, shape=shape)

        event_1 = Events(event_dir_1)
        event_2 = Events(event_dir_2)
        total_num_events = len(event_1.events) + len(event_2.events)

        comb_events = Events([event_dir_1, event_dir_2])
        num_events = len(comb_events.events)
        assert total_num_events == num_events

    @pytest.mark.skip(reason="legacy")
    def test_get_event_timing(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("test_get_event_timing_sim.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        events = Events(event_dir)
        events.get_event_timing()

    @pytest.mark.skip(reason="legacy")
    def test_set_timings(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("test_set_timings_sim.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        events = Events(event_dir)
        events.set_timings()

    @pytest.mark.skip(reason="Not implemented")
    def test_align(self, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("test_align.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        events = Events(event_dir)
        events.align()

    @pytest.mark.parametrize(
        "param", [dict(index=None), dict(smooth=5), dict(gradient=True)]
    )
    def test_get_average_event_trace(self, param, shape=(50, 100, 100)):

        path = self.tmp_path.joinpath("test_get_average_event_trace.h5")

        # Create dummy data
        sim = EventSim()
        event_dir = sim.create_dataset(path, shape=shape)

        events = Events(event_dir)
        avg = events.get_average_event_trace(**param)

        assert np.squeeze(avg.shape) == np.squeeze(np.zeros(shape=(shape[0])).shape)

    def test_to_numpy(self):

        events = Events(event_dir=None)
        events.num_frames = 10

        df = pd.DataFrame(
            {"z0": [0, 2, 5], "z1": [3, 3, 9], "trace": [np.array([1, 1, 1]), np.array([2]), np.array([3, 3, 3, 3]), ]}
        )

        arr_expected = np.zeros((3, 10))
        arr_expected[0, 0:3] = 1
        arr_expected[0, 2:3] = 1
        arr_expected[0, 5:10] = 1

        arr_out = events.to_numpy(events=df)

        np.allclose(arr_expected, arr_out)

    def test_frequency(self, n_groups=3, n_clusters=10):

        dg = helper.DummyGenerator(n_groups=n_groups, n_clusters=n_clusters, num_rows=200, trace_length=2)
        events = dg.get_events()

        freq = events.get_frequency(grouping_column="group", cluster_column="clusters", normalization_instructions=None)
        assert len(freq) == n_clusters
        assert len(freq.columns) == n_groups

        instr = {0: ["subtract", {"mode": "max"}]}
        freq = events.get_frequency(
            grouping_column="group", cluster_column="clusters", normalization_instructions=instr
        )
        assert freq.max().max() == 0


class TestVideo:

    @staticmethod
    def basic_load(input_type, z_slice, lazy, proj_func, window, shape=(50, 25, 25)):

        data = np.random.random(size=shape)

        if z_slice is not None:
            z0, z1 = z_slice
            original_data = data[z0:z1, :, :]
        else:
            original_data = data.copy()

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            tmp_path = Path(tmpdir).joinpath("out")

            if input_type == "numpy":
                vid = Video(data=data, lazy=lazy, z_slice=z_slice)

            elif input_type == "dask":
                data = dask.array.from_array(data, chunks="auto")
                vid = Video(data=data, lazy=lazy, z_slice=z_slice)

            elif input_type == ".h5":

                path = tmp_path.with_suffix(input_type)
                loc = "data/ch0"

                io = IO()
                io.save(path=path, data=data, loc=loc)

                vid = Video(data=path, loc="data/ch0", lazy=lazy, z_slice=z_slice)

            elif input_type in [".tdb", ".tiff"]:

                path = tmp_path.with_suffix(input_type)

                io = IO()
                io.save(path=path, data=data)

                vid = Video(data=path, lazy=lazy, z_slice=z_slice)

            d = vid.get_data(in_memory=True)

            # test same result
            assert original_data.shape == d.shape, f"shape unequal: orig>{original_data.shape} vs load>{d.shape}"
            assert np.allclose(original_data, d)

            # test project
            if proj_func is not None:

                if window is None:
                    proj_org = proj_func(original_data, axis=0)

                else:
                    proj_org = np.zeros(original_data.shape)
                    for x in range(original_data.shape[1]):
                        for y in range(original_data.shape[2]):
                            proj_org[:, x, y] = pd.Series(original_data[:, x, y]).rolling(window=window).apply(
                                proj_func
                            ).values

                proj = vid.get_image_project(agg_func=proj_func, window=window)
                assert proj_org.shape == proj.shape, f"shape unequal: orig>{proj_org.shape} vs load>{proj.shape}"
                assert np.allclose(proj_org, proj)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("z_slice", [None, (10, 40), (10, -1)])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", ".h5", ".tdb", ".tiff"])
    def test_basic_loading(self, input_type, z_slice, lazy, proj_func=None, window=None):
        self.basic_load(input_type, z_slice, lazy, proj_func, window=window)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("z_slice", [None, (10, 40), (10, -1)])
    @pytest.mark.parametrize("input_type", ["numpy", "dask"])
    @pytest.mark.parametrize("proj_func", [np.mean, np.min, None])
    def test_proj(self, input_type, z_slice, lazy, proj_func, window=None):
        self.basic_load(input_type, z_slice, lazy, proj_func, window=window)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("z_slice", [None, (10, 40), (10, -1)])
    @pytest.mark.parametrize("input_type", ["numpy", "dask"])
    @pytest.mark.parametrize("proj_func", [np.mean, None])
    @pytest.mark.parametrize("window", [None, 3])
    def test_windowed_loading(self, input_type, z_slice, lazy, proj_func, window):

        try:
            self.basic_load(input_type, z_slice, lazy, proj_func, window=window)

        except AssertionError:
            # TODO don't really know how to test for windowing here!?
            logging.warning("testing currently insufficient to check whether or not the result is correct.")
