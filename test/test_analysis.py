import tempfile
import time

import dask.array
import pytest

from astrocast.analysis import *
from astrocast.clustering import Distance
from astrocast.detection import Detector
from astrocast.helper import EventSim, DummyGenerator

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

import matplotlib.pyplot as plt

class Test_Events:

    def test_load_events_minimal(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)
            assert tmpdir.is_dir()

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(tmpdir.joinpath("sim.h5"))

            events = Events(event_dir)
            result = events.load_events(event_dir, custom_columns=None)

            assert isinstance(result, pd.DataFrame)

    def test_load_events_custom_columns(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"))

            custom_columns = ["area_norm", "cx", "cy", "area_footprint",
                              # "pix_num_norm",
                              {"add_one": lambda events: events.z0 + 1}]

            # Call the function and assert the expected result
            events = Events(event_dir)
            result = events.load_events(event_dir, custom_columns=custom_columns)
            assert isinstance(result, pd.DataFrame)

            for col in custom_columns:

                if isinstance(col, str):
                    assert col in result.columns
                elif isinstance(col, dict):
                    assert list(col.keys())[0] in result.columns
                else:
                    raise ValueError(f"please provide valid custom_colum instead of: {col}")

    def test_load_events_z_slice(self, z_slice=(10, 25)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"))

            # Call the function and assert the expected result
            events = Events(event_dir)
            result = events.load_events(event_dir, z_slice=z_slice)
            assert isinstance(result, pd.DataFrame)

            assert z_slice[0] <= result.z0.min(), f"slicing unsuccesful; {z_slice[0]} !<= {result.z0.min()} "
            assert z_slice[1] >= result.z1.max(), f"slicing unsuccesful; {z_slice[1]} !>= {result.z1.max()} "

    def test_load_events_prefix(self, prefix="prefix_"):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"))

            # Call the function and assert the expected result
            events = Events(event_dir)
            result = events.load_events(event_dir, index_prefix=prefix)
            assert isinstance(result, pd.DataFrame)

            for ind in result.index.tolist():
                assert ind.startswith(prefix)

    @pytest.mark.parametrize("input_type", ["dir", "event_map"])
    def test_get_time_map(self, input_type, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"))

            # Call the function and assert the expected result
            events = Events(event_dir)

            if input_type == "dir":
                time_map, event_start_frame, event_stop_frame = events.get_time_map(event_dir=event_dir)
                assert time_map.shape[0] == shape[0], f"wrong second dimension: {time_map.shape[0]} instead of {shape[0]}"

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
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"))

            # Call the function and assert the expected result
            events = Events(event_dir)

            # get event map
            event_map, shape, dtype = events.get_event_map(event_dir)

    def test_create_event_map(self, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape,
                                           blob_size_fraction=0.05, event_probability=0.2)

            # Call the function and assert the expected result
            events = Events(event_dir)

            # get event map
            event_map, shape, dtype = events.get_event_map(event_dir)

            df = events.load_events(event_dir)
            event_map_recreated = events.create_event_map(df, video_dim=shape)

            assert event_map.shape == event_map_recreated.shape
            assert np.allclose(event_map > 0, event_map_recreated > 0)
            # assert np.allclose(event_map, event_map_recreated) # fails because

    @pytest.mark.parametrize("param", [
        dict(memmap_path=False),
        dict(normalization_instructions={0: ["subtract", {"mode": "mean"}], 1: ["divide", {"mode": "std"}]},
             memmap_path=False),
        dict(normalization_instructions={0: ["subtract", {"mode": "mean"}], 1: ["divide", {"mode": "std"}]},
             memmap_path=True)
    ])
    def test_extension_full(self, param, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            if param["memmap_path"]:
                param["memmap_path"] = Path(tmpdir).joinpath("arr.mmap")
            else:
                param["memmap_path"] = None

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            df = events.load_events(event_dir)
            arr, shape, dtype = events.get_event_map(event_dir, lazy=False)

            traces, _, _ = events.get_extended_events(video=Video(arr), return_array=True, **param)

            assert traces.shape == (len(df), shape[0])

            logging.warning(f"trace: {traces}")
            logging.warning(f"arr: {arr}")

            data_unique_values = np.unique(arr.flatten().astype(int))
            trace_unique_values = np.unique(traces.flatten().astype(int))

            assert abs(len(data_unique_values) - len(trace_unique_values)) <= 2, f"data_unique: {data_unique_values}\n" \
                                                                                 f"trace_unique: {trace_unique_values}"

    @pytest.mark.parametrize("extend", [4, (3, 2), (-1, 2), (2, -1)])
    def test_extension_partial(self, extend, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)
            events = Events(event_dir)
            df = events.load_events(event_dir)
            arr, shape, dtype = events.get_event_map(event_dir, lazy=False)

            traces, _, _ = events.get_extended_events(video=Video(arr), extend=extend, return_array=True)

            assert traces.shape == (len(df), shape[0])

            # this works because we are feeding the event_map as dummy data
            # hence all the trace values will be a constant number
            data_unique_values = np.unique(arr.flatten().astype(int))
            trace_unique_values = np.unique(traces.flatten().astype(int))
            assert abs(len(data_unique_values) - len(trace_unique_values)) <= 1, f"data_unique: {data_unique_values}\n" \
                                                                                 f"trace_unique: {trace_unique_values}"

    def test_extension_cache(self, lazy=False, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)
            cache_path = tmpdir.joinpath("cache_dir/")

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

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

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            df = events.load_events(event_dir)
            arr, shape, dtype = events.get_event_map(event_dir, lazy=True)

            save_path=Path(tmpdir).joinpath("footprints.npy")
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

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            # test dictionary mapping
            frame_to_time_mapping={i:i*2 for i in range(1000)}
            events = Events(event_dir, frame_to_time_mapping=frame_to_time_mapping)
            assert "t0" in events.events.columns
            assert "t1" in events.events.columns

            # test function mapping
            frame_to_time_function=lambda x: x*2
            events = Events(event_dir, frame_to_time_function=frame_to_time_function)
            assert "t0" in events.events.columns
            assert "t1" in events.events.columns

    def test_load_data(self, shape=(50, 100, 100)):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            # test path
            events = Events(event_dir, data=event_dir.joinpath("event_map.tiff"))

            # test video
            events = Events(event_dir, data=Video(event_dir.joinpath("event_map.tiff")))

            # test array
            events = Events(event_dir, data=np.zeros((5, 5, 5)))

            # test object
            events = Events(event_dir, data=bool)

    def test_multi_file_support(self, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir_1:
            with tempfile.TemporaryDirectory() as tmpdir_2:

                sim = EventSim()
                event_dir_1 = sim.create_dataset(Path(tmpdir_1).joinpath("sim.h5"), shape=shape)
                event_dir_2 = sim.create_dataset(Path(tmpdir_2).joinpath("sim.h5"), shape=shape)

                event_1 = Events(event_dir_1)
                event_2 = Events(event_dir_2)
                total_num_events = len(event_1.events) + len(event_2.events)

                comb_events = Events([event_dir_1, event_dir_2])
                num_events = len(comb_events.events)
                assert total_num_events == num_events

    @pytest.mark.xfail
    def test_get_event_timing(self, shape=(50, 100, 100)):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            events.get_event_timing()

    @pytest.mark.xfail
    def test_set_timings(self, shape=(50, 100, 100)):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            events.set_timings()

    @pytest.mark.xfail
    def test_align(self, shape=(50, 100, 100)):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            events.align()

    @pytest.mark.parametrize("param", [
        dict(index=None), dict(smooth=5), dict(gradient=True)
    ])
    def test_get_average_event_trace(self, param, shape=(50, 100, 100)):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            sim = EventSim()
            event_dir = sim.create_dataset(Path(tmpdir).joinpath("sim.h5"), shape=shape)

            events = Events(event_dir)
            avg = events.get_average_event_trace(**param)

            assert np.squeeze(avg.shape) == np.squeeze(np.zeros(shape=(shape[0])).shape)

    def test_to_numpy(self):

        events = Events(event_dir=None)

        df = pd.DataFrame({
            "z0": [0, 2, 5],
            "z1": [3, 3, 9],
            "trace": [
                np.array([1, 1, 1]), np.array([2]), np.array([3, 3, 3, 3]),
            ]
        })

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
        freq = events.get_frequency(grouping_column="group", cluster_column="clusters",
                            normalization_instructions=instr)
        assert freq.max().max() == 0

class Test_Video:

    def basic_load(self, input_type, z_slice, lazy, proj_func, window, shape=(50, 25, 25)):

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
                h5_loc = "data"

                io = IO()
                io.save(path=path, data=data, h5_loc=h5_loc)

                vid = Video(data=path, h5_loc="data/ch0", lazy=lazy, z_slice=z_slice)

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
                            proj_org[:, x, y] = pd.Series(original_data[:, x, y]).rolling(window=window).apply(proj_func).values

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
