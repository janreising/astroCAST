import os
import shutil
import tempfile

import dask.array
import numpy as np
import pandas as pd
import pytest
import tifffile
import dask.array as da

from astroCAST.analysis import *
from astroCAST.detection import Detector
from astroCAST.helper import EventSim, DummyGenerator


def create_sim_data(dir, name="sim.h5", h5_loc="dff/ch0", save_active_pixels=False, shape=(50, 100, 100), sim_param={}):

    tmpdir = Path(dir)
    assert tmpdir.is_dir()

    path = tmpdir.joinpath(name)

    sim = EventSim()
    video, num_events = sim.simulate(shape=shape, **sim_param)
    IO.save(path=path, prefix="", data={h5_loc:video})

    det = Detector(path.as_posix(),  output=None)
    det.run(dataset=h5_loc, use_dask=True, save_activepixels=save_active_pixels)

    return det.output_directory

class Test_Events:

    def test_load_events_minimal(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir))

            # Call the function and assert the expected result
            events = Events(event_dir)
            result = events.load_events(event_dir, custom_columns=None)
            assert isinstance(result, pd.DataFrame)
            # Add additional assertions as needed

    def test_load_events_custom_columns(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir))

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
            event_dir = create_sim_data(Path(tmpdir), shape=(50, 100, 100))

            # Call the function and assert the expected result
            events = Events(event_dir)
            result = events.load_events(event_dir, z_slice=z_slice)
            assert isinstance(result, pd.DataFrame)

            assert z_slice[0] <= result.z0.min(), f"slicing unsuccesful; {z_slice[0]} !<= {result.z0.min()} "
            assert z_slice[1] >= result.z1.max(), f"slicing unsuccesful; {z_slice[1]} !>= {result.z1.max()} "

    def test_load_events_prefix(self, prefix="prefix_"):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=(50, 100, 100))

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
            event_dir = create_sim_data(Path(tmpdir), shape=shape)

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
            event_dir = create_sim_data(Path(tmpdir), shape=(50, 100, 100))

            # Call the function and assert the expected result
            events = Events(event_dir)

            # get event map
            event_map, shape, dtype = events.get_event_map(event_dir)

    def test_create_event_map(self, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=shape,
                                        sim_param=dict(blob_size_fraction=0.01, event_probability=0.1))

            # Call the function and assert the expected result
            events = Events(event_dir)

            # get event map
            event_map, shape, dtype = events.get_event_map(event_dir)

            df = events.load_events(event_dir)
            event_map_recreated = events.create_event_map(df, video_dim=shape)

            assert event_map.shape == event_map_recreated.shape
            assert np.allclose(event_map > 0, event_map_recreated > 0)
            # assert np.allclose(event_map, event_map_recreated) # fails because

    @pytest.mark.parametrize("param", [dict(normalize=None, lazy=False), dict(normalize="mean"), dict(lazy=True)])
    def test_extension_full(self, param, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=shape)
            events = Events(event_dir)
            df = events.load_events(event_dir)
            video, shape, dtype = events.get_event_map(event_dir, in_memory=True)

            trace = events.get_extended_events(events=df, video=video, **param)

            assert trace.shape == (len(df), shape[0])
            assert len(np.unique(trace.astype(int))) == len(np.unique(video.astype(int)))

    @pytest.mark.parametrize("extend", [4, (3, 2), (-1, 2), (2, -1)])
    def test_extension_partial(self, extend, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=shape)
            events = Events(event_dir)
            df = events.load_events(event_dir)
            video, shape, dtype = events.get_event_map(event_dir, in_memory=True)

            trace = events.get_extended_events(events=df, video=video, extend=extend)

            assert trace.shape == (len(df), shape[0])
            assert len(np.unique(trace.astype(int))) == len(np.unique(video.astype(int)))

    def test_extension_save(self, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=shape)
            events = Events(event_dir)
            df = events.load_events(event_dir)
            video, shape, dtype = events.get_event_map(event_dir, in_memory=True)

            save_path=Path(tmpdir).joinpath("footprints.npy")
            trace = events.get_extended_events(events=df, video=video, save_path=save_path)

            io = IO()
            trace_loaded = io.load(save_path)

            assert np.allclose(trace, trace_loaded)

    def test_z_slice(self, z_slice=(10, 40), shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Create dummy data
            event_dir = create_sim_data(Path(tmpdir), shape=shape)
            events = Events(event_dir, z_slice=z_slice)

    def test_multi_file_support(self, shape=(50, 100, 100)):

        with tempfile.TemporaryDirectory() as tmpdir_1:
            with tempfile.TemporaryDirectory() as tmpdir_2:

                event_dir_1 = create_sim_data(Path(tmpdir_1), shape=shape)
                event_dir_2 = create_sim_data(Path(tmpdir_2), shape=shape)

                events = Events([event_dir_1, event_dir_2])

class Test_Correlation:

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("mmap", [True, False])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", "pandas"])
    def test_get_correlation_matrix(self, input_type, ragged, mmap):

        dg = DummyGenerator(num_rows=25, trace_length=12, ragged=ragged)

        if input_type == "numpy":
            data = dg.get_array()

        elif input_type == "dask":
            data = da.from_array(dg.get_array(), chunks="auto")

        elif input_type == "list":
            data = dg.get_list()

        elif input_type == "pandas":
            data = dg.get_dataframe()

        else:
            raise ValueError(f"unknown attribute: {input_type}")

        corr = Correlation()
        c = corr.get_correlation_matrix(events=data, mmap=mmap)

class Test_Video:

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("z_slice", [None, (10, 40), (10, -1)])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", ".h5", ".tdb", ".tiff"])
    @pytest.mark.parametrize("proj_func", [np.mean, np.min, None])
    # TODO test window
    def test_basic_loading(self, input_type, z_slice, lazy, proj_func, shape=(50, 25, 25)):

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
                io.save(path=path, data=data, prefix=h5_loc)

                vid = Video(data=path, h5_loc="data/ch0", lazy=lazy, z_slice=z_slice)

            elif input_type in [".tdb", ".tiff"]:

                path = tmp_path.with_suffix(input_type)

                io = IO()
                io.save(path=path, data=data)

                vid = Video(data=path, lazy=lazy, z_slice=z_slice)

            d = vid.get_data(in_memory=True)

            # test same result
            if z_slice is None:
                assert np.allclose(original_data, d)

            else:
                z0, z1 = z_slice
                assert np.allclose(original_data, d)

            # test project
            if proj_func is not None:
                proj_org = proj_func(original_data, axis=0)
                proj = vid.get_image_project(agg_func=proj_func)

                assert np.allclose(proj_org, proj)

