import tempfile

import numpy as np
import pandas as pd
import pytest

from astroCAST.analysis import *
from astroCAST.detection import Detector
from astroCAST.helper import EventSim


def create_sim_data(dir, name="sim.h5", h5_loc="dff/ch0", save_active_pixels=False, shape=(50, 100, 100)):

    tmpdir = Path(dir)
    assert tmpdir.is_dir()

    path = tmpdir.joinpath(name)

    sim = EventSim()
    video, num_events = sim.simulate(shape=shape)
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
            logging.warning(event_dir)

            # Call the function and assert the expected result
            events = Events(event_dir)

            if input_type == "dir":
                time_map, event_start_frame, event_stop_frame = events.get_time_map(event_dir=event_dir)
                assert time_map.shape[1] == shape[0], f"wrong second dimension: {time_map.shape[1]} instead of {shape[0]}"

            else:
                event_map = np.array([[0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
                time_map, event_start_frame, event_stop_frame = events.get_time_map(event_map=event_map)
                assert time_map.shape == (3, 4), f"wrong dimensions: {time_map.shape} vs. (3, 4)"
                assert np.allclose(event_start_frame, np.array((3, 0, 2)))
                assert np.allclose(event_stop_frame, np.array((3, 1, 3)))

