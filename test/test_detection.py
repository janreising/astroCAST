import tempfile

import pytest

from astroCAST.detection import *
from astroCAST.helper import EventSim
from astroCAST.preparation import IO


class Test_Detector:

    # TODO test custom threshold
    # TODO test custom output folder
    # TODO test dummy data
    @pytest.mark.parametrize("save_active_pixels", [True, False])
    def test_real_data(self, save_active_pixels, threshold=None, output=None):

        det = Detector("testdata/sample_0.h5",  output = output)
        det.run(dataset = "data/ch0", use_dask = True, save_activepixels = save_active_pixels)

        dir_ = det.output_directory

        assert dir_.is_dir(), "Output folder does not exist"
        assert bool(det.meta), "metadata dictionary is empty"
        assert det.data.size != 0, "data object is empty"
        assert det.data.shape is not None, "data has no dimensions"

        expected_files = ["event_map.tdb", "event_map.tiff", "active_pixels.tiff", "time_map.npy", "time_map.npy", "events.npy", "meta.json", ""]
        for file_name in expected_files:
            is_file = dir_.joinpath(file_name)

            if file_name == "active_pixels.tiff":
                assert is_file == save_active_pixels, "can't/can find active_pixels.tiff but should/shouldn't"
            else:
                assert is_file, f"{file_name} file does not exist in output directory"

    def test_sim_data(self):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            path = tmpdir.joinpath("sim.h5")
            h5_loc = "dff/ch0"
            save_active_pixels = False

            sim = EventSim()
            video, num_events = sim.simulate(shape=(50, 100, 100))
            IO.save(path=path, prefix="", data={h5_loc:video})

            det = Detector(path.as_posix(),  output=None)
            events = det.run(dataset=h5_loc, use_dask=True, save_activepixels=save_active_pixels)

            dir_ = det.output_directory

            assert dir_.is_dir(), "Output folder does not exist"
            assert bool(det.meta), "metadata dictionary is empty"
            assert det.data.size != 0, "data object is empty"
            assert det.data.shape is not None, "data has no dimensions"

            expected_files = ["event_map.tdb", "event_map.tiff", "time_map.npy", "events.npy", "meta.json"]
            for file_name in expected_files:
                assert dir_.joinpath(file_name).exists(), f"cannot find {file_name}"

            # optional
            if save_active_pixels:
                assert dir_.joinpath("active_pixels.tiff").is_file(), "can't find active_pixels.tiff but should"
            else:
                assert not dir_.joinpath("active_pixels.tiff").is_file(), "can find active_pixels.tiff but shouldn't"

            # check event detection
            assert np.allclose(len(events), num_events, rtol=0.1), f"Found {len(events)} instead of {num_events}."



