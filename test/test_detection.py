import tempfile

import pytest

from astrocast.analysis import Events
from astrocast.detection import *
from astrocast.helper import EventSim, SampleInput
from astrocast.preparation import IO


class Test_Detector:

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("debug", [True, False])
    def test_real_data(self, extension, debug):

        si = SampleInput()
        input_ = si.get_test_data(extension=extension)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            assert tmpdir.is_dir()

            det = Detector(input_, output=tmpdir.joinpath("tempData"))
            det.run(h5_loc="dff/ch0", lazy=False, debug=debug)

            dir_ = det.output_directory

            assert dir_.is_dir(), "Output folder does not exist"
            assert bool(det.meta), "metadata dictionary is empty"
            assert det.data.size != 0, "data object is empty"
            assert det.data.shape is not None, "data has no dimensions"

            for file_name in ["event_map.tdb", "event_map.tiff", "active_pixels.tiff", "time_map.npy", "events.npy",
                              "meta.json"]:
                is_file = dir_.joinpath(file_name)
                assert is_file, f"{file_name} file does not exist in output directory"

            if debug:
                assert dir_.joinpath("debug_smoothed_input.tiff").is_file()
                assert dir_.joinpath("debug_active_pixels.tiff").is_file()
                assert dir_.joinpath("debug_active_pixels_morphed.tiff").is_file()

    def test_sim_data(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            assert tmpdir.is_dir()

            path = tmpdir.joinpath("sim.h5")
            h5_loc = "dff/ch0"

            sim = EventSim()
            video, num_events = sim.simulate(
                shape=(50, 100, 100), skip_n=5, gap_space=5, gap_time=3, event_intensity=100, background_noise=1
            )
            io = IO()
            io.save(path=path, data={h5_loc: video})

            det = Detector(path.as_posix())
            dir_ = det.run(h5_loc=h5_loc, lazy=True, debug=False)

            assert dir_.is_dir(), "Output folder does not exist"
            assert bool(det.meta), "metadata dictionary is empty"
            assert det.data.size != 0, "data object is empty"
            assert det.data.shape is not None, "data has no dimensions"

            expected_files = ["event_map.tiff", "time_map.npy", "events.npy", "meta.json"]
            for file_name in expected_files:
                assert dir_.joinpath(file_name).exists(), f"cannot find {file_name}"

            # check event detection
            events = Events(dir_)
            assert np.allclose(len(events), num_events, rtol=0.15), f"Found {len(events)} instead of {num_events}."
