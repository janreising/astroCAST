import platform
from pathlib import Path

import numpy as np
import pytest

from astrocast.analysis import Events
from astrocast.detection import Detector
from astrocast.helper import EventSim, SampleInput
from astrocast.preparation import IO


class TestDetector:

    @pytest.mark.parametrize("extension", [".h5"])
    @pytest.mark.parametrize("debug", [True, False])
    def test_real_data(self, tmpdir, extension, debug):

        si = SampleInput()
        input_ = si.get_test_data(extension=extension)

        path = Path(tmpdir.strpath).joinpath(f"{np.random.randint(10000)}_tempData")
        det = Detector(input_, output=path)
        det.run(loc="dff/ch0", lazy=False, debug=debug, z_slice=(0, 25))

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

        del input_
        del det

    @pytest.mark.skipif(platform.system() in ["Windows", "win32"],
                        reason="Deletion of temporary directory is unsuccessful on windows")
    @pytest.mark.parametrize("parallel", [True, False])
    def test_sim_data(self, tmpdir, parallel):

        sim_dir = Path(tmpdir.strpath).joinpath("sim/")
        sim_dir.mkdir()

        path = sim_dir.joinpath(f"{np.random.randint(1000)}_{np.random.randint(1000)}_sim.h5")

        loc = "dff/ch0"

        # simulate events
        sim = EventSim()
        video, num_events = sim.simulate(
            shape=(100, 250, 250), skip_n=5, gap_space=10, gap_time=5,
            event_intensity=100, background_noise=1
        )
        del sim

        # save output
        io = IO()
        io.save(path=path, data={loc: video})
        del io
        del video

        # detect artificial events
        det = Detector(path.as_posix())
        dir_ = det.run(loc=loc, lazy=True, debug=False, parallel=parallel)

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

        del det
        del events
        del dir_

    def test_on_disk_sharing(self, tmpdir, extension=".h5", debug=False):

        si = SampleInput()
        input_ = si.get_test_data(extension=extension)

        path = Path(tmpdir.strpath).joinpath(f"{np.random.randint(10000)}_tempData")

        det = Detector(input_, output=path)
        det.run(loc="dff/ch0", lazy=False, debug=debug, use_on_disk_sharing=True)

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

        assert len(list(path.glob("*.mmap"))) < 1, f"mmap files were not removed: {list(path.glob('*.mmap'))}"

        del si
        del det
