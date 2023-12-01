import multiprocessing
import platform
import tempfile
import time

import pytest
import requests

from astrocast.app_analysis import *
from astrocast.app_preparation import *
from astrocast.detection import Detector
from astrocast.helper import EventSim


@pytest.mark.skipif(platform.system() in ['Darwin', 'Windows'],
                    reason="Testing the app utilizes multiprocessing which does not properly work on MacOS.")
class TestAppPreparation:

    def setup_method(self):
        # create dummy
        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(
            shape=(50, 100, 100), skip_n=5, event_intensity=100, background_noise=1
        )
        io = IO()
        io.save(path=path, data=video, loc=loc)

        # create explorer
        exp = Explorer(input_path=path.as_posix(), loc=loc)

        # Define a process for the Shiny app
        self.app_process = multiprocessing.Process(target=exp.run, kwargs=dict(port=8090))
        self.app_process.start()

        # Give the server some time to start properly
        time.sleep(3)

    def teardown_method(self):
        # Terminate the app process
        self.app_process.terminate()
        # Wait for the process to shut down
        self.app_process.join()

    def test_server_starts(self):
        # Make a request to the running Shiny app
        response = requests.get("http://127.0.0.1:8090")
        assert response.status_code == 200


@pytest.mark.skipif(platform.system() in ['Darwin', 'Windows'],
                    reason="Testing the app utilizes multiprocessing which does not properly work on MacOS.")
class TestAppAnalysis:

    def setup_method(self):
        # create dummy
        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(
            shape=(50, 100, 100), skip_n=5, event_intensity=100, background_noise=1
        )
        io = IO()
        io.save(path=path, data=video, loc=loc)

        det = Detector(path.as_posix(), output=None)
        det.run(loc=loc, lazy=True, debug=False)

        dir_ = det.output_directory

        # create explorer
        ana = Analysis(input_path=dir_.as_posix(), video_path=path.as_posix(), loc=loc)

        # Define a process for the Shiny app
        self.app_process = multiprocessing.Process(target=ana.run, kwargs=dict(port=8091))
        self.app_process.start()

        # Give the server some time to start properly
        time.sleep(3)

    def teardown_method(self):
        # Terminate the app process
        self.app_process.terminate()
        # Wait for the process to shut down
        self.app_process.join()

    def test_server_starts(self):
        # Make a request to the running Shiny app
        response = requests.get("http://127.0.0.1:8091")
        assert response.status_code == 200
