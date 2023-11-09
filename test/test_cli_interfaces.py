import tempfile

import pytest
import tifffile
from click.testing import CliRunner
import h5py as h5

from astrocast.analysis import Events
from astrocast.cli_interfaces import *
from astrocast.detection import Detector
from astrocast.helper import EventSim
from astrocast.preparation import IO


class Test_ConvertInput:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()

        temp_file = Path(temp_dir.name).joinpath("temp.h5")
        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=np.random.randint(0, 100, size=(10, 25, 25)))

        temp_tiffs = Path(temp_dir.name).joinpath("imgs/")
        temp_tiffs.mkdir()
        for i in range(216):
            tifffile.imwrite(temp_tiffs.joinpath(f"img_{i}.tiff").as_posix(), data=np.random.randint(0, 100, size=(1, 25, 25)))

        self.runner = CliRunner()

        self.temp_dir = temp_dir
        self.temp_file = temp_file
        self.temp_tiffs = temp_tiffs

    def teardown_method(self):
        self.temp_dir.cleanup()

    @pytest.mark.parametrize("num_channels", [1, 2, 3])
    @pytest.mark.parametrize("name_prefix", [None, "channel_"])
    def test_tiffs(self, num_channels, name_prefix):

        out_file = self.temp_tiffs.parent.joinpath(f"out_{name_prefix}{num_channels}.h5")

        if name_prefix is None:
            channel_names = [f"ch{i}" for i in range(num_channels)]
        else:
            channel_names = [f"{name_prefix}{i}" for i in range(num_channels)]

        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", str(num_channels)]
        if name_prefix is not None:
            args.append("--channel-names")
            args.append(",".join(channel_names))

        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:

            for ch_name in channel_names:
                assert ch_name in f['data'], f"channel name: {ch_name} not found in output file; {list(f['data'].keys())}"

            lengths = [len(f[f'data/{ch_name}']) for ch_name in channel_names]
            assert len(np.unique(lengths)) == 1, f"lengths: {lengths}"

    def test_z_slice(self):

        out_file = self.temp_tiffs.parent.joinpath(f"out_z.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "1",
                "--z-slice", "0", "50"
                ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].shape == (50, 25, 25)

    def test_lazy(self):
        out_file = self.temp_tiffs.parent.joinpath(f"out_l.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "1",
                "--lazy"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

    def test_subtract_background(self):

        out_file_ref = self.temp_tiffs.parent.joinpath(f"out_ref.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file_ref.as_posix(),
                "--num-channels", "2"
                ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file_ref.exists()

        out_file = self.temp_tiffs.parent.joinpath(f"out_sb.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "2",
                "--subtract-background", "ch1"
                ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f_out:
            with h5.File(out_file_ref.as_posix(), "r") as f_ref:

                assert "data/ch0" in f_ref
                assert "data/ch1" in f_ref

                assert "data/ch0" in f_out
                assert "data/ch1" not in f_out

                ch0_ref = f_ref["data/ch0"][:]
                ch1_ref = f_ref["data/ch1"][:]

                ch1_mean = np.mean(ch1_ref, axis=0)
                ref = ch0_ref - ch1_mean

                np.allclose(ref, f_out["data/ch0"][:], rtol=1e-5)

    def test_in_memory(self):
        out_file = self.temp_tiffs.parent.joinpath(f"out_im.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "1",
                "--in-memory"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

    def test_h5(self):
        out_file = self.temp_tiffs.parent.joinpath(f"out_h.h5")
        args = [self.temp_file.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "1",
                "--h5-loc-in", "data/ch0",
                "--h5-loc-out", "data/ch1",
                ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f

    @pytest.mark.parametrize("chunks", [None, "infer", "1, 10, 10"])
    def test_chunks(self, chunks):
        out_file = self.temp_tiffs.parent.joinpath(f"out_inf.h5")
        args = [self.temp_tiffs.as_posix(),
                "--output-path", out_file.as_posix(),
                "--num-channels", "1",
                "--chunks", chunks
                ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f

            if chunks is None:
                assert f["data/ch0"].chunks == (10, 25, 25)

            elif chunks == "infer":
                assert f["data/ch0"].chunks != None

            else:
                assert f["data/ch0"].chunks == tuple(map(int, chunks.split(",")))

class Test_MotionCorrection:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        h5_loc = "data/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(shape=(250, 250, 250),
                                         skip_n=5, event_intensity=100, background_noise=1,
                                         gap_space=5, gap_time=3)

        io = IO()
        io.save(path=path, data=video, h5_loc=h5_loc)

        assert path.exists()

        self.temp_dir = temp_dir
        self.video_path = path
        self.h5_loc = h5_loc
        self.num_events = num_events

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def run_with_parameters(self, params):

        args = [self.video_path.as_posix(), "--h5-loc", self.h5_loc]
        args += params

        result = self.runner.invoke(motion_correction, args)

        assert result.exit_code == 0, f"error: {result.output}"

    def test_custom_output_path(self):

        out = self.video_path.with_suffix(f".custom.h5")
        self.run_with_parameters(["--output-path", out.as_posix()])

    def test_custom_chunks(self):
        self.run_with_parameters(["--infer-chunks", "--chunks", "2", "10", "10"])

    def test_non_inference(self):
        self.run_with_parameters(["--infer-chunks"])


class Test_SubtractDelta:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = Path(temp_dir.name).joinpath("temp.h5")

        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=np.random.randint(0, 100, size=(10, 100, 100)))

        self.runner = CliRunner()

        self.temp_dir = temp_dir
        self.temp_file = temp_file

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_default(self):

        out_file = self.temp_file.with_suffix(".def.h5")
        result = self.runner.invoke(subtract_delta, [self.temp_file.as_posix(),
                                                     "--output-path", out_file.as_posix(),
                                                     "--h5-loc-in", "data/ch0",
                                                     "--window", 2])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        with h5.File(self.temp_file.as_posix(), "r") as f:
            assert "data/ch0" in f

    @pytest.mark.parametrize("method", ['background', 'dF', 'dFF'])
    def test_method(self, method):

        out_file = self.temp_file.with_suffix(".met.h5")
        result = self.runner.invoke(subtract_delta, [self.temp_file.as_posix(),
                                                     "--output-path", out_file.as_posix(),
                                                     "--h5-loc-in", "data/ch0",
                                                     "--method", method,
                                                     "--window", 2])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        with h5.File(self.temp_file.as_posix(), "r") as f:
            assert "data/ch0" in f

    def test_pchunks(self):

        out_file = self.temp_file.with_suffix(".met.h5")
        result = self.runner.invoke(subtract_delta, [self.temp_file.as_posix(),
                                                     "--output-path", out_file.as_posix(),
                                                     "--h5-loc-in", "data/ch0",
                                                     "--processing-chunks", "1,25,25",
                                                     "--window", 3])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        with h5.File(self.temp_file.as_posix(), "r") as f:
            assert "data/ch0" in f

    def test_overwrite(self):

        out_file = self.temp_file.with_suffix(".ov.h5")
        result = self.runner.invoke(subtract_delta, [self.temp_file.as_posix(),
                                                     "--output-path", out_file.as_posix(),
                                                     "--h5-loc-in", "data/ch0",
                                                     "--overwrite-first-frame", False,
                                                     "--window", 3])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        with h5.File(self.temp_file.as_posix(), "r") as f:
            assert "data/ch0" in f

    def test_lazy(self):

        out_file = self.temp_file.with_suffix(".laz.h5")
        result = self.runner.invoke(subtract_delta, [self.temp_file.as_posix(),
                                                     "--output-path", out_file.as_posix(),
                                                     "--h5-loc-in", "data/ch0",
                                                     "--lazy", False,
                                                     "--window", 3])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        with h5.File(self.temp_file.as_posix(), "r") as f:
            assert "data/ch0" in f

class Test_TrainDenoiser:

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_default(self):
        raise NotImplementedError

class Test_Denoise:

    def setup_method(self):
        pass

    def teardown_method(self):
        pass


class Test_Detection:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        h5_loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(shape=(250, 250, 250),
                                         skip_n=5, event_intensity=100, background_noise=1,
                                         gap_space=5, gap_time=3)

        io = IO()
        io.save(path=path, data=video, h5_loc=h5_loc)

        assert path.exists()

        self.temp_dir = temp_dir
        self.video_path = path
        self.h5_loc = h5_loc
        self.num_events = num_events

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def run_with_parameters(self, params):

        out = self.video_path.with_suffix(f".{np.random.randint(1, int(10e6), size=1)}.roi")
        args = [self.video_path.as_posix(), "--h5-loc", self.h5_loc, "--output-path", out.as_posix()]
        args += params

        result = self.runner.invoke(detect_events, args)

        assert result.exit_code == 0, f"error: {result.output}"

        events = Events(out)

        assert out.is_dir(), "Output folder does not exist"

        if "--split-events" in params:
            assert len(events) >= self.num_events
        elif "--subset" in params:
            assert len(events) <= self.num_events
        else:
            assert np.allclose(len(events), self.num_events, rtol=0.1), f"Number of events does not match: {len(events)} vs {self.num_events}"

    def test_default(self):
        self.run_with_parameters([])

    def test_threshold(self):
        self.run_with_parameters(["--threshold", "10"])

    def test_exclude_border(self):
        self.run_with_parameters(["--exclude-border", "5"])

    def test_no_spatial(self):
        self.run_with_parameters(["--use-spatial", False])

    def test_no_temporal(self):
        self.run_with_parameters(["--use-temporal", False, "--spatial-min-ratio", "5"])

    def test_lazy(self):
        self.run_with_parameters(["--lazy", False])

    def test_adjust_for_noise(self):
        self.run_with_parameters(["--adjust-for-noise", True])

    def test_serial(self):
        self.run_with_parameters(["--parallel", False])

    def test_split_events(self):
        self.run_with_parameters(["--split-events", True])

    def test_subset(self):
        self.run_with_parameters(["--subset", "0", "100"])

    def test_depth(self):
        self.run_with_parameters(["--holes-depth", "2", "--objects-depth", "2"])

class Test_Export_Video:

    def setup_method(self):
        size = (10, 100, 100)

        self.A = np.random.randint(0, 100, size=size)
        self.B = np.random.randint(0, 100, size=size)

        # Create a CliRunner to invoke the command
        self.runner = CliRunner()

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = Path(temp_dir.name).joinpath("temp.h5")

        # Create a temporary file to store the datasets
        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=self.A)
            f.create_dataset("data/ch1", data=self.B)

        self.temp_dir = temp_dir
        self.temp_file = temp_file

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_tiff(self):

        out_file = self.temp_file.with_suffix(".tiff")
        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0"])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        data = tifffile.imread(out_file.as_posix())
        assert np.allclose(self.A, data)

    def test_alternative_h5(self):

        out_file = self.temp_file.with_suffix(".alt.h5")

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0",
                                          "--h5-loc-out", "cop/ch0"])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "cop/ch0" in f
            assert np.allclose(self.A, f["cop/ch0"])

    def test_overwrite(self):

        new_path = self.temp_file.with_suffix(".ov.h5")
        shutil.copy(self.temp_file.as_posix(), new_path.as_posix())

        out_file = new_path
        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch1",
                                          "--h5-loc-out", "data/ch0",
                                          "--overwrite", True])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        assert out_file.exists()
        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.B, f["data/ch0"])

    @pytest.mark.parametrize("rescale", [0.5, 1.0, 2.0])
    def test_rescale(self, rescale):

        out_file = self.temp_file.with_suffix(f".resc.{rescale}.h5")

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0",
                                          "--h5-loc-out", "data/ch0",
                                          "--chunk-size", "1", "5", "5",
                                          "--rescale", rescale])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}\n{result.exception}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            with h5.File(self.temp_file.as_posix(), "r") as ft:

                data = f["data/ch0"]
                A = ft["data/ch0"]

                assert data.shape == (A.shape[0], A.shape[1]*rescale, A.shape[2]*rescale)

    def test_compression(self):

        out_file = self.temp_file.with_suffix(".comp.h5")

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0",
                                          "--h5-loc-out", "data/ch0",
                                          "--compression", "gzip"])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].compression == "gzip"

    def test_z_select(self):

        out_file = self.temp_file.with_suffix(".z.h5")

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0",
                                          "--h5-loc-out", "data/ch0",
                                          "--z-select", "0", "2"])
        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].shape == (2, self.A.shape[1], self.A.shape[2])

    def test_chunk(self):

        out_file = self.temp_file.with_suffix(".chunk.h5")

        result = self.runner.invoke(export_video, [self.temp_file.as_posix(),
                                          "--output-path", out_file.as_posix(),
                                          "--h5-loc-in", "data/ch0",
                                          "--h5-loc-out", "data/ch0",
                                          "--chunk-size", "1", "5", "5"])
        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].chunks == (1, 5, 5)

class Test_MoveDataset:

    def setup_method(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name).joinpath("temp.h5")
        self.A = np.random.randint(0, 100, size=(10, 100, 100))
        self.B = np.random.randint(0, 100, size=(10, 100, 100))

        # Create a temporary file to store the datasets
        with h5.File(self.temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=self.A)
            f.create_dataset("data/ch1", data=self.B)

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_move_dataset(self):

        temp_file_2 = self.temp_file.with_suffix(".2.h5")

        result = self.runner.invoke(move_h5_dataset,
                                    [self.temp_file.as_posix(),  temp_file_2.as_posix(),
                                     "data/ch0", "data/ch0"])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        assert temp_file_2.exists()
        with h5.File(temp_file_2.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.A, f["data/ch0"])

    def test_overwrite_dataset(self):

        temp_file_2 = self.temp_file.with_suffix(".2.h5")
        shutil.copy(self.temp_file.as_posix(), temp_file_2.as_posix())

        result = self.runner.invoke(move_h5_dataset,
                                    [self.temp_file.as_posix(),  temp_file_2.as_posix(),
                                     "data/ch1", "data/ch0",
                                     "--overwrite", True])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        assert temp_file_2.exists()
        with h5.File(temp_file_2.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.B, f["data/ch0"])

class Test_ViewData:

    def setup_method(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name).joinpath("temp.h5")
        self.A = np.random.randint(0, 100, size=(10, 100, 100))
        self.B = np.random.randint(0, 100, size=(10, 100, 100))

        with h5.File(self.temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=self.A)
            f.create_dataset("data/ch1", data=self.B)

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_view_data(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0",
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_data_color(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0",
                                                "--colormap", "plasma",
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_data_z_select(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0",
                                                "--z-select", "1", "5",
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_data_lazy(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0",
                                                "--lazy", False,
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_data_trace(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0",
                                                "--show-trace", True,
                                                "--window", "5",
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_data_multi(self):

        result = self.runner.invoke(view_data, [self.temp_file.as_posix(), "data/ch0", "data/ch1",
                                                "--testing", True])

        assert result.exit_code == 0, f"error: {result.output}"

class Test_ViewDetectionResults:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        h5_loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(shape=(50, 100, 100), skip_n=5, event_intensity=100, background_noise=1)
        io = IO()
        io.save(path=path, data=video, h5_loc=h5_loc)

        det = Detector(path.as_posix(), output=None)
        events = det.run(h5_loc=h5_loc, lazy=True, debug=False)

        dir_ = det.output_directory

        assert dir_.is_dir(), "Output folder does not exist"
        assert bool(det.meta), "metadata dictionary is empty"
        assert det.data.size != 0, "data object is empty"
        assert det.data.shape is not None, "data has no dimensions"
        assert path.exists(), "Cannot find video file: {path}"

        self.temp_dir = temp_dir
        self.video_path = path
        self.event_dir = dir_
        self.h5_loc = h5_loc

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_view_detection_results(self):

        event_dir = str(self.event_dir.as_posix())

        result = self.runner.invoke(view_detection_results, [event_dir,
                                                             "--testing", True])
        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_detection_infer(self):

        result = self.runner.invoke(view_detection_results, [self.event_dir.as_posix(),
                                                             "--video-path", "infer",
                                                             "--h5-loc", str(self.h5_loc),
                                                             "--testing", True])
        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_detection_z(self):

        result = self.runner.invoke(view_detection_results, [self.event_dir.as_posix(),
                                                             "--video-path", self.video_path.as_posix(),
                                                             "--h5-loc", str(self.h5_loc),
                                                             "--z-select", "0", "5",
                                                             "--testing", True])
        assert result.exit_code == 0, f"error: {result.output}"

    def test_view_detection_lazy(self):

        result = self.runner.invoke(view_detection_results, [self.event_dir.as_posix(),
                                                             "--video-path", self.video_path.as_posix(),
                                                             "--h5-loc", str(self.h5_loc),
                                                             "--lazy", False,
                                                             "--testing", True])
        assert result.exit_code == 0, f"error: {result.output}"

def test_delete_h5_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file = Path(temp_dir).joinpath("temp.h5")

        # Create a temporary file to store the datasets
        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("test", data=[1, 2, 3])
            f.create_dataset("test2", data=[4, 5, 6])

        assert temp_file.exists()

        # Create a CliRunner to invoke the command
        runner = CliRunner()

        # Use the CliRunner to invoke the command with the input file as argument
        result = runner.invoke(delete_h5_dataset, [temp_file.as_posix(), "--loc", "test"])

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

        # Check that the dataset was deleted
        with h5.File(temp_file.as_posix(), "r") as f:

            assert "test2" in f
            assert "test" not in f

def test_visualize_h5():

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file = Path(temp_dir).joinpath("temp.h5")
        A = np.random.randint(0, 100, size=(10, 100, 100))
        B = np.random.randint(0, 100, size=(10, 100, 100))

        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=A)
            f.create_dataset("data/ch1", data=B)
            f.create_dataset("mc/ch0", data=B)
            f.create_dataset("mc/ch1", data=A)
            f.create_group("dummy")

        result = CliRunner().invoke(visualize_h5, [temp_file.as_posix()])
        assert result.exit_code == 0, f"error: {result.output}"

@pytest.mark.parametrize("z", ["0", "1,2"])
@pytest.mark.parametrize("size", [(8, 8), (50, 50), (75, 75)])
@pytest.mark.parametrize("equalize", [True, False])
def test_climage(z, size, equalize):

    data = np.random.randint(0, 100, (4, 100, 100))

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file = Path(temp_dir).joinpath("temp.h5")

        # Create a temporary file to store the datasets
        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=data)

        # Create a CliRunner to invoke the command
        runner = CliRunner()

        # collect argumnts
        args = [temp_file.as_posix()]

        for z_ in z.split(","):
            args.append(z_)

        args += ["--size", size[0], size[1], "--equalize", str(equalize)]

        # Use the CliRunner to invoke the command with the input file as argument
        result = runner.invoke(climage, args=args)

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}"

def test_explorer_gui():
    raise NotImplementedError

def test_analysis_gui():
    raise NotImplementedError
