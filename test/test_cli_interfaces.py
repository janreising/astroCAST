import platform
import tempfile
import traceback

import h5py as h5
import pytest
import tifffile
from click.testing import CliRunner

from astrocast.analysis import Events
from astrocast.cli_interfaces import *
from astrocast.detection import Detector
from astrocast.helper import EventSim, is_docker
from astrocast.preparation import IO


class TestConvertInput:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()

        temp_file = Path(temp_dir.name).joinpath("temp.h5")
        self.arr_size = (10, 25, 25)
        with h5.File(temp_file.as_posix(), "a") as f:
            f.create_dataset("data/ch0", data=np.random.randint(0, 100, size=self.arr_size))

        temp_tiffs = Path(temp_dir.name).joinpath("imgs/")
        temp_tiffs.mkdir()
        for i in range(216):
            tifffile.imwrite(
                temp_tiffs.joinpath(f"img_{i}.tiff").as_posix(), data=np.random.randint(0, 100, size=(1, 25, 25))
            )

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
            channel_names = [f"data/ch{i}" for i in range(num_channels)]
        else:
            channel_names = [f"data/{name_prefix}{i}" for i in range(num_channels)]

        args = [self.temp_tiffs.as_posix(), "--output-path", out_file.as_posix(), "--num-channels", str(num_channels)]
        if name_prefix is None and num_channels > 1:
            args.append("--loc-out")
            args.append("data")

        elif name_prefix is None and num_channels == 1:
            args.append("--loc-out")
            args.append("data/ch0")

        else:
            args.append("--channel-names")
            args.append(",".join(channel_names))

        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:

            for ch_name in channel_names:

                if ch_name not in f:
                    logging.error(f"name: {ch_name} not in output file: {list(f.keys())}")
                    if "/" in ch_name:
                        base_name = ch_name.split("/")[0]
                        logging.error(f"{base_name}/... > {list(f[base_name].keys())}")

                    raise FileNotFoundError

                assert ch_name in f, f"name: {ch_name} not in output file: {list(f.keys())}"

            lengths = [len(f[ch_name]) for ch_name in channel_names]
            assert len(np.unique(lengths)) == 1, f"lengths: {lengths}"

    def test_z_slice(self):

        out_file = self.temp_tiffs.parent.joinpath(f"out_z.h5")
        args = [self.temp_tiffs.as_posix(), "--output-path", out_file.as_posix(), "--num-channels", "1", "--z-slice",
                "0", "50", "--loc-out", "data/ch0"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].shape == (50, 25, 25)

    def test_lazy(self):
        out_file = self.temp_tiffs.parent.joinpath(f"out_l.h5")
        args = [self.temp_tiffs.as_posix()]
        args += ["--output-path", out_file.as_posix()]
        args += ["--num-channels", "1"]
        args += ["--z-slice", "0", "50"]
        args += ["--lazy"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

    def test_subtract_background(self):

        out_file_ref = self.temp_tiffs.parent.joinpath(f"out_ref.h5")
        args = [self.temp_tiffs.as_posix(), "--output-path", out_file_ref.as_posix(), "--num-channels", "2",
                "--loc-out", "data"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file_ref.exists()

        out_file = self.temp_tiffs.parent.joinpath(f"out_sb.h5")
        args = [self.temp_tiffs.as_posix(), "--output-path", out_file.as_posix(), "--num-channels", "2",
                "--subtract-background", "data/ch1", "--loc-out", "data"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
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
        args = [self.temp_tiffs.as_posix(), "--output-path", out_file.as_posix(), "--num-channels", "1", "--in-memory"]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

    def test_h5(self):
        out_file = self.temp_tiffs.parent.joinpath(f"out_h.h5")
        args = [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--num-channels", "1", "--loc-in",
                "data/ch0", "--loc-out", "data/ch1", ]
        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch1" in f

    @pytest.mark.parametrize("chunks", [None, ["2", "10", "10"]])
    @pytest.mark.parametrize("chunk_strategy", [None, "balanced", "XY", "Z"])
    def test_chunks(self, chunks, chunk_strategy):

        out_file = self.temp_tiffs.parent.joinpath(f"out_inf_{str(chunk_strategy)}_{str(chunks)}.h5")
        args = [self.temp_tiffs.as_posix()]
        args += ["--output-path", out_file.as_posix()]
        args += ["--num-channels", "1"]
        args += ["--loc-out", "data/ch0"]

        if chunks is not None:
            args += ["--chunks"] + chunks

        if chunk_strategy is not None:
            args += ["--chunk-strategy", chunk_strategy]
        else:
            args += ["--chunk-strategy", "None"]

        result = self.runner.invoke(convert_input, args)

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f

            saved_chunks = f["data/ch0"].chunks
            if saved_chunks is not None:
                cz, cx, cy = saved_chunks
            Z, X, Y = f["data/ch0"].shape

            if chunks is None and chunk_strategy is None:
                assert saved_chunks is None or saved_chunks == f["data/ch0"].shape

            elif chunks is not None:
                assert saved_chunks == tuple(map(int, chunks))

            elif chunk_strategy == "balanced":
                assert cz > 1
                assert cx > 1
                assert cy > 1

            elif chunk_strategy == "XY":
                assert cz > 1
                assert cx == X
                assert cy == Y

            elif chunk_strategy == "Z":
                assert cz == Z
                assert cx > 1
                assert cy > 1

            else:
                raise ValueError(f"unexpected condition: {chunks} & {chunk_strategy}")


class TestMotionCorrection:

    def setup_method(self):
        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        loc = "data/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(
            shape=(250, 250, 250), skip_n=5, event_intensity=100, background_noise=1, gap_space=5, gap_time=3
        )

        io = IO()
        io.save(path=path, data=video, loc=loc)

        assert path.exists()

        self.temp_dir = temp_dir
        self.video_path = path
        self.loc = loc
        self.num_events = num_events

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def run_with_parameters(self, params):
        args = [self.video_path.as_posix(), "--loc-in", self.loc]
        args += params

        result = self.runner.invoke(motion_correction, args)

        assert result.exit_code == 0, f"error: {result.output}"

    def test_custom_output_path(self):
        out = self.video_path.with_suffix(f".custom.h5")
        self.run_with_parameters(["--output-path", out.as_posix()])

    def test_custom_chunks(self):
        self.run_with_parameters(["--chunks", "2", "10", "10"])

    def test_non_inference(self):
        self.run_with_parameters(["--chunk-strategy", None])


class TestSubtractDelta:

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
        result = self.runner.invoke(
            subtract_delta,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "df/ch0", "--window", 2]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        with h5.File(out_file.as_posix(), "r") as f:
            assert "df/ch0" in f

    @pytest.mark.parametrize("method", ['background', 'dF', 'dFF'])
    def test_method(self, method):
        out_file = self.temp_file.with_suffix(".met.h5")
        result = self.runner.invoke(
            subtract_delta,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "df/ch0", "--method", method, "--window", 2]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        with h5.File(out_file.as_posix(), "r") as f:
            assert "df/ch0" in f

    def test_manual_chunks(self):
        out_file = self.temp_file.with_suffix(".met.h5")
        result = self.runner.invoke(
            subtract_delta,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "df/ch0", "--chunks", "1", "10", "10", "--window", 3]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        with h5.File(out_file.as_posix(), "r") as f:
            assert "df/ch0" in f
            assert f["df/ch0"].chunks == (1, 10, 10)

    @pytest.mark.parametrize("overwrite", [True, False])
    def test_overwrite(self, overwrite):
        out_file = self.temp_file.with_suffix(".ov.h5")
        result = self.runner.invoke(
            subtract_delta,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "df/ch0", "--overwrite-first-frame", overwrite, "--window", 3]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        with h5.File(out_file.as_posix(), "r") as f:
            assert "df/ch0" in f

            if overwrite:
                assert np.allclose(f["df/ch0"][0], f["df/ch0"][1])
            else:
                assert not np.allclose(f["df/ch0"][0], f["df/ch0"][1])


class TestTrainDenoiserDenoise:

    def setup_method(self):
        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        # parameters
        loc = "data/ch0"
        X, Y = (250, 250)

        sim = EventSim()
        io = IO()
        train_dir = tmpdir.joinpath("train")
        train_dir.mkdir()

        # create training files
        for i in range(5):
            video, _ = sim.simulate(
                shape=(100, X, Y), skip_n=5, event_intensity=100, background_noise=1, gap_space=5, gap_time=3
            )
            io.save(path=train_dir.joinpath(f"train_{i}.h5"), data=video, loc=loc)

        # create validation files
        for i in range(2):
            video, _ = sim.simulate(
                shape=(50, X, Y), skip_n=5, event_intensity=100, background_noise=1, gap_space=5, gap_time=3
            )
            io.save(path=train_dir.joinpath(f"val_{i}.h5"), data=video, loc=loc)

        # create inf file
        video, _ = sim.simulate(
            shape=(25, X, Y), skip_n=5, event_intensity=100, background_noise=1, gap_space=5, gap_time=3
        )

        inf_path = tmpdir.joinpath(f"inf.h5")
        io.save(path=inf_path, data=video, loc=loc)

        # model path
        model_path = tmpdir.joinpath("model.h5")

        # make sure directories exist
        assert inf_path.is_file()
        assert train_dir.is_dir()
        assert len(list(train_dir.glob("*"))) > 3

        self.temp_dir = temp_dir
        self.runner = CliRunner()
        self.loc = loc
        self.train_dir = train_dir
        self.inf_path = inf_path
        self.model_path = model_path

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_train_inf(self):

        args = ["--training-files", self.train_dir.joinpath("train_*.h5").as_posix()]
        args += ["--validation-files", self.train_dir.joinpath("val_*.h5").as_posix()]
        args += ["--input-size", "128", "128"]
        args += ["--loc", self.loc]
        args += ["--epochs", 2]
        args += ["--pre-post-frames", 2]
        args += ["--max-per-file", 2]
        args += ["--max-per-val-file", 2]
        args += ["--save-path", self.model_path]
        results = self.runner.invoke(train_denoiser, args)

        assert results.exit_code == 0, f"error: {results.output}"
        assert self.model_path.exists(), f"{list(self.train_dir.parent.glob('*'))}"

        args = []
        args += [self.inf_path.as_posix()]
        args += ["--model", self.model_path]
        args += ["--loc", self.loc]
        args += ["--out-loc", "inf/ch0"]
        args += ["--input-size", "128", "128"]
        args += ["--pre-post-frames", 2]
        results = self.runner.invoke(denoise, args)

        assert results.exit_code == 0, f"error: {results.output}"
        assert self.inf_path.is_file()
        with h5.File(self.inf_path.as_posix(), "r") as f:
            assert "inf/ch0" in f
            assert f["data/ch0"].shape == f["inf/ch0"].shape


class TestDetection:

    def setup_method(self):

        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(
            shape=(250, 250, 250), skip_n=5, event_intensity=100, background_noise=1, gap_space=5, gap_time=3
        )

        io = IO()
        io.save(path=path, data=video, loc=loc)

        assert path.exists()

        self.temp_dir = temp_dir
        self.video_path = path
        self.loc = loc
        self.num_events = num_events

        self.runner = CliRunner()

    def teardown_method(self):

        # necessary to give Windows time to release files
        if platform.system() == "Windows":

            for file in list(Path(self.temp_dir.name).glob("*/*")):
                file.unlink(missing_ok=True)

            logging.warning(f"Assuming to be on windows. Waiting for files to be released!")
            time.sleep(20)

        self.temp_dir.cleanup()

    def run_with_parameters(self, params):

        out = self.video_path.with_suffix(f".{np.random.randint(1, int(10e6), size=1)}.roi")
        args = [self.video_path.as_posix(), "--loc", self.loc, "--output-path", out.as_posix()]

        # check container
        if is_docker():
            logging.warning("Suspecting to be in container, switching to 'on_disk=True'.")
            args += ["--on-disk", True]

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
            assert np.allclose(
                len(events), self.num_events, rtol=0.1
            ), f"Number of events does not match: {len(events)} vs {self.num_events}"

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


class TestExportVideo:

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
        result = self.runner.invoke(
            export_video, [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0"]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        data = tifffile.imread(out_file.as_posix())
        assert np.allclose(self.A, data)

    def test_alternative_h5(self):
        out_file = self.temp_file.with_suffix(".alt.h5")

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "cop/ch0"]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
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

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch1", "--loc-out",
             "data/ch0", "--overwrite", True]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        assert out_file.exists()
        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.B, f["data/ch0"])

    @pytest.mark.parametrize("rescale", [0.5, 1.0, 2.0])
    def test_rescale(self, rescale):
        with h5.File(self.temp_file.as_posix(), "r") as f_in:
            data_in = f_in["data/ch0"]

        out_file = self.temp_file.with_suffix(f".resc.{str(rescale).replace('.', '')}.h5")

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "data/ch0", "--rescale", rescale]
        )

        # Check that the command ran successfully
        assert result.exit_code == 0, f"error: {result.output}\n{result.exception}"
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f_out:
            with h5.File(self.temp_file.as_posix(), "r") as f_in:
                data_in = f_in["data/ch0"]
                data_out = f_out["data/ch0"]

                exp_shape = (data_in.shape[0], data_in.shape[1] * rescale, data_in.shape[2] * rescale)
                out_shape = data_out.shape

                logging.warning(f"data_in; exp: out: {data_in.shape}; {exp_shape}, {out_shape}")

                assert exp_shape == out_shape, f"rescaling factor: {rescale}"

    def test_compression(self):
        out_file = self.temp_file.with_suffix(".comp.h5")

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "data/ch0", "--compression", "gzip"]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].compression == "gzip"

    def test_z_select(self):
        out_file = self.temp_file.with_suffix(".z.h5")

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "data/ch0", "--z-select", "0", "2"]
        )
        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].shape == (2, self.A.shape[1], self.A.shape[2])

    def test_chunk(self):
        out_file = self.temp_file.with_suffix(".chunk.h5")

        result = self.runner.invoke(
            export_video,
            [self.temp_file.as_posix(), "--output-path", out_file.as_posix(), "--loc-in", "data/ch0", "--loc-out",
             "data/ch0", "--chunk-size", "1", "5", "5"]
        )
        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception
        assert out_file.exists()

        with h5.File(out_file.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert f["data/ch0"].chunks == (1, 5, 5)


class TestMoveDataset:

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

        result = self.runner.invoke(
            move_h5_dataset, [self.temp_file.as_posix(), temp_file_2.as_posix(), "data/ch0", "data/ch0"]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        assert temp_file_2.exists()
        with h5.File(temp_file_2.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.A, f["data/ch0"])

    def test_overwrite_dataset(self):
        temp_file_2 = self.temp_file.with_suffix(".2.h5")
        shutil.copy(self.temp_file.as_posix(), temp_file_2.as_posix())

        result = self.runner.invoke(
            move_h5_dataset,
            [self.temp_file.as_posix(), temp_file_2.as_posix(), "data/ch1", "data/ch0", "--overwrite", True]
        )

        # Check that the command ran successfully
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

        assert temp_file_2.exists()
        with h5.File(temp_file_2.as_posix(), "r") as f:
            assert "data/ch0" in f
            assert np.allclose(self.B, f["data/ch0"])


class TestViewData:

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

    @pytest.mark.vis
    def test_view_data(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_data_color(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "--colormap", "plasma", "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_data_z_select(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "--z-select", "1", "5", "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_data_lazy(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "--lazy", False, "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_data_trace(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "--show-trace", True, "--window", "5", "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_data_multi(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_data, [self.temp_file.as_posix(), "data/ch0", "data/ch1", "--testing", True]
        )

        assert result.exit_code == 0, f"error: {result.output}"


class TestViewDetectionResults:

    def setup_method(self):
        temp_dir = tempfile.TemporaryDirectory()
        tmpdir = Path(temp_dir.name)
        assert tmpdir.is_dir()

        path = tmpdir.joinpath("sim.h5")
        loc = "df/ch0"

        sim = EventSim()
        video, num_events = sim.simulate(shape=(50, 100, 100), skip_n=5, event_intensity=100, background_noise=1)
        io = IO()
        io.save(path=path, data=video, loc=loc)

        det = Detector(path.as_posix(), output=None)
        events = det.run(loc=loc, lazy=True, debug=False)

        dir_ = det.output_directory

        assert dir_.is_dir(), "Output folder does not exist"
        assert bool(det.meta), "metadata dictionary is empty"
        assert det.data.size != 0, "data object is empty"
        assert det.data.shape is not None, "data has no dimensions"
        assert path.exists(), "Cannot find video file: {path}"

        self.temp_dir = temp_dir
        self.video_path = path
        self.event_dir = dir_
        self.loc = loc

        self.runner = CliRunner()

    def teardown_method(self):
        self.temp_dir.cleanup()

    @pytest.mark.vis
    def test_view_detection_results(self):
        napari = pytest.importorskip("napari")

        event_dir = str(self.event_dir.as_posix())

        result = self.runner.invoke(
            view_detection_results, [event_dir, "--testing", True]
        )
        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_detection_infer(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_detection_results,
            [self.event_dir.as_posix(), "--video-path", "infer", "--loc", str(self.loc), "--testing", True]
        )
        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_detection_z(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_detection_results,
            [self.event_dir.as_posix(), "--video-path", self.video_path.as_posix(), "--loc", str(self.loc),
             "--z-select", "0", "5", "--testing", True]
        )
        assert result.exit_code == 0, f"error: {result.output}"

    @pytest.mark.vis
    def test_view_detection_lazy(self):
        napari = pytest.importorskip("napari")

        result = self.runner.invoke(
            view_detection_results,
            [self.event_dir.as_posix(), "--video-path", self.video_path.as_posix(), "--loc", str(self.loc), "--lazy",
             False, "--testing", True]
        )
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
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception

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
        if result.exit_code != 0:
            print(f"error: {result.output}")
            traceback.print_exception(*result.exc_info)
            raise Exception


@pytest.mark.skip(reason="Not implemented")
def test_explorer_gui():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_analysis_gui():
    pass
