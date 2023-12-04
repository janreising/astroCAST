import logging
import platform
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import dask.array as da
import tiledb
from tifffile import tifffile

from astrocast.helper import SampleInput, remove_temp_safe
from astrocast.preparation import Delta, Input, IO, MotionCorrection


class TestDelta:

    @pytest.mark.parametrize("input_type", [np.ndarray, ".tiff", ".h5", "tiledb"])
    def test_load_save(self, input_type, shape=(50, 10, 10)):

        Z, X, Y = shape
        loc = None

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            if input_type == np.ndarray:
                data = np.random.randint(0, 100, (Z, X, Y), dtype=int)

            elif input_type == "tiledb":

                arr = da.from_array(
                    x=np.random.randint(0, 100, (Z, X, Y), dtype=int), chunks=(Z, "auto", "auto")
                )

                loc = None
                path = tmpdir.joinpath("temp.tdb")
                arr.to_tiledb(path.as_posix())
                data = path

            elif isinstance(input_type, str):

                si = SampleInput()
                data = si.get_test_data(extension=input_type)
                loc = si.get_loc()

            else:
                raise TypeError

            delta = Delta(data, loc=loc)

            delta.run(method="background", window=5)

            # save
            output_path = tmpdir.joinpath("out.tiff")
            delta.save(output_path)
            assert output_path.exists()

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_methods_run(self, method, lazy):

        with tempfile.TemporaryDirectory() as tmpdir:
            Z, X, Y = 25, 2, 2

            data = np.random.randint(0, 100, (Z, X, Y), dtype=int)
            loc = None

            delta = Delta(data, loc=loc)

            delta.run(method=method, window=5)

    @pytest.mark.parametrize("dim", [(100), (100, 5), (100, 5, 5), (100, 2, 10)])
    def test_background_dimensions(self, dim):

        arr = np.random.randint(0, 100, dim, dtype=int)
        orig_shape = arr.shape

        res = Delta._calculate_delta_min_filter(arr, window=10)

        assert res.shape == orig_shape, f"dimensions are not the same input: {dim} vs output: {res.shape}"

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_result_for_parallel(self, method, lazy):

        dim = (250, 50, 50)
        window = 10

        arr = np.random.randint(0, 100, dim, dtype=int)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = Delta._calculate_delta_min_filter(arr.copy(), window, method=method)
            logging.warning(f"sum of ctrl: {np.sum(ctrl)}")

            delta = Delta(arr, loc=None)
            res = delta.run(method=method, window=window, overwrite_first_frame=False)

            assert np.allclose(ctrl, res)

    @pytest.mark.skip(reason="Not implemented")
    def test_quality_of_dff(self):
        raise NotImplementedError


class TestInput:

    @pytest.mark.parametrize("num_files", [1, 12])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_convert_single_tiff_series(self, num_files, in_memory):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = []
            for n in range(num_files):
                img = np.random.random((1, 10, 10))
                images.append(img)

                tifffile.imwrite(tmpdir.joinpath(f"ss_single_{n}.tiff"), img)

            img_stack = np.stack(images)
            img_stack = np.squeeze(img_stack)

            # Loaded
            inp = Input()

            tmpdir = list(tmpdir.glob("*"))[0] if num_files == 1 else tmpdir
            stack = inp.run(input_path=tmpdir, dtype=None, in_memory=in_memory, loc_out="data")
            stack = stack["data/ch0"]

            img_stack = np.squeeze(img_stack)
            stack = np.squeeze(stack)

            assert img_stack.shape == stack.shape
            assert np.array_equal(img_stack, stack)

    @pytest.mark.parametrize("num_channels", [2, 3])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_convert_multi_channel(self, num_channels, in_memory):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = {f"data/ch{n}": [] for n in range(num_channels)}
            c = 0
            for _ in range(7):
                for n in range(num_channels):
                    img = np.random.random((1, 10, 10))
                    images[f"data/ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c = c + 1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            # Loaded
            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels, dtype=None, in_memory=in_memory, loc_out="data")

            for ch in images.keys():
                ref = np.squeeze(images[ch])
                res = np.squeeze(stack[ch])

                assert ref.shape == res.shape
                assert np.array_equal(ref, res)

    @pytest.mark.parametrize("num_channels", [1, 2, 3])
    @pytest.mark.parametrize("subtract_background", ["arr"])
    @pytest.mark.parametrize("subtract_func", [np.mean, "mean", "std", "min", "max"])
    def test_subtract(self, num_channels, subtract_background, subtract_func, in_memory=True):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            X, Y = 10, 10

            # Reference
            images = {f"data/ch{n}": [] for n in range(num_channels)}
            c = 0
            for _ in range(7):
                for n in range(num_channels):
                    img = np.random.random((1, X, Y))
                    images[f"data/ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c = c + 1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            if num_channels == 1:
                subtract_background = np.random.random((X, Y))

            else:
                subtract_background = "data/ch0"

            # Loaded
            inp = Input()
            stack = inp.run(
                input_path=tmpdir, channels=num_channels, subtract_background=subtract_background, loc_out="data",
                subtract_func=subtract_func, dtype=None, in_memory=in_memory
            )

            # check result
            func_reduction = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}

            if num_channels == 1:

                assert np.array_equal(
                    stack["data/ch0"], images["data/ch0"] - subtract_background
                )

            else:

                func = func_reduction[subtract_func] if not callable(subtract_func) else subtract_func
                background = func(images["data/ch0"], axis=0)

                for ch in stack.keys():

                    if ch == "data/ch0":
                        pass

                    ctrl = images[ch] - background
                    res = stack[ch]

                    assert ctrl.shape == res.shape, f"dimensions are not equal: {ctrl.shape} vs. {res.shape}"
                    assert np.allclose(res, ctrl), "values are not equal"

    @pytest.mark.parametrize("rescale", [1, 0.5, 20, (0.5, 0.3), (20, 15)])
    def test_resize(self, rescale, num_channels=1):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            Z = np.random.randint(5, 25)
            X = np.random.randint(10, 50)
            Y = np.random.randint(10, 50)
            logging.warning(f"random video shape: {(Z, X, Y)}")

            # Reference
            images = {f"data/ch{n}": [] for n in range(num_channels)}
            c = 0
            for n in range(Z):
                for channel_num in range(num_channels):
                    img = np.random.random((1, X, Y))
                    images[f"data/ch{channel_num}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c = c + 1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            inp = Input()
            stack = inp.run(
                input_path=tmpdir, channels=num_channels, rescale=rescale, subtract_background=None, loc_out="data",
                subtract_func=None, dtype=None, in_memory=True
            )

            res = stack["data/ch0"]

            if rescale == 1:
                assert res.shape == (Z, X, Y)
                return True

            if isinstance(rescale, (int, float)):
                r0 = r1 = rescale

            elif isinstance(rescale, (tuple, list)):
                r0, r1 = rescale
            else:
                raise ValueError(f"unknown rescale format: {rescale}")

            if isinstance(r0, int):
                # absolute value
                assert np.allclose(res.shape, (Z, r0, r1), atol=1)

            elif isinstance(r1, float):
                # relative value
                assert np.allclose(res.shape, (Z, int(X * r0), int(Y * r1)), atol=1)

    @pytest.mark.parametrize("output_path", ["out_def.h5", "out_def.tdb", "out_def.tiff"])
    @pytest.mark.parametrize("chunks", [None, (5, 5, 5)])
    def test_output(self, output_path, chunks, size=(25, 10, 10)):

        with tempfile.TemporaryDirectory() as _dir:
            tmpdir = Path(_dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            inp = Input()
            data = np.random.random(size)
            inp._save(output_path, data=data, chunks=chunks, loc="test/test1")

            assert output_path.exists()

    @pytest.mark.parametrize("output_path", ["out_in_out.h5", "out_in_out.tdb", "out_in_out.tiff"])
    def test_intput_output(self, output_path):

        num_files = 25

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            # Reference
            images = []
            for n in range(num_files):
                img = np.random.random((1, 10, 10))
                images.append(img)

                tifffile.imwrite(tmpdir.joinpath(f"ss_single_{n}.tiff"), img)

            img_stack = np.stack(images)
            img_stack = np.squeeze(img_stack)

            # Loaded
            inp = Input()

            tmpdir = list(tmpdir.glob("*"))[0] if num_files == 1 else tmpdir
            output_path = tmpdir.joinpath(output_path)

            inp.run(
                input_path=tmpdir, output_path=output_path, dtype=None, in_memory=False, loc_in="data",
                loc_out="data", )

            assert output_path.is_file() or output_path.is_dir(), f"cannot find output file: {output_path}"

            # load back
            if output_path.suffix == ".h5":

                with h5py.File(output_path.as_posix(), "r") as f:
                    res = f["data/ch0"][:]

            elif output_path.suffix == ".tiff":
                res = tifffile.imread(output_path.as_posix())

            elif output_path.suffix == ".tdb":
                res = tiledb.open(output_path.as_posix())

            else:
                raise NotImplementedError

            res = np.squeeze(res)

            assert img_stack.shape == res.shape
            assert np.array_equal(img_stack, res)


class TestIO:

    @pytest.mark.parametrize("prefix", ["", "00000"])
    @pytest.mark.parametrize("sep", ["_", "x", "-"])
    def test_alphanumerical_names(self, prefix, sep):

        names = []
        for n in range(1000):
            name = f"img{sep}{prefix}{n}.ext"
            names.append(name)

        names_shuffled = np.random.shuffle(names.copy())

        assert names != names_shuffled, "randomization did not work"

        io = IO()
        names_sorted = io.sort_alpha_numerical_names(names, sep=sep)

        assert names == names_sorted, "sorting did not work"

    @pytest.mark.skip(reason="Cannot create dummy .czi file.")
    def test_load_czi(self, output_path="out.czi", shape=(10, 5, 5)):
        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Loaded
            io = IO()

            prefix = None if output_path.suffix != ".h5" else "data/"
            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            data = {"ch0": arr}

            output_path = io.save(output_path, data, loc=prefix)

            arr_load = io.load(output_path, loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    def test_save_load(self, tmpdir, output_path, shape):

        if ".npy" in output_path and platform.system() in ["Windows", "win32"]:
            pytest.skip("Windows throws permission error for .npy output.")

        tmpdir = Path(tmpdir.strpath)
        assert tmpdir.is_dir()

        output_path = tmpdir.joinpath(output_path)

        # Reference
        arr = np.random.random(shape)

        # Loaded
        io = IO()

        data = {"data/ch0": arr}
        output_path = io.save(output_path, data)

        h5loc = None if output_path.suffix != ".h5" else "data/ch0"
        arr_load = io.load(output_path, loc=h5loc)

        assert arr.shape == arr_load.shape
        assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    @pytest.mark.parametrize("chunk_strategy", ["balanced", "Z", "XY", None])
    @pytest.mark.parametrize("chunks", [None, (2, 2, 2)])
    def test_save_chunks(self, shape, chunk_strategy, chunks, output_path="out.h5"):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.random.random(shape)

            # Loaded
            io = IO()

            data = {"data/ch0": arr}
            output_path = io.save(output_path, data, chunks=chunks, chunk_strategy=chunk_strategy)

            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            arr_load = io.load(output_path, loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("compression", ["gzip", "infer"])
    @pytest.mark.parametrize("shape", [(100, 100, 100), (512, 512, 350)])
    def test_save_compression(self, compression, shape, chunks=(100, 100, 100), output_path="out.h5"):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            # Reference
            arr = np.zeros(shape, dtype=float)

            # Loaded
            io = IO()

            data = {"data/ch0": arr}
            output_path = io.save(output_path, data, chunks=chunks, compression=compression)

            h5loc = None if output_path.suffix != ".h5" else "data/ch0"
            arr_load = io.load(output_path, loc=h5loc)

            assert arr.shape == arr_load.shape
            assert np.array_equal(arr, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    @pytest.mark.parametrize("shape", [(10, 5, 5), (100, 100, 100)])
    def test_z_slice(self, tmpdir, output_path, shape, z_slice=(2, 8)):

        if ".npy" in output_path and platform.system() in ["Windows", "win32"]:
            pytest.skip("Windows throws permission error for .npy output.")

        tmpdir = Path(tmpdir.strpath)
        assert tmpdir.is_dir()

        output_path = tmpdir.joinpath(output_path)

        # Reference
        arr = np.random.random(shape)

        z0, z1 = z_slice
        original_array = arr[z0:z1, :, :]

        # Loaded
        io = IO()

        output_path = io.save(output_path, arr, loc="data/ch0")
        arr_load = io.load(output_path, loc="data/ch0", z_slice=z_slice)

        assert original_array.shape == arr_load.shape
        assert np.array_equal(original_array, arr_load)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff", "out.npy"])
    def test_lazy_load(self, tmpdir, output_path, shape=(100, 100, 100)):

        tmp_path = Path(tmpdir.strpath).joinpath(f"test_{output_path.replace('.', '_')}")
        tmp_path.mkdir()
        assert tmp_path.is_dir()

        output_path = tmp_path.joinpath(output_path)

        # Reference
        arr = np.random.random(shape)

        # Saving
        io = IO()

        output_path = io.save(output_path, arr, loc="data/ch0")
        logging.warning(output_path)

        # Loading
        arr_load = io.load(output_path, loc="data/ch0", lazy=True)

        assert isinstance(arr_load, (da.Array, da.core.Array)), f"type: {type(arr_load)}"
        assert arr.shape == arr_load.shape
        assert np.array_equal(arr, arr_load)

        del io
        del arr_load

    @pytest.mark.parametrize("sep", ["_", "x"])
    def test_load_sequential_tiff(self, tmpdir, sep, shape=(100, 100, 100)):

        tmpdir = Path(tmpdir.strpath)
        assert tmpdir.is_dir()

        input_dir = tmpdir.joinpath("seq_tiff")
        input_dir.mkdir()

        # create tiff files
        arr = np.random.random(shape)

        for z in range(len(arr)):
            img = arr[z, :, :]
            tifffile.imwrite(input_dir.joinpath(f"img{sep}{z}.tiff"), data=img)

        # load data
        io = IO()
        arr_load = io.load(input_dir, sep=sep)

        assert arr.shape == arr_load.shape
        assert np.array_equal(arr, arr_load)


class TestMotionCorrection:

    @pytest.mark.parametrize("input_type", ["array", ".h5", ".tiff"])
    def test_random(self, input_type, shape=(100, 100, 100)):

        data = np.random.random(shape)
        loc = ""

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            io = IO()

            if input_type == ".h5":

                loc = "mc/ch0"
                temp_path = tmpdir.joinpath("test.h5")
                io.save(temp_path, data=data, loc=loc)

                data = temp_path

            elif input_type == ".tiff":

                temp_path = tmpdir.joinpath("test.tiff")
                io.save(temp_path, data={"ch0": data})

                data = temp_path

            elif input_type == ".tdb":

                temp_path = tmpdir.joinpath("test.tdb")
                temp_path = io.save(temp_path, data=data)

                assert temp_path.is_dir(), f"cannot find {temp_path}"
                data = temp_path

            elif input_type == "array":
                pass

            else:
                raise ValueError

            wd = tmpdir.joinpath("wd/")
            if not wd.exists():
                wd.mkdir()

            mc = MotionCorrection(working_directory=wd)
            mc.run(data, loc=loc, max_shifts=(6, 6))

            data = mc.save(output=None)
            assert type(data) == np.ndarray

    @pytest.mark.parametrize("input_type", [".tdb"])
    @pytest.mark.skip(reason="currently doesn't work. revisit later.")
    def test_random_tdb(self, input_type, shape=(100, 100, 100)):

        data = np.random.random(shape)
        loc = None

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            io = IO()

            if input_type == ".h5":

                loc = "mc/ch0"
                temp_path = tmpdir.joinpath("test.h5")
                io.save(temp_path, data={"test/ch0": data}, loc="mc")

                data = temp_path

            elif input_type == ".tiff":

                temp_path = tmpdir.joinpath("test.tiff")
                io.save(temp_path, data={"ch0": data})

                data = temp_path

            elif input_type == ".tdb":

                temp_path = tmpdir.joinpath("test.tdb")
                temp_path = io.save(temp_path, data={"ch0": data}, loc="")

                assert temp_path.is_file(), f"cannot find {temp_path}"
                data = temp_path

            elif input_type == "array":
                pass

            else:
                raise ValueError

            mc = MotionCorrection()
            mc.run(data, loc=loc, max_shifts=(6, 6))

            data = mc.save(output=None)
            assert type(data) == np.ndarray

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_real_input(self, extension, loc="dff/ch0"):

        si = SampleInput()
        input_ = si.get_test_data(extension=extension)

        mc = MotionCorrection()
        mc.run(path=input_, loc=loc, max_shifts=(6, 6))

        data = mc.save(output=None)
        assert type(data) == np.ndarray

    @pytest.mark.parametrize(
        "video_param", [{"speed": 0, "size": (100, 50, 50)}, {"speed": 0.1, "size": (100, 50, 50)},
                        {"speed": 0.01, "size": (1000, 250, 250)}, {"speed": 0.01, "size": (1000, 250, 250)}]
    )
    def test_motion_correct_performance(self, video_param, dtype=np.uint8):

        motion_speed = video_param["speed"]
        Z, X, Y = video_param["size"]

        # Generate random structure
        data = np.zeros((Z, X, Y), dtype=dtype)
        structure = np.random.randint(low=0, high=255, size=(X, Y), dtype=dtype)

        # Add motion to each frame
        for t in range(Z):
            shift = int(t * motion_speed)
            shifted_structure = np.roll(structure, shift, axis=(0, 1))
            data[t] = shifted_structure

        mc = MotionCorrection()
        mc.run(data, max_shifts=(int(X / 2) - 1, int(Y / 2) - 1))

        data = mc.save(output=None)
        assert isinstance(data, np.ndarray)

        # get average shift per frame
        mcs = np.array(mc.shifts)[:, 0]
        mcs = np.mean(np.abs(np.diff(mcs)))

        assert np.allclose(mcs, motion_speed, atol=1.5)
