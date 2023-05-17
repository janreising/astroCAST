import logging
import tempfile
from pathlib import Path

import dask
import numpy as np
import pytest
import tifffile

from astroCAST.preparation import *


class Test_Delta:

    @pytest.mark.parametrize("input_type", [np.ndarray, "testdata/sample_0.tiff", "testdata/sample_0.h5", "tiledb"])
    @pytest.mark.parametrize("in_memory", (True, False))
    @pytest.mark.parametrize("parallel", (True, False))
    def test_load(self, input_type, in_memory, parallel):

        Z, X, Y = 50, 10, 10

        if input_type == np.ndarray:
            data = np.random.randint(0, 100, (Z, X, Y), dtype=int)
            loc = None

        elif input_type == "tiledb":

            arr = da.from_array(
                x=np.random.randint(0, 100, (Z, X, Y), dtype=int),
                chunks=(Z, "auto", "auto")
            )

            loc = None
            tmpdir = Path(tempfile.mkdtemp()).joinpath("temp.tdb")
            logging.warning(f"tmpdir: {tmpdir}, {type(tmpdir)}")

            arr.to_tiledb(tmpdir.as_posix())
            data = tmpdir

        elif isinstance(input_type, str):

            path = Path(input_type)
            assert path.is_file()
            data = path

            loc = "data/ch0" if path.suffix == ".h5" else None

        else:
            raise TypeError

        delta = Delta(data, loc=loc, in_memory=in_memory, parallel=parallel)

        delta.run(method="background", window=5)

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("parallel", (True, False))
    @pytest.mark.parametrize("use_dask", (True, False))
    @pytest.mark.parametrize("in_memory", (True, False))
    def test_methods_run(self, method, parallel, use_dask, in_memory):

        Z, X, Y = 25, 2, 2

        data = np.random.randint(0, 100, (Z, X, Y), dtype=int)
        loc = None

        delta = Delta(data, loc=loc,
                      in_memory=in_memory, parallel=parallel)

        delta.run(method=method, window=5, use_dask=use_dask)

    @pytest.mark.parametrize("dim", [(100), (100, 5), (100, 5, 5), (100, 2, 10)])
    def test_background_dimensions(self, dim):

        arr = np.random.randint(0, 100, dim, dtype=int)
        res = Delta.calculate_delta_min_filter(arr, window=10)

        assert res.shape == dim, f"dimensions are not the same input: {dim} vs output: {res.shape}"

    @pytest.mark.parametrize("method", ("background", "dF", "dFF"))
    @pytest.mark.parametrize("parallel", (True, False))
    @pytest.mark.parametrize("use_dask", (True, False))
    @pytest.mark.parametrize("in_memory", (True, False))
    def test_result_for_parallel(self, method, parallel, use_dask, in_memory):

        dim = (250, 50, 50)
        window = 10

        arr = np.random.randint(0, 100, dim, dtype=int)

        ctrl = Delta.calculate_delta_min_filter(arr.copy(), window, method=method)
        logging.warning(f"sum of ctrl: {np.sum(ctrl)}")

        delta = Delta(arr, loc=None, in_memory=in_memory, parallel=parallel)
        res = delta.run(method=method, window=window, use_dask=use_dask, overwrite_first_frame=False)

        assert np.allclose(ctrl, res)

    def test_quality_of_dff(self):
        raise NotImplementedError

    def test_new_dFF_version(self):
        raise NotImplementedError

class Test_Input:

    @pytest.mark.parametrize("prefix", ["", "00000"])
    @pytest.mark.parametrize("sep", ["_", "x", "-"])
    def test_alphanumerical_names(self, prefix, sep):

        names = []
        for n in range(1000):
            name = f"img{sep}{prefix}{n}.ext"
            names.append(name)

        names_shuffled = np.random.shuffle(names.copy())

        assert names != names_shuffled, "randomization did not work"

        inp = Input()
        names_sorted = inp.sort_alpha_numerical_names(names, sep=sep)

        assert names == names_sorted, "sorting did not work"

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
            stack = inp.run(input_path=tmpdir, dtype=None, in_memory=in_memory)
            stack = stack["ch0"]

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
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(7):
                for n in range(num_channels):

                    img = np.random.random((1, 10, 10))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            # Loaded
            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels, dtype=None, in_memory=in_memory)

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
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(7):
                for n in range(num_channels):

                    img = np.random.random((1, X, Y))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            if num_channels == 1:
                subtract_background = np.random.random((X, Y))

            else:
                subtract_background = "ch0"

            # Loaded
            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels,
                                  subtract_background=subtract_background, subtract_func=subtract_func,
                                  dtype=None, in_memory=in_memory)

            # check result
            func_reduction = {"mean": np.mean, "std": np.std, "min": np.min, "max":np.max}

            if num_channels == 1:

                assert np.array_equal(
                    stack["ch0"],
                    images["ch0"] - subtract_background
                )

            else:

                func = func_reduction[subtract_func] if not callable(subtract_func) else subtract_func
                background = func(images["ch0"], axis=0)

                for ch in stack.keys():

                    if ch == "ch0":
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

            Z, X, Y = 20, 12, 12

            # Reference
            images = {f"ch{n}":[] for n in range(num_channels)}
            c=0
            for n in range(20):
                for n in range(num_channels):

                    img = np.random.random((1, X, Y))
                    images[f"ch{n}"].append(img)

                    tifffile.imwrite(tmpdir.joinpath(f"ss_single_{c}.tiff"), img)
                    c=c+1

            for k in images.keys():
                images[k] = np.squeeze(np.stack(images[k]))

            inp = Input()
            stack = inp.run(input_path=tmpdir, channels=num_channels, rescale=rescale,
                                  subtract_background=None, subtract_func=None,
                                  dtype=None, in_memory=True)

            res = stack["ch0"]

            if rescale == 1:
                assert res.shape == (Z, X, Y)

            elif isinstance(rescale, float):
                assert res.shape == (Z, int(X*rescale), int(Y*rescale))

            elif isinstance(rescale, int):
                assert res.shape == (Z, rescale, rescale)

            elif isinstance(rescale, tuple):
                rx, ry = rescale

                if isinstance(rx, int):
                    assert res.shape == (Z, rx, ry)

                elif isinstance(rx, float):

                    assert res.shape == (Z, int(X*rx), int(Y*ry))

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff"])
    @pytest.mark.parametrize("chunks", [None, (5, 5, 5)])
    def test_output(self, output_path, chunks):

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            output_path = tmpdir.joinpath(output_path)

            data = {0: np.random.random((25, 10, 10))}

            inp = Input()
            inp.save(output_path, data, chunks=chunks)

    @pytest.mark.parametrize("output_path", ["out.h5", "out.tdb", "out.tiff"])
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

            inp.run(input_path=tmpdir, output_path=output_path,
                            dtype=None, in_memory=False, prefix="data")

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

    @pytest.mark.xfail
    def test_convert_czi(self):
        # TODO I do not know if python can write czi files
        raise NotImplementedError

class Test_IO:

    def test_(self):
        raise NotImplementedError

class Test_MotionCorrection:

    @pytest.mark.parametrize("input_type", ["array", ".h5", ".tdb", ".tiff"])
    def test_random(self, input_type, shape=(100, 100, 100)):

        data = np.random.random(shape)
        h5_loc = None

        with tempfile.TemporaryDirectory() as dir:
            tmpdir = Path(dir)
            assert tmpdir.is_dir()

            io = IO()

            if input_type == ".h5":

                h5_loc = "mc/ch0"
                temp_path = tmpdir.joinpath("test.h5")
                io.save(temp_path, data={"ch0":data}, prefix="mc")

                data = temp_path

            elif input_type == ".tiff":
                # data = "testdata/sample_0.tiff"

                temp_path = tmpdir.joinpath("test.tiff")
                io.save(temp_path, data={"ch0":data})

                data = temp_path

            elif input_type == ".tdb":
                raise NotImplementedError("Error in IO.save() when saving .tdb")

                temp_path = tmpdir.joinpath("test.tdb")
                io.save(temp_path, data={"ch0":data})

                data = temp_path

            elif input_type == "array":
                pass

            else:
                raise ValueError

            mc = MotionCorrection()
            mc.run(data, h5_loc=h5_loc, max_shifts=(6, 6))

            data = mc.get_data(output=None, remove_mmap=True)
            assert type(data) == np.ndarray

    @pytest.mark.xfail
    def test_real_input(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_motion_correct_performance(self):
        raise NotImplementedError
