import logging
import tempfile
from pathlib import Path

import dask
import numpy as np
import pytest

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
    def test_convert_single_tiff_series(self, num_files):

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
            stack = inp.load_tiff(path=tmpdir, dtype=None, in_memory=True)
            stack = stack["ch0"]

            img_stack = np.squeeze(img_stack)
            stack = np.squeeze(stack)

            assert img_stack.shape == stack.shape
            assert np.array_equal(img_stack, stack)

    @pytest.mark.parametrize("num_channels", [2, 3])
    def test_convert_multi_channel(self, num_channels):

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
            stack = inp.load_tiff(path=tmpdir, channels=num_channels, dtype=None, in_memory=True)

            for ch in images.keys():

                ref = np.squeeze(images[ch])
                res = np.squeeze(stack[ch])

                assert ref.shape == res.shape
                assert np.array_equal(ref, res)

    def test_output(self):

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
            stack = inp.load_tiff(path=tmpdir, dtype=None, in_memory=False)

            output_path = tmpdir.joinpath("out.h5")
            inp.save(output_path, stack, prefix="data")

            # load back
            with h5py.File(output_path.as_posix(), "r") as f:
                res = f["data/ch0"][:]

            res = np.squeeze(res)

            assert img_stack.shape == res.shape
            assert np.array_equal(img_stack, res)

"""
    def test_convert_czi(self):
        raise NotImplementedError

    def test_compression(self):
        raise NotImplementedError

    def test_output(self):
        raise NotImplementedError

    def test_subtract(self):
        raise NotImplementedError

    def test_summary(self):
        raise NotImplementedError

"""
