import logging
import tempfile
from pathlib import Path

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


