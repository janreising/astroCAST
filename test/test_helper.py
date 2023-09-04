import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from astroCAST.helper import *

class Test_LocalCache:

    def setup_method(self):

        self.dir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.dir.name)
        assert self.tmpdir.is_dir(), f"Creation of {self.tmpdir} failed."

    def teardown_method(self):
        self.dir.cleanup()
        assert not self.tmpdir.is_dir(), f"Teardown of {self.tmpdir} failed."

    def test_simple(self):

        @wrapper_local_cache
        def func(cache_path=None):
            N = float(np.random.random(1))
            time.sleep(2)
            return N

        t0 = time.time()
        n1 = func(cache_path=self.tmpdir)
        d1 = time.time() - t0

        t0 = time.time()
        n2 = func(cache_path=self.tmpdir)
        d2 = time.time() - t0

        assert n1 == n2, f"cached result is incorrect: {n1} != {n2}"
        assert d2 < d1, f"cached result took too long: {d1} <= {d2}"

    @pytest.mark.parametrize("position", ["arg", "kwarg"])
    @pytest.mark.parametrize("data", ["A", 2, 2.9, True, (1, 2), [1, 2], np.zeros(1),
                                      {"a":1}, {"n":np.zeros(1)}, {"outer":{"inner":3}},
                                      pd.DataFrame({"A":[1, 2]}), pd.Series([1, 2, 3]),
                                      np.array_equal])
    def test_input_type(self, position, data):

        if position == "arg":

            @wrapper_local_cache
            def func(data, cache_path=None):
                N = float(np.random.random(1))
                time.sleep(0.2)
                return N

            t0 = time.time()
            n1 = func(data, cache_path=self.tmpdir)
            d1 = time.time() - t0

            t0 = time.time()
            n2 = func(data, cache_path=self.tmpdir)
            d2 = time.time() - t0

        elif position == "kwarg":

            @wrapper_local_cache
            def func(data=data, cache_path=None):
                N = float(np.random.random(1))
                time.sleep(1)
                return N

            t0 = time.time()
            n1 = func(data=data, cache_path=self.tmpdir)
            d1 = time.time() - t0

            t0 = time.time()
            n2 = func(data=data, cache_path=self.tmpdir)
            d2 = time.time() - t0

        else:
            raise ValueError

        assert n1 == n2, f"cached result is incorrect: {n1} != {n2}"
        assert d2 < d1, f"cached result took too long: {d1} <= {d2}"

    def test_CustomClasses(self):

        cc = CachedClass(cache_path=self.tmpdir)

        t0 = time.time()
        n1 = cc.print_cache_path()
        d1 = time.time() - t0

        t0 = time.time()
        n2 = cc.print_cache_path()
        d2 = time.time() - t0

        assert n1 == n2, f"cached result is incorrect: {n1} != {n2}"
        assert d2 < d1, f"cached result took too long: {d1} <= {d2}"

@pytest.mark.parametrize("typ", ["pandas", "list", "numpy", "dask", "events"])
@pytest.mark.parametrize("ragged", ["equal", "ragged"])
@pytest.mark.parametrize("num_rows", [1, 10])
def test_dummy_generator(num_rows, typ, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
        data = DG.get_by_name(typ)

        assert len(data) == num_rows

        # TODO needed?
        # if typ in ["numpy", "dask"]:
        #     assert data.shape[0] == num_rows

@pytest.mark.parametrize("typ", ["pandas", "list", "numpy", "dask"])
@pytest.mark.parametrize("ragged", ["equal", "ragged"])
def test_is_ragged(typ, ragged, num_rows=10):
    DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
    data = DG.get_by_name(typ)

    if typ == "pandas":
        data = data.trace

    logging.warning(f"type: {type(data)}")

    ragged = True if ragged == "ragged" else False
    assert is_ragged(data) == ragged

@pytest.mark.parametrize("num_rows", [1, 10])
@pytest.mark.parametrize("ragged", ["equal", "ragged"])
class Test_normalization:

    @staticmethod
    @pytest.mark.parametrize("population_wide", [True, False])
    @pytest.mark.parametrize("value_mode", ["first", "mean", "min", "min_abs", "max", "max_abs", "std"])
    def test_div_zero(num_rows, ragged, value_mode, population_wide):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)

        data = DG.get_array()
        data[0][0] = 0

        norm = Normalization(data)
        norm.run({0: ["divide", dict(mode=value_mode, population_wide=population_wide)]})

    @staticmethod
    @pytest.mark.parametrize("value_mode", ["first", "mean", "min", "min_abs", "max", "max_abs", "std"])
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    @pytest.mark.parametrize("population_wide", [True, False])
    def test_subtract(data_type, num_rows, ragged, value_mode, population_wide):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)

        if data_type == "list":
            data = DG.get_list()
        elif data_type == "dataframe":
            data = DG.get_dataframe().trace
        elif data_type == "array":
            data = DG.get_array()
        else:
            raise NotImplementedError

        norm = Normalization(data)
        res = norm.run({0: ["subtract", dict(mode=value_mode, population_wide=population_wide)]})

        for l in [0, -1]:

            if not population_wide:

                if value_mode == "first":
                    control = norm.data[l] - norm.data[l][0]
                if value_mode == "mean":
                    control = norm.data[l] - np.mean(norm.data[l])
                if value_mode == "min":
                    control = norm.data[l] - np.min(norm.data[l])
                if value_mode == "min_abs":
                    control = norm.data[l] - np.min(np.abs(norm.data[l]))
                if value_mode == "max":
                    control = norm.data[l] - np.max(norm.data[l])
                if value_mode == "max_abs":
                    control = norm.data[l] - np.max(np.abs(norm.data[l]))
                if value_mode == "std":
                    control = norm.data[l] - np.std(norm.data[l])

            else:

                if value_mode == "first":
                    control = norm.data[l] - np.mean([data[i][0] for i in range(len(data))])
                if value_mode == "mean":
                    control = norm.data[l] - np.mean(norm.data)
                if value_mode == "min":
                    control = norm.data[l] - np.min(norm.data)
                if value_mode == "min_abs":
                    control = norm.data[l] - np.min(np.abs(norm.data))
                if value_mode == "max":
                    control = norm.data[l] - np.max(norm.data)
                if value_mode == "max_abs":
                    control = norm.data[l] - np.max(np.abs(norm.data))
                if value_mode == "std":
                    control = norm.data[l] - np.std(norm.data)

            assert np.allclose(res[l], control)

    @staticmethod
    @pytest.mark.parametrize("value_mode", ["first", "mean", "min", "min_abs", "max", "max_abs", "std"])
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    @pytest.mark.parametrize("population_wide", [True, False])
    def test_divide(data_type, num_rows, ragged, value_mode, population_wide):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged, offset=50)

        if data_type == "list":
            data = DG.get_list()
        elif data_type == "dataframe":
            data = DG.get_dataframe().trace
        elif data_type == "array":
            data = DG.get_array()
        else:
            raise NotImplementedError

        norm = Normalization(data)
        res = norm.run({0: ["divide", dict(mode=value_mode, population_wide=population_wide)]})


        for l in [0, -1]:

            if not population_wide:

                if value_mode == "first":
                    control = norm.data[l] / norm.data[l][0]
                if value_mode == "mean":
                    control = norm.data[l] / np.mean(norm.data[l])
                if value_mode == "min":
                    control = norm.data[l] / np.min(norm.data[l])
                if value_mode == "min_abs":
                    control = norm.data[l] / np.min(np.abs(norm.data[l]))
                if value_mode == "max":
                    control = norm.data[l] / np.max(norm.data[l])
                if value_mode == "max_abs":
                    control = norm.data[l] / np.max(np.abs(norm.data[l]))
                if value_mode == "std":
                    control = norm.data[l] / np.std(norm.data[l])

            else:

                if value_mode == "first":
                    control = norm.data[l] / np.mean([data[i][0] for i in range(len(data))])
                if value_mode == "mean":
                    control = norm.data[l] / np.mean(norm.data)
                if value_mode == "min":
                    control = norm.data[l] / np.min(norm.data)
                if value_mode == "min_abs":
                    control = norm.data[l] / np.min(np.abs(norm.data))
                if value_mode == "max":
                    control = norm.data[l] / np.max(norm.data)
                if value_mode == "max_abs":
                    control = norm.data[l] / np.max(np.abs(norm.data))
                if value_mode == "std":
                    control = norm.data[l] / np.std(norm.data)

            assert np.allclose(res[l], control), f"res: {res[l]} \n {control}"

    @staticmethod
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    def test_diff(data_type, num_rows, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)

        if data_type == "list":
            data = DG.get_list()
        elif data_type == "dataframe":
            data = DG.get_dataframe().trace
        elif data_type == "array":
            data = DG.get_array()
        else:
            raise NotImplementedError

        norm = Normalization(data)
        res = norm.run({0: "diff"})

        for r in range(len(data)):
            a = res[r].astype(float) if isinstance(res[r], np.ndarray) else res[r].to_numpy().astype(float)
            b = np.diff(norm.data[r].astype(float)) if isinstance(norm.data[r], np.ndarray) else np.diff(norm.data[r].to_numpy().astype(float))
            assert np.allclose(a, b)

    @staticmethod
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    def test_min_max(data_type, num_rows, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)

        if data_type == "list":
            data = DG.get_list()
        elif data_type == "dataframe":
            data = DG.get_dataframe().trace
        elif data_type == "array":
            data = DG.get_array()
        else:
            raise NotImplementedError

        norm = Normalization(data)
        norm.min_max()

    @staticmethod
    @pytest.mark.parametrize("fixed_value", [None, 999])
    @pytest.mark.parametrize("nan_value", [np.nan])
    def test_impute_nan(num_rows, nan_value, ragged, fixed_value):

        DG = DummyGenerator(num_rows=num_rows, trace_length=25, ragged=ragged)
        data = DG.get_array()

        for r in range(len(data)):

            if len(data[r]) < 2:
                pass

            row = data[r]
            rand_idx = np.random.randint(0, max(len(row), 1))
            row[rand_idx] = nan_value
            data[r] = row

        norm = Normalization(data)
        assert np.sum(np.isnan(norm.data if isinstance(norm.data, np.ndarray) else ak.ravel(norm.data))) > 0

        imputed = norm.run({
            0: ["impute_nan", dict(fixed_value=fixed_value)]
        })

        if isinstance(imputed, ak.Array):
            imputed = ak.ravel(imputed)

        assert np.sum(np.isnan(imputed)) == 0

    def test_column_wise(self, num_rows, ragged):

        if ragged == "ragged":

            with pytest.raises(ValueError):

                data = np.array([(np.random.random((np.random.randint(1, 10)))*10).astype(int) for _ in range(3)])

                norm = Normalization(data)

                instr = {0: ["divide", {"mode": "max", "rows":False}],}
                res = norm.run(instr)

        else:

            data = np.random.random((num_rows, 3))
            data = data * 10
            data = data.astype(int)

            norm = Normalization(data)

            instr = {0: ["divide", {"mode": "max", "rows":False}],}
            res = norm.run(instr)
            assert np.max(res) <= 1

            # force 0s
            data[:, 2] -= np.max(data[:, 2])
            norm = Normalization(data)

            instr = {0: ["divide", {"mode": "max", "rows":False}],}
            res = norm.run(instr)
            np.allclose(data[:, 2], res[:, 2])

class Test_EventSim:
    def test_simulate_default_arguments(self):
        sim = EventSim()

        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape)

        assert event_map.shape == shape
        assert num_events >= 0

    def test_simulate_custom_arguments(self):
        sim = EventSim()

        shape = (25, 50, 50)
        z_fraction = 0.3
        xy_fraction = 0.15
        gap_space = 2
        gap_time = 2
        blob_size_fraction = 0.1
        event_probability = 0.5

        event_map, num_events = sim.simulate(shape, z_fraction, xy_fraction, gap_space, gap_time,
                                             blob_size_fraction, event_probability)

        assert event_map.shape == shape
        assert num_events >= 0

class Test_SampleInput:

    def test_load_and_delete(self):
        si = SampleInput()

        temp_dir = si.get_dir()
        assert temp_dir.is_dir()

        del si
        assert not temp_dir.is_dir()

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_load_file(self, extension):
        si = SampleInput()
        input_path = si.get_test_data(extension=extension)
        assert input_path.exists()

        del si
        assert not input_path.exists()

    @pytest.mark.parametrize("loc", [None, "dff/ch0"])
    def test_h5_loc(self, loc, extension=".h5"):

        si = SampleInput()
        input_path = si.get_test_data(extension=extension)

        h5_loc = si.get_h5_loc(ref=loc)
        assert isinstance(h5_loc, str), f"h5_loc is type: {type(h5_loc)}"
