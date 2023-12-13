import logging
import tempfile
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from astrocast.helper import CachedClass, DummyGenerator, EventSim, Normalization, SampleInput, \
    download_pretrained_models, download_sample_data, get_data_dimensions, is_ragged, load_yaml_defaults, \
    wrapper_local_cache
from astrocast.preparation import IO


class TestLocalCache:
    
    @staticmethod
    @wrapper_local_cache
    def func_simple(cache_path=None):
        N = float(np.random.random(100000))
        time.sleep(2)
        return N
    
    @staticmethod
    @wrapper_local_cache
    def func_arg(add_fix, cache_path=None):
        N = float(np.random.random(100000)) + add_fix
        time.sleep(2)
        return N
    
    @staticmethod
    @wrapper_local_cache
    def func_kwarg(add_fix=2.0, cache_path=None):
        N = float(np.random.random(100000)) + add_fix
        time.sleep(2)
        return N
    
    def test_simple(self, tmpdir):
        
        temp_dir = Path(tmpdir.strpath)
        
        t0 = time.time()
        n1 = self.func_simple(cache_path=temp_dir)
        d1 = time.time() - t0
        
        t0 = time.time()
        n2 = self.func_simple(cache_path=temp_dir)
        d2 = time.time() - t0
        
        assert n1 == n2, f"cached result is incorrect: {n1} != {n2}"
        assert d2 < d1, f"cached result took too long: {d1} <= {d2}"
    
    @pytest.mark.parametrize("position", ["arg", "kwarg"])
    @pytest.mark.parametrize(
            "data",
            ["A", 2, 2.9, True, (1, 2), [1, 2], np.zeros(1), {"a": 1}, {"n": np.zeros(1)}, {"outer": {"inner": 3}},
             pd.DataFrame({"A": [1, 2]}), pd.Series([1, 2, 3]), np.array_equal]
            )
    def test_input_type(self, tmpdir, position, data):
        
        temp_dir = Path(tmpdir.strpath)
        
        if position == "arg":
            
            t0 = time.time()
            n1 = self.func_arg(data=data, cache_path=temp_dir)
            d1 = time.time() - t0
            
            t0 = time.time()
            n2 = self.func_arg(data=data, cache_path=temp_dir)
            d2 = time.time() - t0
        
        elif position == "kwarg":
            
            t0 = time.time()
            n1 = self.func_kwarg(data, cache_path=temp_dir)
            d1 = time.time() - t0
            
            t0 = time.time()
            n2 = self.func_kwarg(data, cache_path=temp_dir)
            d2 = time.time() - t0
        
        else:
            raise ValueError
        
        assert n1 == n2, f"cached result is incorrect: {n1} != {n2}"
        assert d2 < d1 or np.allclose(d2, d1), f"cached result took too long: {d1} <= {d2}"
    
    def test_custom_classes(self, tmpdir):
        
        temp_dir = Path(tmpdir.strpath)
        
        cc = CachedClass(cache_path=temp_dir)
        
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
class TestNormalization:
    
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
            
            a = res[r]
            if not isinstance(a, np.ndarray):
                a = a.to_numpy()
            a = a.astype(float)
            
            b = norm.data[r]
            if not isinstance(b, np.ndarray):
                b = b.to_numpy()
            b = b.astype(float)
            b = np.diff(b)
            b = np.concatenate([np.array([0]), b])
            
            assert a.shape == b.shape
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
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    def test_mean_std(data_type, num_rows, ragged):
        
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
        norm.mean_std()
    
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
        
        imputed = norm.run(
                {0: ["impute_nan", dict(fixed_value=fixed_value)]}
                )
        
        if isinstance(imputed, ak.Array):
            imputed = ak.ravel(imputed)
        
        assert np.sum(np.isnan(imputed)) == 0
    
    def test_column_wise(self, num_rows, ragged):
        
        if ragged == "ragged":
            
            with pytest.raises(ValueError):
                
                data = np.array([(np.random.random((np.random.randint(1, 10))) * 10).astype(int) for _ in range(3)])
                
                norm = Normalization(data)
                
                instr = {0: ["divide", {"mode": "max", "rows": False}], }
                res = norm.run(instr)
        
        else:
            
            data = np.random.random((num_rows, 3))
            data = data * 10
            data = data.astype(int)
            
            norm = Normalization(data)
            
            instr = {0: ["divide", {"mode": "max", "rows": False}], }
            res = norm.run(instr)
            assert np.max(res) <= 1
            
            # force 0s
            data[:, 2] -= np.max(data[:, 2])
            norm = Normalization(data)
            
            instr = {0: ["divide", {"mode": "max", "rows": False}], }
            res = norm.run(instr)
            np.allclose(data[:, 2], res[:, 2])
    
    @pytest.mark.parametrize("data_type", ["list", "dataframe", "array"])
    def test_not_inplace(self, data_type, num_rows, ragged):
        
        DG = DummyGenerator(num_rows=num_rows, ragged=False)
        
        if data_type == "list":
            data = DG.get_list()
        elif data_type == "dataframe":
            data = DG.get_dataframe().trace
        elif data_type == "array":
            data = DG.get_array()
        else:
            raise NotImplementedError
        
        norm = Normalization(data, inplace=False)
        res = norm.run({0: ["subtract", dict(mode="mean", population_wide=False)]})


class TestEventSim:
    
    def test_simulate_default_arguments(self):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    # Parameterized test for different shapes
    @pytest.mark.parametrize("shape", [[50, 100, 100], [100, 200, 200]])
    def test_simulate_different_shapes(self, shape):
        sim = EventSim()
        event_map, num_events = sim.simulate(shape)
        
        assert np.allclose(event_map.shape, shape)
        assert num_events >= 0
    
    # Parameterized test for different z_fraction and xy_fraction
    @pytest.mark.parametrize("z_fraction, xy_fraction", [(0.3, 0.15), (0.5, 0.25)])
    def test_simulate_different_fractions(self, z_fraction, xy_fraction):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, z_fraction=z_fraction, xy_fraction=xy_fraction)
        
        assert np.allclose(event_map.shape, shape)
        assert num_events >= 0
    
    # Testing with different gap_space and gap_time
    @pytest.mark.parametrize("gap_space, gap_time", [
        [1, 1],
        [2, 3],
        ])
    def test_simulate_different_gaps(self, gap_space, gap_time):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, gap_space=gap_space, gap_time=gap_time)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    # Testing different event_probability
    @pytest.mark.parametrize("event_probability", [0.1, 0.5, 0.9])
    def test_simulate_different_event_probability(self, event_probability):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, event_probability=event_probability)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    # Testing different event_intensity values
    @pytest.mark.parametrize("event_intensity", ["incr", 100, 200])
    def test_simulate_different_event_intensity(self, event_intensity):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, event_intensity=event_intensity)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    # Testing different background_noise values
    @pytest.mark.parametrize("background_noise", [None, 0.5, 1])
    def test_simulate_different_background_noise(self, background_noise):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, background_noise=background_noise)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    # Testing different blob_size_fraction values
    @pytest.mark.parametrize("blob_size_fraction", [0.05, 0.1, 0.2])
    def test_simulate_different_blob_size_fraction(self, blob_size_fraction):
        sim = EventSim()
        shape = (50, 100, 100)
        event_map, num_events = sim.simulate(shape, blob_size_fraction=blob_size_fraction)
        
        assert event_map.shape == shape
        assert num_events >= 0
    
    def test_simulate_common_setting(self, shape=(50, 100, 100)):
        sim = EventSim()
        event_map, num_events = sim.simulate(
                shape=shape, skip_n=5, gap_space=5, gap_time=3, event_intensity=100, background_noise=1
                )
        
        assert event_map.shape == shape
        assert num_events >= 0


class TestSampleInput:
    
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
    def test_loc(self, loc, extension=".h5"):
        si = SampleInput()
        input_path = si.get_test_data(extension=extension)
        
        loc = si.get_loc(ref=loc)
        assert isinstance(loc, str), f"loc is type: {type(loc)}"


def test_load_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        
        # Create a temporary YAML file for testing
        yaml_content = """
        param1: value1
        param2: value2
        """
        
        yaml_file = tmpdir.joinpath("test_config.yaml")
        with open(yaml_file.as_posix(), "w") as yaml_f:
            yaml_f.write(yaml_content)
        
        # Run the load_yaml_defaults function
        result = load_yaml_defaults(yaml_file.as_posix())
        
        # Check if the function read the values correctly
        assert result == {"param1": "value1", "param2": "value2"}


@pytest.mark.long
def test_sample_download():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        
        download_sample_data(tmpdir)
        
        assert len(list(tmpdir.glob("*/*"))) > 0


@pytest.mark.long
def test_model_download():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        
        download_pretrained_models(tmpdir)
        
        assert len(list(tmpdir.glob("*/*"))) > 0


@pytest.mark.parametrize("data_type", ["np.ndarray", ".h5", ".tiff", ".tdb", ".err"])
def test_data_dimensions(data_type):
    with tempfile.TemporaryDirectory() as tmp:
        
        tmpdir = Path(tmp)
        
        arr = np.random.random((10, 10, 10))
        loc = "data/ch0"
        
        if data_type == ".err":
            
            input_ = tmpdir.joinpath(f"file.{data_type}")
            with open(input_.as_posix(), "w") as w:
                w.write("hello")
            
            with pytest.raises(TypeError):
                _ = get_data_dimensions(input_, loc=loc, return_dtype=True)
        
        else:
            
            if data_type == "np.ndarray":
                input_ = arr
            
            else:
                input_ = tmpdir.joinpath(f"file.{data_type}")
                io = IO()
                io.save(input_, data=arr, loc=loc)
            
            _ = get_data_dimensions(input_, loc=loc, return_dtype=True)
