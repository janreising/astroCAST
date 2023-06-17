import pytest

from astroCAST.helper import *

@pytest.mark.xfail
def test_local_caching_wrapper():
    raise NotImplementedError

@pytest.mark.xfail
def test_get_data_dimensions():
    raise NotImplementedError

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

            assert np.allclose(res[l], control)

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
    @pytest.mark.parametrize("min_length", [None, 5, 20])
    @pytest.mark.parametrize("max_length", [None, 5, 20])
    def test_enforced_length(num_rows, ragged, min_length, max_length):

        if (min_length is not None) and (max_length is not None) and (min_length != max_length):
            return None

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
        data = DG.get_array()

        norm = Normalization(data)
        res = norm.run({0: ["enforce_length", dict(min_length=min_length, max_length=max_length)]})

        for r in range(len(res)):
            row = res[r]

            if min_length is not None:
                assert len(row) >= min_length

            if max_length is not None:
                assert len(row) <= max_length

    @staticmethod
    @pytest.mark.parametrize("fixed_value", [None, 999])
    @pytest.mark.parametrize("enforced", [True, False])
    def test_impute_nan(num_rows, ragged, fixed_value, enforced):

        DG = DummyGenerator(num_rows=num_rows, trace_length=10, ragged=ragged)
        data = DG.get_array()

        if enforced:

            norm = Normalization(data)
            imputed = norm.run({
                0: ["enforce_length", dict(min_length=14, max_length=None)],
                1: ["impute_nan", dict(fixed_value=fixed_value)]
            })

        else:

            for r in range(len(data)):

                if len(data[r]) < 2:
                    pass

                row = data[r]
                rand_idx = np.random.randint(0, max(len(row), 1))
                row[rand_idx] = np.nan
                data[r] = row

            norm = Normalization(data)
            assert np.sum(np.isnan(norm.data if isinstance(norm.data, np.ndarray) else ak.ravel(norm.data))) > 0

            imputed = norm.run({
                0: ["impute_nan", dict(fixed_value=fixed_value)]
            })

        assert np.sum(np.isnan(imputed if isinstance(imputed, np.ndarray) else ak.ravel(imputed))) == 0

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
