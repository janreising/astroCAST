import pytest

from astroCAST.helper import *

def test_local_caching_wrapper():
    raise NotImplementedError

@pytest.mark.parametrize("typ", ["dataframe", "list", "array"])
@pytest.mark.parametrize("ragged", [True, False])
def test_dummy_generator(typ, ragged):

        num_rows = 5
        DG = DummyGenerator(num_rows=5, ragged=ragged)

        if typ == "dataframe":
            data = DG.get_dataframe()
            assert len(data) == num_rows

        elif typ == "list":
            data = DG.get_list()
            assert len(data) == num_rows

        elif typ == "array":
            data = DG.get_array()
            assert data.shape[0] == num_rows

@pytest.mark.parametrize("approach", ["min_max", "sub0_max", "standardize"])
@pytest.mark.parametrize("num_rows", [1, 10])
@pytest.mark.parametrize("ragged", [False, True])
class Test_Normalization:

    def test_list(self, num_rows, approach, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
        data = DG.get_list()

        norm = Normalization(data, approach)
        norm.run()

    def test_dataframe(self, num_rows, approach, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
        data = DG.get_dataframe()

        norm = Normalization(data.trace, approach)
        norm.run()

    def test_array(self, num_rows, approach, ragged):

        DG = DummyGenerator(num_rows=num_rows, ragged=ragged)
        data = DG.get_array()

        norm = Normalization(data, approach)
        norm.run()

class Test_Padding:

    def test_0(self):
        raise NotImplementedError
