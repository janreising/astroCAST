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

class Test_Normalization:

    def test_min_max(self):

        norm = Normalization()

        for _ in range(10):
            norm.min_max(np.random.random(size=(10)))

    def test_start_max(self):

        norm = Normalization()

        for _ in range(10):
            norm.start_max(np.random.random(size=(10)))
