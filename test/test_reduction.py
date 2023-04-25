import pytest

from astroCAST.reduction import *
from astroCAST.helper import DummyGenerator

DG_equal = DummyGenerator()
DG_ragged = DummyGenerator(ragged=True)

class Test_FeatureExtraction:

    @pytest.mark.parametrize("typ", ["dataframe", "list", "array"])
    @pytest.mark.parametrize("ragged", [True, False])
    def test_input_type(self, typ, ragged):

        DG = DG_ragged if ragged else DG_equal

        if typ == "dataframe":
            data = DG.get_dataframe()

        elif typ == "list":
            data = DG.get_list()

        elif typ == "array":
            data = DG.get_array()

        else:
            raise TypeError

        FE = FeatureExtraction()
        FE.get_features(data=data)

    @pytest.mark.parametrize("normalize", [None, "min_max"])
    def test_normalization(self, normalize):

        data = DG_ragged.get_dataframe()

        FE = FeatureExtraction()
        FE.get_features(data=data, normalize=normalize)
    @pytest.mark.parametrize("padding", [None, "edge"])
    def test_padding(self, padding):

        data = DG_ragged.get_dataframe()

        FE = FeatureExtraction()
        FE.get_features(data=data, normalize=padding)

# TODO test local caching
