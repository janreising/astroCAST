import numpy as np
import pandas as pd
import pytest

from astroCAST.reduction import *
from astroCAST.helper import DummyGenerator

DG_equal = DummyGenerator()
DG_ragged = DummyGenerator(ragged=True)

class Test_FeatureExtraction:

    def test_one(self, typ, ragged):

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
        features = FE.get_features(data=data)

# TODO test local caching
