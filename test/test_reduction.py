import numpy as np
import pandas as pd
import pytest

from astroCAST.reduction import *

class DummyGenerator():

    def __init__(self, num_rows=25, trace_length=12, ragged=False):

        self.data = self.get_data(num_rows=num_rows, trace_length=trace_length, ragged=ragged)

    def get_data(self, num_rows, trace_length, ragged):

        if ragged:
            data = [np.random.random(size=trace_length+np.random.randint(low=-trace_length+1, high=trace_length-1)) for _ in range(num_rows)]
        else:
            data = np.random.random(size=(num_rows, trace_length))

        return data

    def get_dataframe(self):

        data = self.data

        if type(data) == list:
            df = pd.DataFrame(dict(trace=data))

        elif type(data) == np.ndarray:
            df = pd.DataFrame(dict(trace=data.tolist()))
        else:
            raise TypeError

        # create dz, z0 and z1
        df["dz"] = df.trace.apply(lambda x: len(x))

        dz_sum = int(df.dz.sum()/2)
        df["z0"] = [np.random.randint(low=0, high=dz_sum) for _ in range(len(df))]
        df["z1"] = df.z0 + df.dz

        # create fake index
        df["idx"] = df.index

        return df

    def get_list(self):

        data = self.data

        if type(data) == list:
            return data

        elif type(data) == np.ndarray:
            return data.tolist()

        else:
            raise TypeError

    def get_array(self):

        data = self.data

        if type(data) == list:
            return np.array(data, dtype='object')

        elif type(data) == np.ndarray:
            return data

        else:
            raise TypeError

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
