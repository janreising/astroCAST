import numpy as np
import pytest

from astroCAST.clustering import *

class Test_hdbscan:

    def test_clustering(self):

        data = np.random.random(size=(12, 25))

        hdb = HdbScan()
        _ = hdb.fit(data)

        data2 = np.random.random(size=(12, 25))
        _ = hdb.predict(data2)

    def test_load_save(self, tmp_path):

        data = np.random.random(size=(12, 25))

        hdb = HdbScan()
        _ = hdb.fit(data)
        hdb.save(tmp_path)

        data2 = np.random.random(size=(12, 25))
        hdb2 = HdbScan()
        hdb2.load(tmp_path)
        _ = hdb2.predict(data2)

