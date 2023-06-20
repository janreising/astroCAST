import time

import numpy as np
import pytest

from astroCAST.clustering import *
from astroCAST.helper import DummyGenerator

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

# TODO test different inputs

class Test_dtw_linkage:

    @pytest.mark.parametrize("use_mmap", [True, False])
    def test_clustering(self, use_mmap):

        dtw = Linkage()
        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        dm = dtw.calculate_distance_matrix(events=data, use_mmap=use_mmap)
        Z = dtw.calculate_linkage_matrix(dm)
        clusters, cluster_labels = dtw.cluster_linkage_matrix(Z, z_threshold=3)

        barycenters = dtw.calculate_barycenters(clusters, cluster_labels, events=data)

    def test_wrapper_function(self):
        dtw = Linkage()
        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        barycenters = dtw.get_barycenters(data, z_threshold=2)

    @pytest.mark.parametrize("z_threshold", [None, 2])
    @pytest.mark.parametrize("min_cluster_size", [None, 10])
    def test_plotting(self, z_threshold, min_cluster_size, tmp_path):

        dtw = Linkage()
        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        dm = dtw.calculate_distance_matrix(data, use_mmap=False)
        Z = dtw.calculate_linkage_matrix(dm)

        # test custom values
        dtw.plot_cluster_fraction_of_retention(Z, z_threshold=z_threshold, min_cluster_size=min_cluster_size)

        # test provided axis
        fig, ax = plt.subplots(1, 1)
        dtw.plot_cluster_fraction_of_retention(Z, ax=ax)

        # test saving
        dtw.plot_cluster_fraction_of_retention(Z, ax=ax, save_path=tmp_path)

    @pytest.mark.parametrize("use_mmap", [True, False])
    def test_local_cache(self, use_mmap):

        with tempfile.TemporaryDirectory() as dir:
            tmp_path = Path(dir)
            assert tmp_path.is_dir()

            DG = DummyGenerator(num_rows=25, trace_length=16, ragged=False)
            data = DG.get_events()

            # test calculate **distance** matrix
            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            dtw.calculate_distance_matrix(data, use_mmap=use_mmap)
            dt = time.time() - t0
            del dtw

            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            dm = dtw.calculate_distance_matrix(data, use_mmap=use_mmap)
            dt2 = time.time() - t0

            assert dt2 < dt, "distance matrix is not cached"

            # test calculate **linkage** matrix
            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            dtw.calculate_linkage_matrix(dm)
            dt = time.time() - t0
            del dtw

            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            dtw.calculate_linkage_matrix(dm)
            dt2 = time.time() - t0

            assert dt2 < dt, "linkage matrix is not cached"
