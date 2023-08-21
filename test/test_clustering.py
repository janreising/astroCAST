import time

import numpy as np
import pytest

from astroCAST.clustering import *
from astroCAST.helper import DummyGenerator

class Test_hdbscan:

    def test_plain(self):

        dg = DummyGenerator(ragged=False)
        arr = dg.get_array()

        hdb = HdbScan()
        lbls = hdb.fit(arr)
        lbls2, strengths = hdb.predict(arr)

        assert isinstance(lbls, np.ndarray), f"lbls is type {type(lbls)} instead of 'np.ndarray'"
        assert isinstance(lbls2, np.ndarray), f"lbls_predicted is type {type(lbls2)} instead of 'np.ndarray'"
        assert isinstance(strengths, np.ndarray), f"strengths is type {type(strengths)} instead of 'np.ndarray'"

    def test_events(self):

        dg = DummyGenerator(ragged=False)
        events = dg.get_events()
        arr = dg.get_array()

        hdb = HdbScan(events=events)

        lut = hdb.fit(arr)
        assert isinstance(lut, dict), f"lut should be dictionary instead of {type(lut)}"

        lut2, strength = hdb.predict(arr)

        events.add_clustering(lut, "cluster_1")
        events.add_clustering(lut2, "cluster_2")
        events.add_clustering(strength, "strength")

    def test_load_save(self):

        with tempfile.TemporaryDirectory() as dir:
            tmp_path = Path(dir)
            assert tmp_path.is_dir()

            dg = DummyGenerator(ragged=False)
            arr = dg.get_array()

            # Fit
            hdb = HdbScan()
            _ = hdb.fit(arr)
            lbls_1 = hdb.predict(arr)

            hdb.save(tmp_path.joinpath("hdb.p"))

            # Load
            hdb_loaded = HdbScan()
            hdb_loaded.load(tmp_path.joinpath("hdb.p"))
            lbls_2 = hdb_loaded.predict(arr)

            assert np.allclose(lbls_1, lbls_2)

class Test_dtw_linkage:

    @pytest.mark.parametrize("distance_type", ["pearson", "dtw"])
    @pytest.mark.parametrize("criterion", ["distance", "maxclust"])
    def test_clustering(self, distance_type, criterion, cutoff=2):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        dtw = Linkage()
        _, _ = dtw.get_barycenters(data, cutoff, distance_type=distance_type, criterion=criterion)

    @pytest.mark.parametrize("cutoff", [2, 6])
    @pytest.mark.parametrize("min_cluster_size", [1, 10])
    @pytest.mark.parametrize("distance_type", ["pearson", "dtw"])
    def test_plotting(self, distance_type, cutoff, min_cluster_size, tmp_path):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(data, correlation_type=distance_type)

        dtw = Linkage()
        linkage_matrix = dtw.calculate_linkage_matrix(distance_matrix)

        # test custom values
        dtw.plot_cluster_fraction_of_retention(linkage_matrix,
                                               cutoff=cutoff, min_cluster_size=min_cluster_size)

        # test provided axis
        fig, ax = plt.subplots(1, 1)
        dtw.plot_cluster_fraction_of_retention(linkage_matrix, cutoff=cutoff, ax=ax)

        # test saving
        dtw.plot_cluster_fraction_of_retention(linkage_matrix, cutoff=cutoff, ax=ax, save_path=tmp_path)

    def test_local_cache(self):

        with tempfile.TemporaryDirectory() as dir:
            tmp_path = Path(dir)
            assert tmp_path.is_dir()

            DG = DummyGenerator(num_rows=25, trace_length=16, ragged=False)
            data = DG.get_events()

            # test calculate **distance** matrix
            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            dtw.get_barycenters(data, cutoff=2)
            dt = time.time() - t0
            del dtw

            dtw = Linkage(cache_path=tmp_path)
            t0 = time.time()
            _, _ = dtw.get_barycenters(data, cutoff=2)
            dt2 = time.time() - t0

            assert dt2 < dt, "results are not cached"
