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


class Test_Correlation:

    def setup_method(self):
        # Set up any necessary data for the tests
        self.correlation = Distance()
        self.corr_matrix = np.random.rand(100, 100)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", "pandas"])
    def test_get_correlation_matrix(self, input_type, ragged):

        dg = DummyGenerator(num_rows=25, trace_length=12, ragged=ragged)
        data = dg.get_by_name(input_type)

        c = self.correlation.get_pearson_correlation(events=data)

    def test_get_correlation_histogram(self, num_bins=1000):
        # Test with precomputed correlation matrix
        counts = self.correlation._get_correlation_histogram(corr=self.corr_matrix, num_bins=num_bins)
        assert np.equal(len(counts), num_bins)  # Adjust the expected value as per the number of bins

        # Test with events array
        counts = Distance()._get_correlation_histogram(events=self.corr_matrix, num_bins=num_bins)
        assert np.equal(len(counts), num_bins)  # Adjust the expected value as per the number of bins

        # Test with event dataframe
        dg = DummyGenerator()
        events = dg.get_dataframe()
        counts = Distance()._get_correlation_histogram(events=events, num_bins=num_bins)
        assert np.equal(len(counts), num_bins)  # Adjust the expected value as per the number of bins


    def test_plot_correlation_characteristics(self):
        # Test the plot_correlation_characteristics function

        # auto figure creation
        result = self.correlation.plot_correlation_characteristics(corr=self.corr_matrix,
                                                                   perc=[0.01, 0.05, 0.1],
                                                                   bin_num=20, log_y=True,
                                                                   figsize=(8, 4))
        assert isinstance(result, plt.Figure)

        # provided figure
        fig, axx = plt.subplots(1, 2)
        logging.warning(f"{axx}, {type(axx)}")
        result = self.correlation.plot_correlation_characteristics(corr=self.corr_matrix, ax=axx,
                                                                   perc=[0.01, 0.05, 0.1],
                                                                   bin_num=20, log_y=True,
                                                                   figsize=(8, 4))
        assert isinstance(result, plt.Figure)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", "pandas"])
    def test_plot_compare_correlated_events(self, ragged, input_type, num_rows=25, trace_length=20,
                                            corr_range = (0.1, 0.999)):

        dg = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = dg.get_by_name(input_type)

        # default arguments
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events)
        assert isinstance(fig, plt.Figure)

        # test style attributes
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    ev0_color="red", ev1_color="blue", ev_alpha=0.2, spine_linewidth=1)
        assert isinstance(fig, plt.Figure)

        # test
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    )
        assert isinstance(fig, plt.Figure)

        # test indices
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    event_index_range=(10, 15))
        assert isinstance(fig, plt.Figure)

        # test custom fig
        _, ax = plt.subplots(1, 1)
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    ax=ax, figsize=(10, 10), title="hello")
        assert isinstance(fig, plt.Figure)

        # test z_range
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    z_range=(2, 10))
        assert isinstance(fig, plt.Figure)

        # test z_range
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events,
                                    z_range=(2, 10))
        assert isinstance(fig, plt.Figure)

        # test corr filtering

        corr_min, corr_max = corr_range
        corr_matrix = np.random.random(size=(num_rows, num_rows))

        fig = self.correlation.plot_compare_correlated_events(corr_matrix, events,
                                    corr_range=(corr_min, corr_max))
        assert isinstance(fig, plt.Figure)

        corr_mask = np.where(np.logical_and(corr_matrix >= corr_min, corr_matrix <= corr_max))
        fig = self.correlation.plot_compare_correlated_events(corr_matrix, events,
                                    corr_mask=corr_mask)
        assert isinstance(fig, plt.Figure)

    def teardown_method(self):
        # Clean up after the tests, if necessary
        plt.close()
