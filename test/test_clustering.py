import platform
import time

import pytest

from astrocast.clustering import *
from astrocast.helper import DummyGenerator


class TestHdbscan:

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


class TestDtwLinkage:

    @pytest.mark.parametrize("criterion", ["distance", "maxclust"])
    @pytest.mark.parametrize("ragged", [True, False])
    def test_clustering_pearson(self, criterion, ragged):
        pytest.importorskip("dtwParallel")

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=ragged)
        data = DG.get_events()

        linkage = Linkage()
        _, _ = linkage.get_barycenters(
            data, cutoff=2, distance_type="pearson", criterion=criterion
        )

    @pytest.mark.skipif(platform.system() == 'Darwin', reason="Skip DTW test on MacOS")
    @pytest.mark.parametrize("criterion", ["distance", "maxclust"])
    def test_clustering_dtw(self, criterion):
        pytest.importorskip("dtwParallel")

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        linkage = Linkage()
        _, _ = linkage.get_barycenters(data, cutoff=2, distance_type="dtw", criterion=criterion)

    @pytest.mark.parametrize("criterion", ["distance", "maxclust"])
    @pytest.mark.parametrize("ragged", [True, False])
    def test_clustering_dtw_parallel(self, criterion, ragged):
        pytest.importorskip("dtwParallel")
        pytest.importorskip("dtaidistance")

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=ragged)
        data = DG.get_events()

        linkage = Linkage()
        _, _ = linkage.get_barycenters(data, cutoff=2, distance_type="dtw_parallel", criterion=criterion)

    @staticmethod
    def _test_plotting_helper(distance_type, cutoff, min_cluster_size, tmp_path):
        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(data, correlation_type=distance_type)
        logging.warning(f"len(distance_matrix): {len(distance_matrix)}")
        logging.warning(f"distance_matrix: {distance_matrix.shape}")

        dtw = Linkage()
        linkage_matrix = dtw.calculate_linkage_matrix(distance_matrix)

        # test custom values
        dtw.plot_cluster_fraction_of_retention(
            linkage_matrix, cutoff=cutoff, min_cluster_size=min_cluster_size
        )

        # test provided axis
        fig, ax = plt.subplots(1, 1)
        dtw.plot_cluster_fraction_of_retention(linkage_matrix, cutoff=cutoff, ax=ax)

        # test saving
        dtw.plot_cluster_fraction_of_retention(linkage_matrix, cutoff=cutoff, ax=ax, save_path=tmp_path)

    @pytest.mark.parametrize("cutoff", [2, 6])
    @pytest.mark.parametrize("min_cluster_size", [1, 10])
    def test_plotting_pearson(self, cutoff, min_cluster_size, tmp_path):
        self._test_plotting_helper("pearson", cutoff, min_cluster_size, tmp_path)

    @pytest.mark.skipif(platform.system() == 'Darwin', reason="Skip DTW test on MacOS")
    def test_plotting_dtw(self, tmp_path, cutoff=2, min_cluster_size=10):
        self._test_plotting_helper("dtw", cutoff, min_cluster_size, tmp_path)

    def test_plotting_dtw_parallel(self, tmp_path, cutoff=2, min_cluster_size=10):
        pytest.importorskip("dtwParallel")

        self._test_plotting_helper("dtw_parallel", cutoff, min_cluster_size, tmp_path)

    def test_local_cache(self):
        pytest.importorskip("dtwParallel")

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


class TestDistance:

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize(
        "local_dissimilarity",
        ["square_euclidean_distance", "gower", "norm1", "norm2", "braycurtis", "canberra", "chebyshev", "cityblock",
         "correlation", "cosine", "euclidean", "jensenshannon", "minkowski", "sqeuclidean"]
    )
    def test_dtw_parallel_local_dissimilarity(self, local_dissimilarity, ragged, num_rows=4, trace_length=8):
        pytest.importorskip("dtwParallel")

        params = dict(local_dissimilarity=local_dissimilarity)

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(events, correlation_type="dtw_parallel", correlation_param=params)

        assert distance_matrix.shape == (num_rows, num_rows)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("itakura_max_slope", [10, 20])
    def test_dtw_parallel_itakura(self, itakura_max_slope, ragged, num_rows=16, trace_length=32):
        pytest.importorskip("dtwParallel")

        params = dict(
            constrained_path_search="itakura", itakura_max_slope=itakura_max_slope,
            local_dissimilarity="square_euclidean_distance"
        )

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged, min_length=16)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(events, correlation_type="dtw_parallel", correlation_param=params)

        assert distance_matrix.shape == (num_rows, num_rows)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("sakoe_chiba_radius", [10, 20])
    def test_dtw_parallel_sakoe_chiba(self, sakoe_chiba_radius, ragged, num_rows=4, trace_length=8):
        pytest.importorskip("dtwParallel")

        params = {"constrained_path_search": "sakoe_chiba", "sakoe_chiba_radius": sakoe_chiba_radius}

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(events, correlation_type="dtw_parallel", correlation_param=params)

        assert distance_matrix.shape == (num_rows, num_rows)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("sigma_kernel", [1, 2])
    @pytest.mark.parametrize("dtw_to_kernel", [False, True])
    def test_dtw_parallel_kernel(self, sigma_kernel, dtw_to_kernel, ragged, num_rows=4, trace_length=8):
        pytest.importorskip("dtwParallel")

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(
            events, correlation_type="dtw_parallel", correlation_param=dict(
                sigma_kernel=sigma_kernel, dtw_to_kernel=dtw_to_kernel
            )
        )

        if dtw_to_kernel:
            assert isinstance(distance_matrix, tuple)

            distance_matrix, similarity_matrix = distance_matrix
            assert distance_matrix.shape == (num_rows, num_rows)
            assert similarity_matrix.shape == (num_rows, num_rows)
            assert np.max(similarity_matrix) <= 1
            assert np.min(similarity_matrix) >= -1

        else:
            assert isinstance(distance_matrix, np.ndarray)
            assert distance_matrix.shape == (num_rows, num_rows)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("type_dtw", ["d"])
    def test_dtw_parallel_type_depended(self, type_dtw, ragged, num_rows=4, trace_length=8):
        pytest.importorskip("dtwParallel")

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(
            events, correlation_type="dtw_parallel", correlation_param=dict(type_dtw=type_dtw)
        )

        assert distance_matrix.shape == (num_rows, num_rows)

    # TODO figure out why this is failing
    @pytest.mark.skip(reason="Failing unexpectedly")
    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("type_dtw", ["i"])
    def test_dtw_parallel_type_independent(self, type_dtw, ragged, num_rows=4, trace_length=8):
        pytest.importorskip("dtwParallel")

        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = DG.get_events()

        corr = Distance()
        distance_matrix = corr.get_correlation(
            events, correlation_type="dtw_parallel", correlation_param=dict(type_dtw=type_dtw)
        )

        assert distance_matrix.shape == (num_rows, num_rows)


class TestCorrelation:
    correlation = None
    corr_matrix = None

    @classmethod
    def setup_class(cls):
        # Set up any necessary data for the tests
        cls.correlation = Distance()
        cls.corr_matrix = np.random.rand(100, 100)

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
        result = self.correlation.plot_correlation_characteristics(
            corr=self.corr_matrix, perc=[0.01, 0.05, 0.1], bin_num=20, log_y=True, figsize=(8, 4)
        )
        assert isinstance(result, plt.Figure)

        # provided figure
        fig, axx = plt.subplots(1, 2)
        logging.warning(f"{axx}, {type(axx)}")
        result = self.correlation.plot_correlation_characteristics(
            corr=self.corr_matrix, ax=axx, perc=[0.01, 0.05, 0.1], bin_num=20, log_y=True, figsize=(8, 4)
        )
        assert isinstance(result, plt.Figure)

    @pytest.mark.parametrize("ragged", [True, False])
    @pytest.mark.parametrize("input_type", ["numpy", "dask", "pandas"])
    def test_plot_compare_correlated_events(
            self, ragged, input_type, num_rows=25, trace_length=20, corr_range=(0.1, 0.999)
    ):
        dg = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=ragged)
        events = dg.get_by_name(input_type)

        # default arguments
        fig = self.correlation.plot_compare_correlated_events(self.corr_matrix, events)
        assert isinstance(fig, plt.Figure)

        # test style attributes
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, ev0_color="red", ev1_color="blue", ev_alpha=0.2, spine_linewidth=1
        )
        assert isinstance(fig, plt.Figure)

        # test
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, )
        assert isinstance(fig, plt.Figure)

        # test indices
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, event_index_range=(10, 15)
        )
        assert isinstance(fig, plt.Figure)

        # test custom fig
        _, ax = plt.subplots(1, 1)
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, ax=ax, figsize=(10, 10), title="hello"
        )
        assert isinstance(fig, plt.Figure)

        # test z_range
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, z_range=(2, 10)
        )
        assert isinstance(fig, plt.Figure)

        # test z_range
        fig = self.correlation.plot_compare_correlated_events(
            self.corr_matrix, events, z_range=(2, 10)
        )
        assert isinstance(fig, plt.Figure)

        # test corr filtering

        corr_min, corr_max = corr_range
        corr_matrix = np.random.random(size=(num_rows, num_rows))

        fig = self.correlation.plot_compare_correlated_events(
            corr_matrix, events, corr_range=(corr_min, corr_max)
        )
        assert isinstance(fig, plt.Figure)

        corr_mask = np.where(np.logical_and(corr_matrix >= corr_min, corr_matrix <= corr_max))
        fig = self.correlation.plot_compare_correlated_events(
            corr_matrix, events, corr_mask=corr_mask
        )
        assert isinstance(fig, plt.Figure)

    def teardown_method(self):
        # Clean up after the tests, if necessary
        plt.close()
