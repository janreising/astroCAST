import logging
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path

import awkward
import fastcluster
import hdbscan
import networkx as nx
import numpy as np
import pandas as pd
from dask import array as da
from dtaidistance import dtw_barycenter, dtw
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networkx.algorithms import community
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from tqdm import tqdm

from astroCAST.analysis import Events
from astroCAST.helper import wrapper_local_cache, is_ragged, CachedClass

# from dtaidistance import dtw_visualisation as dtwvis
# from dtaidistance import clustering
# from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram

import fastcluster


class HdbScan:

    def __init__(self, min_samples=2, min_cluster_size=2, allow_single_cluster=True, n_jobs=-1):

        self.hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                                   allow_single_cluster=allow_single_cluster, core_dist_n_jobs=n_jobs,
                                   prediction_data=True)

    def fit(self, data, y=None):
        hdb_labels =  self.hdb.fit_predict(data, y=y)
        return hdb_labels

    def predict(self, data):
        test_labels, strengths = hdbscan.approximate_predict(self.hdb, data)
        return test_labels, strengths

    def save(self, path):

        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            path = path.with_name("hdb.p")
            logging.info(f"saving umap to {path}")

        assert not path.is_file(), f"file already exists: {path}"
        pickle.dump(self.hdb, open(path, "wb"))

    def load(self, path):

        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            path = path.with_name("hdb.p")
            logging.info(f"loading umap from {path}")

        assert path.is_file(), f"can't find hdb: {path}"
        self.hdb = pickle.load(open(path, "rb"))


class DTW_Linkage(CachedClass):

    """
	"trace_parameters": {
		"cutoff":28, "min_size":10, "max_length":36, "fixed_extension":4, 			"normalization":"standard", "enforce_length": null,
		"extend_curve":true, "differential":true, "use_footprint":false, 			"dff":null, "loc":"ast"
		},
	"max_events": 500000,
	"z_threshold":2, "min_cluster_size":15,
	"max_trace_plot":5, "max_plots":25
"""

    def get_barycenters(self, events, z_threshold, default_cluster = -1,
                        distance_matrix=None,
                        distance_type="pearson", param_distance={},
                        param_linkage_matrix={}, param_clustering={}, param_barycenter={}):

        if distance_matrix is None:
            corr = Distance(cache_path=self.cache_path)
            distance_matrix = corr.get_correlation(events,
                                                   correlation_type=distance_type,
                                                   correlation_param=param_distance)

        linkage_matrix = self.calculate_linkage_matrix(distance_matrix, **param_linkage_matrix)

        clusters, cluster_labels = self.cluster_linkage_matrix(linkage_matrix, z_threshold, **param_clustering)
        barycenters = self.calculate_barycenters(clusters, cluster_labels, events, **param_barycenter)

        # create a lookup table to sort event indices into clusters
        cluster_lookup_table = defaultdict(lambda: default_cluster)
        for _, row in barycenters.iterrows():
            cluster_lookup_table.update({idx_: row.cluster for idx_ in row.idx})

        return barycenters, cluster_lookup_table

    @wrapper_local_cache
    def get_two_step_barycenters(self, events, step_one_column="subject_id",
                               step_one_threshold=2, step_two_threshold=2,
                              step_one_param={}, step_two_param={},
                              default_cluster=-1):
        """

        Sometimes it is computationally not feasible to cluster by events trace directly. In that case choosing
        a two-step clustering approach is an alternative.

        :param events:
        :return:
        """

        # Step 1
        # calculate individual barycenters
        combined_barycenters = []
        internal_lookup_tables = {}
        for step_one_group in events[step_one_column].unique():

            # create a new Events instance that contains only one group
            event_group = events.copy()
            event_group.events = event_group.events[event_group.events[step_one_column] == step_one_group]

            barycenter, lookup_table = self.get_barycenters(event_group, z_threshold=step_one_threshold,
                                                 default_cluster=default_cluster, **step_one_param)

            combined_barycenters.append(barycenter)
            internal_lookup_tables.update(lookup_table)

        combined_barycenters = pd.concat(combined_barycenters).reset_index(drop=True)
        combined_barycenters.rename(columns={"bc":"trace"}, inplace=True)

        # Step 2
        # create empty Events instance
        combined_events = Events(event_dir=None)
        combined_events.events = combined_barycenters
        combined_events.seed = 2

        # calculate barycenters again
        step_two_barycenters, step_two_lookup_table = self.get_barycenters(combined_events, step_two_threshold,
                                                                           default_cluster=default_cluster,
                                                                           **step_two_param)

        external_lookup_table = defaultdict(lambda: default_cluster)
        for key in internal_lookup_tables.keys():
            bary_id = internal_lookup_tables[key]
            external_lookup_table[key] = step_two_lookup_table[bary_id]

        return combined_barycenters, internal_lookup_tables, step_two_barycenters, external_lookup_table

    @wrapper_local_cache
    def calculate_linkage_matrix(self, distance_matrix, method="average", metric="euclidean"):
        Z = fastcluster.linkage(distance_matrix, method=method, metric=metric, preserve_input=False)
        return Z

    @staticmethod
    def cluster_linkage_matrix(Z, z_threshold, criterion="distance",
                               min_cluster_size=1, max_cluster_size=None):

        cluster_labels = fcluster(Z, z_threshold, criterion=criterion)
        clusters = pd.Series(cluster_labels).value_counts().sort_index()

        if (min_cluster_size > 0) and (min_cluster_size < 1):
            min_cluster_size = int(clusters.sum() * min_cluster_size)

        if min_cluster_size > 1:
            clusters = clusters[clusters >= min_cluster_size]
        elif min_cluster_size < 0:
            logging.warning("min_cluster_size < 0. ignoring argument.")

        if max_cluster_size is not None:
            clusters = clusters[clusters <= min_cluster_size]

        return clusters, cluster_labels

    @wrapper_local_cache
    def calculate_barycenters(self, clusters, cluster_labels, events,
                              max_it=100, thr=1e-5, penalty=0, psi=None,
                              show_progress=True):

        """ Calculate consensus trace (barycenter) for each cluster"""

        traces = events.events.trace.tolist()
        indices = events.events.index.tolist()

        c_idx_, c_bc, c_num, c_cluster = list(), list(), list(), list()
        iterator = tqdm(enumerate(clusters.index), total=len(clusters), desc="barycenters:") if show_progress else enumerate(clusters.index)
        for i, cl in iterator:

            idx_ = np.where(cluster_labels == cl)[0]
            sel = [traces[id_] for id_ in idx_]
            idx = [indices[id_] for id_ in idx_]

            nb_initial_samples = len(sel) if len(sel) < 11 else int(0.1*len(sel))
            bc = dtw_barycenter.dba_loop(sel, c=None,
                                         nb_initial_samples=nb_initial_samples,
                                         max_it=max_it, thr=thr, use_c=True, penalty=penalty, psi=psi)

            c_idx_ += [idx]
            c_bc += [bc]
            c_num += [clusters.iloc[i]]
            c_cluster += [cl]

        barycenters = pd.DataFrame({"idx":c_idx_, "bc":c_bc, "num":c_num, "cluster":c_cluster})

        return barycenters

    @staticmethod
    def plot_cluster_fraction_of_retention(Z, z_threshold=None, min_cluster_size=None, ax=None, save_path=None):

        """ plot fraction of included traces for levels of 'z_threshold' and 'min_cluster_size' """

        if save_path is not None:
            if isinstance(save_path, str):
                save_path = Path(save_path)

            if save_path.is_dir():
                save_path = save_path.joinpath("cluster_fraction_of_retention.png")
                logging.info(f"saving to: {save_path}")

            # create logarithmic x and y scaling
            mcs = np.logspace(start=1, stop=9, num=20, base=2, endpoint=True)
            zs = np.logspace(start=-1, stop=1, num=20, base=10, endpoint=True)

            # calculate the inclusion fraction for each log threshold
            fraction = np.zeros((len(mcs), len(zs)), dtype=np.half)
            for i, mc_ in enumerate(tqdm(mcs)):
                for j, z_ in enumerate(zs):

                    cluster_labels = fcluster(Z, z_, criterion='distance')
                    clusters = pd.Series(cluster_labels).value_counts().sort_index()

                    clusters = clusters[clusters > mc_]

                    fraction[i, j] = clusters.sum()/len(cluster_labels)*100

            # create figure if necessary
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(20, 7))
            else:
                fig = ax.get_figure()

            # plot heatmap
            sns.heatmap(fraction, ax=ax)

            # labeling
            ax.set_xticklabels(np.round(zs, 2))
            ax.set_xlabel("z threshold")
            ax.set_ylabel("num_cluster threshold")

            # convert chosen values to log scale and plot
            if z_threshold is None: x_ = None
            else:
                x_ = 0
                for z_ in zs:
                    if z_threshold > z_:
                        x_ += 1

            if min_cluster_size is None: y_ = 0
            else:
                y_ = 0
                for mc_ in mcs:
                    if min_cluster_size > mc_:
                        y_ += 1

            if (x_ is not None) and (y_ is not None):
                ax.scatter(x_, y_, color="blue", marker="x", s=125, linewidth=5)
            elif x_ is None:
                ax.axhline(y_, color="blue")
            elif y_ is None:
                ax.avhline(x_, color="blue")

            # save figure
            if save_path is not None:
                fig.savefig(save_path.as_posix())

            return fig


class Distance(CachedClass):
    """
    A class for computing correlation matrices and histograms.
    """

    @wrapper_local_cache
    def get_pearson_correlation(self, events, dtype=np.single):
        """
        Computes the correlation matrix of events.

        Args:
            events (np.ndarray or da.Array or pd.DataFrame): Input events data.
            dtype (np.dtype, optional): Data type of the correlation matrix. Defaults to np.single.
            mmap (bool, optional): Flag indicating whether to use memory-mapped arrays. Defaults to False.

        Returns:
            np.ndarray: Correlation matrix.

        Raises:
            ValueError: If events is not one of (np.ndarray, da.Array, pd.DataFrame).
            ValueError: If events DataFrame does not have a 'trace' column.
        """

        if not isinstance(events, (np.ndarray, pd.DataFrame, da.Array, Events)):
            raise ValueError(f"Please provide events as one of (np.ndarray, pd.DataFrame, Events) instead of {type(events)}.")

        if isinstance(events, Events):
            events = events.events

        if isinstance(events, pd.DataFrame):
            if "trace" not in events.columns:
                raise ValueError("Events DataFrame is expected to have a 'trace' column.")

            events = events["trace"].tolist()
            events = np.array(events, dtype=object) if is_ragged(events) else np.array(events)

        if is_ragged(events):

            logging.warning(f"Events are ragged (unequal length), default to slow correlation calculation.")

            N = len(events)
            corr = np.zeros((N, N), dtype=dtype)
            for x in tqdm(range(N)):
                for y in range(N):

                    if corr[y, x] == 0:

                        ex = events[x]
                        ey = events[y]

                        ex = ex - np.mean(ex)
                        ey = ey - np.mean(ey)

                        c = np.correlate(ex, ey, mode="valid")

                        # ensure result between -1 and 1
                        c = np.max(c)
                        c = c / (max(len(ex), len(ey) * np.std(ex) * np.std(ey)))

                        corr[x, y] = c

                    else:
                        corr[x, y] = corr[y, x]
        else:
            corr = np.corrcoef(events).astype(dtype)
            corr = np.tril(corr)

        return corr

    @wrapper_local_cache
    def get_dtw_correlation(self, events, use_mmap=False, block=10000, show_progress=True):

        traces = events.events.trace.tolist()
        N = len(traces)

        if not use_mmap:
            distance_matrix = dtw.distance_matrix_fast(traces,
                        use_pruning=False, parallel=True, compact=True, only_triu=True)

            distance_matrix = np.array(distance_matrix)

        else:

            logging.info("creating mmap of shape ({}, 1)".format(int((N*N-N)/2)))

            tmp = tempfile.TemporaryFile() # todo might not be a good idea to drop a temporary file in the working directory
            distance_matrix = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(int((N*N-N)/2)))

            iterator = range(0, N, block) if not show_progress else tqdm(range(0, N, block),desc="distance matrix:")

            i=0
            for x0 in iterator:

                x1 = min(x0 + block, N)

                dm_ = dtw.distance_matrix_fast(traces, block=((x0, x1), (0, N)),
                            use_pruning=False, parallel=True, compact=True, only_triu=True)

                distance_matrix[i:i+len(dm_)] = dm_
                distance_matrix.flush()

                i = i+len(dm_)

                del dm_

        return distance_matrix

    def get_correlation(self, events, correlation_type="pearson", correlation_param={}):

        funcs = {
            "pearson": lambda x: self.get_pearson_correlation(x, **correlation_param),
            "dtw": lambda x: self.get_dtw_correlation(x, **correlation_param)
        }

        if correlation_type not in funcs.keys():
            raise ValueError(f"cannot find correlation type. Choose one of: {funcs.keys()}")
        else:
            corr_func = funcs[correlation_type]

        return corr_func(events)

    def _get_correlation_histogram(self, corr=None, events=None,
                                   correlation_type="pearson", correlation_param={},
                                   start=-1, stop=1, num_bins=1000, density=False):
        """
        Computes the correlation histogram.

        Args:
            corr (np.ndarray, optional): Precomputed correlation matrix. If not provided, events will be used.
            events (np.ndarray or pd.DataFrame, optional): Input events data. Required if corr is not provided.
            start (float, optional): Start value of the histogram range. Defaults to -1.
            stop (float, optional): Stop value of the histogram range. Defaults to 1.
            num_bins (int, optional): Number of histogram bins. Defaults to 1000.
            density (bool, optional): Flag indicating whether to compute the histogram density. Defaults to False.

        Returns:
            np.ndarray: Correlation histogram counts.

        Raises:
            ValueError: If neither corr nor events is provided.
        """

        if corr is None:
            if events is None:
                raise ValueError("Please provide either 'corr' or 'events' flag.")

            corr = self.get_correlation(events, correlation_type=correlation_type, correlation_param=correlation_param)

        counts, _ = np.histogram(corr, bins=num_bins, range=(start, stop), density=density)

        return counts

    def plot_correlation_characteristics(self, corr=None, events=None, ax=None,
                                         perc=(5e-5, 5e-4, 1e-3, 1e-2, 0.05), bin_num=50, log_y=True,
                                         figsize=(10, 3)):
        """
        Plots the correlation characteristics.

        Args:
            corr (np.ndarray, optional): Precomputed correlation matrix. If not provided, footprint correlation is used.
            ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes, optional): Subplots axes to plot the figure.
            perc (list, optional): Percentiles to plot vertical lines on the cumulative plot. Defaults to [5e-5, 5e-4, 1e-3, 1e-2, 0.05].
            bin_num (int, optional): Number of histogram bins. Defaults to 50.
            log_y (bool, optional): Flag indicating whether to use log scale on the y-axis. Defaults to True.
            figsize (tuple, optional): Figure size. Defaults to (10, 3).

        Returns:
            matplotlib.figure.Figure: Plotted figure.

        Raises:
            ValueError: If ax is provided but is not a tuple of (ax0, ax1).
        """

        if corr is None:
            if events is None:
                raise ValueError("Please provide either 'corr' or 'events' flag.")
            corr = self.get_pearson_correlation(events)

        if ax is None:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
        else:
            if not isinstance(ax, (tuple, list, np.ndarray)) or len(ax) != 2:
                raise ValueError("'ax' argument expects a tuple/list/np.ndarray of (ax0, ax1)")

            ax0, ax1 = ax
            fig = ax0.get_figure()

        # Plot histogram
        bins = ax0.hist(corr.flatten(), bins=bin_num)
        if log_y:
            ax0.set_yscale("log")
        ax0.set_ylabel("Counts")
        ax0.set_xlabel("Correlation")

        # Plot cumulative distribution
        counts, xaxis, _ = bins
        counts = np.flip(counts)
        xaxis = np.flip(xaxis)
        cumm = np.cumsum(counts)
        cumm = cumm / np.sum(counts)

        ax1.plot(xaxis[1:], cumm)
        if log_y:
            ax1.set_yscale("log")
        ax1.invert_xaxis()
        ax1.set_ylabel("Fraction")
        ax1.set_xlabel("Correlation")

        # Plot vertical lines at percentiles
        pos = [np.argmin(abs(cumm - p)) for p in perc]
        vlines = [xaxis[p] for p in pos]
        for v in vlines:
            ax1.axvline(v, color="gray", linestyle="--")

        return fig

    def plot_compare_correlated_events(self, corr, events, event_ids=None,
                                   event_index_range=(0, -1), z_range=None,
                                   corr_mask=None, corr_range=None,
                                   ev0_color="blue", ev1_color="red", ev_alpha=0.5, spine_linewidth=3,
                                   ax=None, figsize=(20, 3), title=None):
        """
        Plot and compare correlated events.

        Args:
            corr (np.ndarray): Correlation matrix.
            events (pd.DataFrame, np.ndarray or Events): Events data.
            event_ids (tuple, optional): Tuple of event IDs to plot.
            event_index_range (tuple, optional): Range of event indices to consider.
            z_range (tuple, optional): Range of z values to plot.
            corr_mask (np.ndarray, optional): Correlation mask.
            corr_range (tuple, optional): Range of correlations to consider.
            ev0_color (str, optional): Color for the first event plot.
            ev1_color (str, optional): Color for the second event plot.
            ev_alpha (float, optional): Alpha value for event plots.
            spine_linewidth (float, optional): Linewidth for spines.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on.
            figsize (tuple, optional): Figure size.
            title (str, optional): Plot title.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if isinstance(events, Events):
            events = events.events

        # Validate event_index_range
        if not isinstance(event_index_range, (tuple, list)) or len(event_index_range) != 2:
            raise ValueError("Please provide event_index_range as a tuple of (start, stop)")

        # Convert events to numpy array if it is a DataFrame
        if isinstance(events, pd.DataFrame):
            if "trace" not in events.columns:
                raise ValueError("'events' dataframe is expected to have a 'trace' column.")

            events = np.array(events.trace.tolist())

        ind_min, ind_max = event_index_range
        if ind_max == -1:
            ind_max = len(events)

        # Choose events
        if event_ids is None:
            # Randomly choose two events if corr_mask and corr_range are not provided
            if corr_mask is None and corr_range is None:
                ev0, ev1 = np.random.randint(ind_min, ind_max, size=2)

            # Choose events based on corr_mask
            elif corr_mask is not None:
                # Warn if corr_range is provided and ignore it
                if corr_range is not None:
                    logging.warning("Prioritizing 'corr_mask'; ignoring 'corr_range' argument.")

                if isinstance(corr_mask, (list, tuple)):
                    corr_mask = np.array(corr_mask)

                    if corr_mask.shape[0] != 2:
                        raise ValueError(f"corr_mask should have a shape of (2xN) instead of {corr_mask.shape}")

                rand_index = np.random.randint(0, corr_mask.shape[1])
                ev0, ev1 = corr_mask[:, rand_index]

            # Choose events based on corr_range
            elif corr_range is not None:
                # Validate corr_range
                if len(corr_range) != 2:
                    raise ValueError("Please provide corr_range as a tuple of (min_corr, max_corr)")

                corr_min, corr_max = corr_range

                # Create corr_mask based on corr_range
                corr_mask = np.array(np.where(np.logical_and(corr >= corr_min, corr <= corr_max)))
                logging.warning("Thresholding the correlation array may take a long time. Consider precalculating the 'corr_mask' with eg. 'np.where(np.logical_and(corr >= corr_min, corr <= corr_max))'")

                rand_index = np.random.randint(0, corr_mask.shape[1])
                ev0, ev1 = corr_mask[:, rand_index]

        else:
            ev0, ev1 = event_ids

        if isinstance(ev0, np.ndarray):
            ev0 = ev0[0]
            ev1 = ev1[0]

        # Choose z range
        trace_0 = np.squeeze(events[ev0]).astype(float)
        trace_1 = np.squeeze(events[ev1]).astype(float)

        if isinstance(trace_0, da.Array):
            trace_0 = trace_0.compute()
            trace_1 = trace_1.compute()

        if z_range is not None:
            z0, z1 = z_range

            if (z0 > len(trace_0)) or (z0 > len(trace_1)):
                raise ValueError(f"Left bound z0 larger than event length: {z0} > {len(trace_0)} or {len(trace_1)}")

            trace_0 = trace_0[z0: min(z1, len(trace_0))]
            trace_1 = trace_1[z0: min(z1, len(trace_1))]

        ax.plot(trace_0, color=ev0_color, alpha=ev_alpha)
        ax.plot(trace_1, color=ev1_color, alpha=ev_alpha)

        if title is None:
            if isinstance(ev0, np.ndarray):
                ev0 = ev0[0]
                ev1 = ev1[0]
            ax.set_title("{:,d} x {:,d} > corr: {:.4f}".format(ev0, ev1, corr[ev0, ev1]))

        def correlation_color_map(colors=None):
            """
            Create a correlation color map.

            Args:
                colors (list, optional): List of colors.

            Returns:
                function: Color map function.
            """
            if colors is None:
                neg_color = (0, "#ff0000")
                neu_color = (0.5, "#ffffff")
                pos_color = (1, "#0a700e")

                colors = [neg_color, neu_color, pos_color]

            cm = LinearSegmentedColormap.from_list("Custom", colors, N=200)

            def lsc(v):
                assert np.abs(v) <= 1, "Value must be between -1 and 1: {}".format(v)

                if v == 0:
                    return cm(100)
                if v < 0:
                    return cm(100 - int(abs(v) * 100))
                elif v > 0:
                    return cm(int(v * 100 + 100))

            return lsc

        lsc = correlation_color_map()
        for spine in ax.spines.values():
            spine.set_edgecolor(lsc(corr[ev0, ev1]))
            spine.set_linewidth(spine_linewidth)

        return fig


class Modules(CachedClass):

    def __init__(self, events, cache_path=None):

        if cache_path is None:
            cache_path = events.cache_path

        super().__init__(cache_path=cache_path)

        if events.is_multi_subject():
            logging.warning("multiple values for 'subject_id' were found in the events table. "
                            "The module class expects all events to belong to a single recording.")

        self.events = events

    def __hash__(self):
        return self.events.__hash__()

    @wrapper_local_cache
    def _create_node_edge_tables(self, correlation, correlation_boundaries=(0.98, 1)):

        # select correlations within given boundaries
        lower_bound, upper_bound = correlation_boundaries
        selected_correlations = np.where(np.logical_and(correlation >= lower_bound, correlation < upper_bound))

        # deal with compact correlation matrices
        if len(selected_correlations) == 1:
            triu_indices = np.array(np.triu_indices(len(self.events)))
            selected_correlations = triu_indices[:, selected_correlations].squeeze()

        # filter events
        selected_idx = np.unique(selected_correlations).tolist()
        # selected_events = self.events.events.iloc[selected_idx]
        selected_events = self.events[selected_idx]

        # create nodes table
        nodes = pd.DataFrame({
            "i_idx":selected_idx, "x": selected_events.cx, "y":selected_events.cy, "trace_idx": selected_events.index
        })

        # create edges table
        edges = pd.DataFrame({
            "source":selected_correlations[0, :], "target":selected_correlations[1, :]
        })

        # convert edge indices from iidx in correlation array
        # to row_idx in nodes table
        lookup = dict(zip(nodes.i_idx.tolist(), nodes.index.tolist()))
        edges.source = edges.source.map(lookup)
        edges.target = edges.target.map(lookup)

        return nodes, edges

    @wrapper_local_cache
    def create_graph(self, correlation, correlation_boundaries=(0.98, 1), exclude_out_of_cluster_connection=True):

        nodes, edges = self._create_node_edge_tables(correlation, correlation_boundaries=correlation_boundaries)

        # create graph and populate with edges
        G = nx.Graph()
        for _, edge in tqdm(edges.iterrows(), total=len(edges)):
            G.add_edge(*edge)

        # calculate modularity
        communities = community.greedy_modularity_communities(G)

        # assign modules
        nodes["module"] = -1
        n_mod = 0
        for module in tqdm(communities):
            for m in module:
                nodes.loc[m, "module"] = n_mod
            n_mod += 1

        # add module column to nodes dataframe
        nodes["module"] = nodes["module"].astype("category")

        # add module to edges
        modules_sources = nodes.module.loc[edges.source.tolist()].tolist()
        modules_targets = nodes.module.loc[edges.target.tolist()].tolist()
        edge_module = modules_sources

        if exclude_out_of_cluster_connection:
            for i in np.where(np.array(modules_sources) != np.array(modules_targets))[0]:
                edge_module[i] = -1

        edges["module"] = edge_module

        lookup_cluster_table = dict(zip(nodes.trace_idx.tolist(), nodes.module.tolist()))

        return nodes, edges, lookup_cluster_table

    def summarize_modules(self, nodes):

        from pointpats import centrography

        summary = {}

        funcs = {
            "mean_center": lambda module: None if len(module) < 1 else centrography.mean_center(module[["x", "y"]].astype(float)),
            "median_center": lambda module: None if len(module) < 1 else centrography.euclidean_median(module[["x", "y"]].astype(float)),
            "std_distance": lambda module: None if len(module) < 1 else centrography.std_distance(module[["x", "y"]].astype(float)),
            "coordinates": lambda module: module[["x", "y"]].astype(float).values,
            "num_events": lambda module: len(module),
        }

        for func_key in funcs.keys():
            func = funcs[func_key]
            summary[func_key] = nodes.groupby("module").apply(func)

        return pd.DataFrame(summary)


