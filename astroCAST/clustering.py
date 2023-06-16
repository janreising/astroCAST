import logging
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path

import awkward
import fastcluster
import hdbscan
import numpy as np
import pandas as pd
from dask import array as da
from dtaidistance import dtw_barycenter, dtw
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from tqdm import tqdm

from astroCAST.analysis import Events
from astroCAST.helper import wrapper_local_cache, is_ragged

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


class DTW_Linkage:

    """
	"trace_parameters": {
		"cutoff":28, "min_size":10, "max_length":36, "fixed_extension":4, 			"normalization":"standard", "enforce_length": null,
		"extend_curve":true, "differential":true, "use_footprint":false, 			"dff":null, "loc":"ast"
		},
	"max_events": 500000,
	"z_threshold":2, "min_cluster_size":15,
	"max_trace_plot":5, "max_plots":25
"""

    def __init__(self, cache_path=None):

        if cache_path is not None:

            if isinstance(cache_path, str):
                cache_path = Path(cache_path)

            if not cache_path.is_dir():
                cache_path.mkdir()

        self.cache_path = cache_path

    def get_barycenters(self, events, z_threshold, default_cluster = -1,
                        correlation_type="pearson",
                        param_distance={}, param_linkage_matrix={}, param_clustering={}, param_barycenter={}):

        corr = Distance(cache_path=self.cache_path)
        distance_matrix = corr.get_correlation(events,
                                               correlation_type=correlation_type,
                                               correlation_param=param_distance)

        linkage_matrix = self.calculate_linkage_matrix(distance_matrix, **param_linkage_matrix)

        clusters, cluster_labels = self.cluster_linkage_matrix(linkage_matrix, z_threshold, **param_clustering)
        barycenters = self.calculate_barycenters(clusters, cluster_labels, events, **param_barycenter)

        # create a lookup table to sort event indices into clusters
        cluster_lookup_table = defaultdict(lambda: default_cluster)
        for _, row in barycenters.iterrows():
            cluster_lookup_table.update({idx_: row.cluster for idx_ in row.trace_idx})

        return barycenters, cluster_lookup_table

    #todo not completely done
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
            internal_lookup_tables.update(lookup_table)  # todo doesn't work because of mapping conflict between cluster names

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
        # todo how would this work?
        # create a lookup table to sort event indices into clusters
        # cluster_lookup_table = defaultdict(lambda: default_cluster)
        # for _, row in step_two_barycenters.iterrows():
        #
        #     for step_two_idx in row.trace_idx:
        #
        #
        #
        #     cluster_lookup_table.update({idx_: row.cluster for idx_ in row.trace_idx})

        # return barycenters, cluster_lookup_table

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

        if is_ragged(traces):
            traces = awkward.Array(traces)
        else:
            traces = np.array(traces)

        barycenters = {}
        iterator = tqdm(enumerate(clusters.index), total=len(clusters), desc="barycenters:") if show_progress else enumerate(clusters.index)
        for i, cl in iterator:

            idx_ = np.where(cluster_labels == cl)[0]
            sel = [traces[id_] for id_ in idx_]

            nb_initial_samples = len(sel) if len(sel) < 11 else int(0.1*len(sel))
            bc = dtw_barycenter.dba_loop(sel, c=None,
                                         nb_initial_samples=nb_initial_samples,
                                         max_it=max_it, thr=thr, use_c=True, penalty=penalty, psi=psi)

            barycenters[cl] = {"idx":idx_, "bc":bc, "num":clusters.iloc[i], "cluster":cl}

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


class TraceClustering:

    def __init__(self, working_directory:Path, local_cache:bool = True, cache_path:Path = None):

        if local_cache:

            if cache_path is None:
                cache_path = working_directory.joinpath("cache")
                logging.warning(f"no 'cache_path' provided. Choosing {cache_path}")

                if not cache_path.is_dir():
                    cache_path.mkdir()

        self.wd = Path(working_directory)
        self.lc_path = cache_path
        self.local_cache = local_cache

    # todo implement wrapper caching
    @wrapper_local_cache
    def load_bary(self, subj):
        """
        Load barycenter is a static method of the class UnbiasedClustering.
        It loads barycenter data for a given subject.

        Parameters:
        - subj (Path): The path to the subject directory.

        Returns:
        - data (dict): A dictionary containing the loaded barycenter data.
        - "traces" (ndarray): The barycenter traces.
        - "raw" (ndarray): The raw traces.
        - "res" (DataFrame): The events data with additional columns "traces" and "raw".
        - "linkage_matrix" (ndarray): The linkage matrix.

        Raises:
        - AssertionError: If the barycenters directory is not found for the subject.

        Notes:
        - This function assumes that the barycenter data files are located in the "cache/barycenters/" directory
        within the subject directory.

        Example usage:
        >>> subj_path = Path("path/to/subject")
        >>> data = UnbiasedClustering.load_bary(subj_path)
        """

        #print(self.lc_path) #Delete
        #sd = subj.joinpath("cache/barycenters/")
        #assert sd.is_dir(), "cannot find barycenters: {}".format(subj)
        sd = subj.joinpath(self.lc_path) # TODO implement cache, option when local_cache = False?
        assert sd.is_dir(), "cannot find barycenters: {}".format(subj)

        data = {}

        traces_path = sd.joinpath("bary_traces.npy")
        traces = np.load(traces_path.as_posix(), allow_pickle=True)
        data["traces"] = traces

        raw_path = sd.joinpath("raw_traces.npy")
        raw = np.load(raw_path.as_posix(), allow_pickle=True)
        data["raw"] = raw

        df_path = sd.joinpath("events.csv")
        res = pd.read_csv(df_path.as_posix())
        res["traces"] = traces
        res["raw"] = raw
        data["res"] = res

        z_path = sd.joinpath("linkage_matrix.npy")
        Z = np.load(z_path.as_posix())
        data["linkage_matrix"]  = Z

        return data

    def bary_prep(self, subjects: list, mem_data = None, z_thr = 2, min_cluster_size = 15, load_bary = False,
                 dtw_parameters = {"penalty": 0, "psi": None}, show_progress = False):

        data = {}
        iterator = tqdm(subjects) if show_progress else subjects
        for subj_i, (subj, condition) in enumerate(iterator):
            assert subj.is_dir(), "cannot find folder: {}".format(subj)

            #sd = subj.joinpath("cache/barycenters/")
            sd = subj.joinpath(self.lc_path) # TODO implement cache, option when local_cache = False?
            assert sd.is_dir(), "cannot find barycenters: {}".format(subj)

            name = ".".join(subj.name.split(".")[:2])
            name = os.path.splitext(subj.name)[0]

            # load
            #data[name] = UnbiasedClustering.load_bary(subj) if mem_data is None else mem_data[name]
            data[name] = self.load_bary(subj) if mem_data is None else mem_data[name]
            res = data[name]["res"]
            res["index"] = res["index"].apply(lambda x: x.replace("0x", f"{subj_i}x")) # convert to unique identifiers

            traces = data[name]["traces"]
            Z = data[name]["linkage_matrix"]

            if load_bary:
                bary_path = sd.joinpath("barycenters.npy")
                barycenters = np.load(bary_path.as_posix(), allow_pickle=True)
                barycenters["trace_idx"] = barycenters["trace_idx"].apply(
                    lambda x: [s.replace("0x", f"{subj_i}x") for s in x])

                data[name]["barycenter"] = pd.DataFrame(barycenters[()]).transpose().sort_values("num", ascending=False)

            else:

                # filter
                cluster_labels = fcluster(Z = Z, t = z_thr, criterion="distance")
                clusters = pd.Series(cluster_labels).value_counts().sort_index()
                clusters = clusters[clusters > min_cluster_size]

                # clusters
                barycenters = {}
                for i, cl in enumerate(clusters.index):

                    idx_ = np.where(cluster_labels == cl)[0]
                    sel = [traces[id_] for id_ in idx_]

                    bc = dtw_barycenter.dba_loop(sel, c=None,
                                                 nb_initial_samples=max(1, int(0.1*len(sel))),
                                                 max_it=100, thr=1e-5, use_c=True, **dtw_parameters)

                    barycenters[cl] = {"trace_idx":[res["index"][id_] for id_ in idx_], "bc":bc, "num":clusters.iloc[i]}

                barycenters = pd.DataFrame(barycenters).transpose()

                if "num" not in barycenters.columns:
                    raise KeyError("KeyError: 'num': {}".format(barycenters))

                data[name]["barycenter"] = barycenters.sort_values("num", ascending=False)

            data[name]["condition"] = condition

        return data


    def combine_barycenters(self, data: dict, z_thr = 2,
                            events = None, barycenters = None, Z = None,
                            add_cluster_to_events:bool = False, default_cluster = -1,
                            verbose = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Combines barycenters from multiple sources and performs clustering.

        Parameters:
            data (dict): Dictionary containing the loaded barycenter data.
            z_thr (int): Threshold value for clustering.
            events (pd.DataFrame, optional): Combined events dataframe. Default is None.
            barycenters (pd.DataFrame, optional): Combined barycenters dataframe. Default is None.
            Z (ndarray, optional): Linkage matrix. Default is None.
            add_cluster_to_events (bool, optional): Flag indicating whether to add cluster labels to events. Default is False.
            default_cluster (int, optional): Default cluster label. Default is -1.
            verbose (int, optional): Verbosity level. Default is 0.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: A tuple containing the combined events dataframe,
            combined barycenters dataframe, and the linkage matrix.
        """

        # create combined dataframe: res
        if events is None:
            events = []
            for i, key in enumerate(data.keys()):
                temp = data[key]["res"]
                temp["name"] = key

                # # create unique identifiers from sample identifiers
                # temp["index"] = temp["index"].apply(lambda x: "{}y{}".format(i, x))

                temp["idx"] = temp["index"]

                events.append(temp)

            events = pd.concat(events).reset_index(drop = True)

        # create combined dataframe: barycenter
        if barycenters is None:
            barycenters = []
            for i, key in enumerate(data.keys()):

                temp = data[key]["barycenter"]
                temp["name"] = key
                temp["idx"] = temp.index

                # # create unique identifiers from sample identifiers
                # temp["idx"] = temp["idx"].apply(lambda x: "{}y{}".format(i, x))

                barycenters.append(temp)

            barycenters = pd.concat(barycenters).reset_index(drop = True)

        if Z is None:

            comb_traces = barycenters.bc.tolist()

            # create distance matrix between barycenters
            dm = dtw.distance_matrix_fast(comb_traces, compact = True)

            # create linkage matrix
            Z = fastcluster.linkage(dm, method = "complete",
                                    metric = "euclidean", preserve_input = False)

        # cluster traces
        cluster_labels = fcluster(Z = Z, t = z_thr, criterion = 'distance')

        # save new labels
        barycenters["cluster"] = cluster_labels

        if add_cluster_to_events:
            print("moved to 'get_barycenters'")

        if verbose > 0:
            print("\t#events:{:,d}".format(len(events)))
            print("\t#barycenter:{:4,d}".format(len(barycenters)))
            print("\treduction: {:.1f}%\n".format(len(barycenters)/len(events)*100))

            print("\t#clusters: {:,d}".format(len(barycenters.cluster.unique())))
            print("\treduction: {:.1f}% / {:.1f}%".format(
                len(barycenters.cluster.unique())/len(barycenters)*100,
                len(barycenters.cluster.unique())/len(events)*100))

        if self.lc_path is not None:
            events.to_pickle(os.path.join(self.lc_path, "combined_events.pkl"))
            barycenters.to_pickle(os.path.join(self.lc_path, "combined_barycenters.pkl"))
            np.save(os.path.join(self.lc_path, "combined_linkage_matrix.npy"), Z)
        else:
            events.to_pickle(os.path.join(self.wd, "combined_events.pkl"))
            barycenters.to_pickle(os.path.join(self.wd, "combined_barycenters.pkl"))
            np.save(os.path.join(self.wd, "combined_linkage_matrix.npy"), Z)

        return events, barycenters, Z


class Distance:
    """
    A class for computing correlation matrices and histograms.
    """
    def __init__(self, cache_path=None):

        if cache_path is not None:

            if isinstance(cache_path, str):
                cache_path = Path(cache_path)

            if not cache_path.is_dir():
                cache_path.mkdir()

        self.cache_path = cache_path

    @staticmethod
    @wrapper_local_cache
    def get_pearson_correlation(events, dtype=np.single):
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

    @staticmethod
    @wrapper_local_cache
    def get_dtw_correlation(events, use_mmap=False, block=10000, show_progress=True):

        traces = events.events.trace.tolist()

        if is_ragged(traces):
            traces = awkward.Array(traces)
        else:
            traces = np.array(traces)

        logging.warning(f"traces.shape: {traces.shape}")
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
