import concurrent.futures
import inspect
import itertools
import logging
import pickle
import tempfile
import traceback
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path
from typing import List, Literal, Tuple, Union

import dask
import fastcluster
import hdbscan
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from dask import array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dtaidistance import dtw, dtw_barycenter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from networkx.algorithms import community
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import KDTree
from sklearn import cluster, ensemble, gaussian_process, linear_model, neighbors, neural_network, tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from astrocast import helper
from astrocast.analysis import Events, MultiEvents
from astrocast.helper import CachedClass, MiniLogger, Normalization, is_ragged, wrapper_local_cache


# from tqdm.auto import tqdm


class HdbScan:
    
    def __init__(self, events=None, min_samples=2, min_cluster_size=2, allow_single_cluster=True, n_jobs=-1):
        
        self.hdb = hdbscan.HDBSCAN(
                min_samples=min_samples, min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster,
                core_dist_n_jobs=n_jobs, prediction_data=True
                )
        
        self.events = events
    
    def fit(self, embedding, y=None) -> np.ndarray:
        
        hdb_labels = self.hdb.fit_predict(embedding, y=y)
        
        if self.events is not None:
            return self.events.create_lookup_table(hdb_labels)
        
        else:
            return hdb_labels
    
    def predict(self, embedding, events=None):
        
        if events is None:
            events = self.events
        
        labels, strengths = hdbscan.approximate_predict(self.hdb, embedding)
        
        if events is not None:
            
            lookup_table = events.create_lookup_table(labels)
            lookup_table_strength = events.create_lookup_table(strengths)
            
            return lookup_table, lookup_table_strength
        
        else:
            return labels, strengths
    
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


class KMeansClustering(CachedClass):
    
    @wrapper_local_cache
    def fit(self, events, embedding, n_clusters, param={}, default_cluster=-1):
        if len(events) != len(embedding):
            raise ValueError(
                    f"embedding and events must have the same length: "
                    f"len(embedding)={len(embedding)} vs. len(events)={len(events)}"
                    )
        
        labels = KMeans(n_clusters=n_clusters, **param).fit_transform(embedding)
        
        cluster_lookup_table = defaultdict(lambda: default_cluster)
        cluster_lookup_table.update({k: label for k, label in list(zip(events.events.index.tolist(), labels.tolist()))})
        
        return cluster_lookup_table


class Linkage(CachedClass):
    
    def __init__(self, events: Union[Events, MultiEvents], cache_path=None, logging_level=logging.INFO):
        super().__init__(logging_level=logging_level, cache_path=cache_path, logger_name="Linkage")
        
        self.n_centroids = None
        self.pre_cluster_labels = None
        self.events = events
        self.my_hash = hash(events) if cache_path is not None else None
        
        self.Z = None
    
    def __hash__(self):
        return self.my_hash
    
    def get_barycenters(
            self, cutoff, criterion="distance", default_cluster=-1,
            distance_matrix=None, distance_type: Literal['pearson', 'dtw', 'dtw_parallel'] = "pearson",
            min_cluster_size: int = 1, max_cluster_size: int = None,
            param_distance={}, return_linkage_matrix=False,
            param_linkage={}, param_barycenter={}
            ):
        
        """

        :param events:
        :param cutoff: maximum cluster distance (criterion='distance') or number of clusters (criterion='maxclust')
        :param criterion: one of 'inconsistent', 'distance', 'monocrit', 'maxclust' or 'maxclust_monocrit'
        :param default_cluster: cluster value for excluded events
        :param distance_matrix:
        :param distance_type:
        :param param_distance:
        :param param_linkage:
        :param param_clustering:
        :param param_barycenter:
        :return:
        """
        
        events = self.events
        
        if distance_matrix is None:
            corr = Distance(events, cache_path=self.cache_path)
            distance_matrix = corr.get_correlation(correlation_type=distance_type, correlation_param=param_distance
                                                   )
        
        linkage_matrix = self.calculate_linkage_matrix(distance_matrix, **param_linkage)
        clusters, cluster_labels = self.cluster_linkage_matrix(
                linkage_matrix, cutoff, criterion=criterion,
                min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                )
        barycenters = self.calculate_barycenters(clusters, cluster_labels, **param_barycenter)
        
        # create a lookup table to sort event indices into clusters
        cluster_lookup_table = defaultdict(lambda: default_cluster)
        for _, row in barycenters.iterrows():
            cluster_lookup_table.update({idx_: row.cluster for idx_ in row.idx})
        
        if return_linkage_matrix:
            return barycenters, cluster_lookup_table, linkage_matrix
        else:
            return barycenters, cluster_lookup_table
    
    @wrapper_local_cache
    def get_two_step_barycenters(
            self, step_one_column="subject_id", step_one_threshold=2, step_two_threshold=2, step_one_param={},
            step_two_param={}, default_cluster=-1
            ):
        """

        Sometimes it is computationally not feasible to cluster by events trace directly. In that case choosing
        a two-step clustering approach is an alternative.

        :param events:
        :return:
        """
        
        events = self.events.events
        
        # Step 1
        # calculate individual barycenters
        combined_barycenters = []
        internal_lookup_tables = {}
        for step_one_group in events[step_one_column].unique():
            # create a new Events instance that contains only one group
            event_group = events.copy()
            event_group.events = event_group.events[event_group.events[step_one_column] == step_one_group]
            
            barycenter, lookup_table = self.get_barycenters(
                    event_group, cutoff=step_one_threshold, default_cluster=default_cluster, **step_one_param
                    )
            
            combined_barycenters.append(barycenter)
            internal_lookup_tables.update(lookup_table)
        
        combined_barycenters = pd.concat(combined_barycenters).reset_index(drop=True)
        combined_barycenters.rename(columns={"bc": "trace"}, inplace=True)
        
        # Step 2
        # create empty Events instance
        combined_events = Events(event_dir=None)
        combined_events.events = combined_barycenters
        combined_events.seed = 2
        
        # calculate barycenters again
        step_two_barycenters, step_two_lookup_table = self.get_barycenters(
                combined_events, step_two_threshold, default_cluster=default_cluster, **step_two_param
                )
        
        external_lookup_table = defaultdict(lambda: default_cluster)
        for key in internal_lookup_tables.keys():
            bary_id = internal_lookup_tables[key]
            external_lookup_table[key] = step_two_lookup_table[bary_id]
        
        return combined_barycenters, internal_lookup_tables, step_two_barycenters, external_lookup_table
    
    @wrapper_local_cache
    def calculate_linkage_matrix(self, distance_matrix, method="average", metric="euclidean",
                                 use_co_association: bool = False,
                                 use_random_sample: bool = False,
                                 fraction_random_samples: float = 0.1, min_k_means_clusters: int = 100,
                                 n_subsets: int = 10, n_sub_clusters: int = 100,
                                 pca_retained_variance: float = 0.95, fraction_pre_clusters: float = 0.1,
                                 use_processes: bool = False
                                 ):
        """
            Calculate the linkage matrix for hierarchical clustering.

            Depending on the parameters, either computes the linkage matrix directly from the distance matrix
            or uses a co-association matrix approach for more robust clustering.

            Args:
                distance_matrix: The distance matrix used for clustering.
                method: The linkage algorithm to use (default is "average").
                metric: The distance metric to use (default is "euclidean").
                use_co_association: Whether to use the co-association matrix approach (default is False).
                use_random_sample: Whether to use random sampling for co-association (default is False).
                fraction_random_samples: Fraction of the data to use for each random sample (default is 0.1).
                n_subsets: Number of subsets to use for co-association (default is 10).
                n_sub_clusters: Number of clusters to form in each subset (default is 100).

            Returns:
                The linkage matrix computed from the distance matrix.
            """
        
        if not use_co_association:
            linkage_matrix = fastcluster.linkage(distance_matrix, method=method, metric=metric, preserve_input=False)
            
            self.Z = linkage_matrix
            return linkage_matrix
        
        else:
            # linkage_matrix, pre_cluster_labels, n_centroids = self._calculate_co_association_linkage(
            linkage_matrix = self._calculate_co_association_linkage(
                    distance_matrix=distance_matrix,
                    method=method, metric=metric,
                    use_random_sample=use_random_sample,
                    fraction_random_samples=fraction_random_samples,
                    min_k_means_clusters=min_k_means_clusters,
                    n_subsets=n_subsets,
                    n_sub_clusters=n_sub_clusters,
                    pca_retained_variance=pca_retained_variance,
                    fraction_pre_clusters=fraction_pre_clusters,
                    use_processes=use_processes)
            
            self.Z = linkage_matrix
            # self.pre_cluster_labels = pre_cluster_labels
            # self.n_centroids = n_centroids
            # return linkage_matrix, pre_cluster_labels, n_centroids
            return linkage_matrix
    
    @staticmethod
    def _calculate_co_association_linkage(distance_matrix: Union[np.ndarray, da.Array], method: str, metric: str,
                                          use_random_sample: bool = False,
                                          fraction_random_samples: float = 0.1, min_k_means_clusters: int = 100,
                                          n_subsets: int = 10, n_sub_clusters: int = 100,
                                          pca_retained_variance: float = 0.95, fraction_pre_clusters: float = 0.1,
                                          use_processes: bool = False):
        """
            Calculate the linkage matrix using a co-association matrix approach.

            This method is useful for more robust clustering by aggregating multiple clustering results
            from different subsets or samples of the data.

            Args:
                distance_matrix: The distance matrix used for clustering.
                method: The linkage algorithm to use.
                metric: The distance metric to use.
                use_random_sample: Whether to use random sampling for co-association (default is False).
                fraction_random_samples: Fraction of the data to use for each random sample (default is 0.1).
                n_subsets: Number of subsets to use for co-association (default is 10).
                n_sub_clusters: Number of clusters to form in each subset (default is 100).

            Returns:
                The linkage matrix computed from the co-association matrix.
            """
        if not isinstance(distance_matrix, da.Array):
            distance_matrix = da.from_array(distance_matrix, chunks=distance_matrix.shape)
        
        def compute_linkage(dm: da.Array, indices_):
            """
                Compute the linkage matrix for a subset of the distance matrix.

                Args:
                    dm: The dask array representing the distance matrix.
                    indices_: The indices of the subset to use for clustering.

                Returns:
                    A tuple containing the linkage matrix and the subset indices.
                """
            subset_dist_matrix = dm[indices_][:, indices_]
            lm = fastcluster.linkage(subset_dist_matrix, method=method, metric=metric, preserve_input=True)
            return lm, indices_
        
        if use_random_sample:
            
            # Generate multiple random samples
            n = len(distance_matrix)
            sample_size_n = int(n * fraction_random_samples)
            chosen_indices = [np.random.choice(n, sample_size_n, replace=False) for _ in range(n_subsets)]
        
        else:
            
            # Split data into subsets
            indices = np.arange(distance_matrix.shape[0])
            chosen_indices = np.array_split(indices, n_subsets)
        
        pbar = ProgressBar(minimum=0)
        pbar.register()
        
        linkage_results = []
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            with Client(memory_limit='auto', processes=use_processes, ) as client:
                for subset_indices in chosen_indices:
                    linkage_results.append(
                            client.submit(compute_linkage, distance_matrix, subset_indices)
                            )
                
                results = client.gather(linkage_results)
        
        pbar.unregister()
        
        logging.info(f"finished calculation of subset linkage matrices.")
        
        # Initialize co-association matrix
        co_association_matrix = np.zeros((len(distance_matrix), len(distance_matrix)))
        
        # Populate co-association matrix
        for link_matrix, subset_indices in tqdm(results, desc='Creating co-association matrix'):
            # Generate flat clusters from linkage matrix
            labels = fcluster(link_matrix, t=n_sub_clusters, criterion='maxclust')
            
            for idx1 in range(len(subset_indices)):
                for idx2 in range(len(subset_indices)):
                    if labels[idx1] == labels[idx2]:
                        co_association_matrix[subset_indices[idx1], subset_indices[idx2]] += 1
        
        # Normalize co-association matrix
        co_association_matrix /= n_subsets
        
        logging.info(f"finished creation of co-association matrix")
        
        # Perform final clustering on the co-association matrix
        # todo: this step is still exactly as slow as before
        # linkage_matrix = fastcluster.linkage(co_association_matrix,
        # method=method, metric=metric, preserve_input=False)
        
        # Optional: Dimensionality Reduction using PCA
        if pca_retained_variance is not None:
            before_n = len(co_association_matrix)
            pca = PCA(n_components=pca_retained_variance)  # Retain 95% of variance
            co_association_matrix = pca.fit_transform(co_association_matrix)
            after_n = len(co_association_matrix)
            print(f"reduced co association matrix from {before_n} to {after_n} "
                  f"({after_n / before_n * 100:.1f}%)")
        
        # Step 2: Pre-Clustering using K-Means
        # n_pre_clusters = min(int(len(distance_matrix) * fraction_pre_clusters), min_k_means_clusters)
        # kmeans = KMeans(n_clusters=n_pre_clusters)
        # pre_cluster_labels = kmeans.fit_predict(co_association_matrix)
        # centroids = kmeans.cluster_centers_
        # n_centroids = len(centroids)
        
        # Step 3: Hierarchical Clustering on Centroids
        # centroid_linkage_matrix = fastcluster.linkage(centroids, method=method, metric=metric)
        linkage_matrix = fastcluster.linkage(co_association_matrix, method=method, metric=metric)
        
        # return centroid_linkage_matrix, pre_cluster_labels, n_centroids
        return linkage_matrix
    
    @staticmethod
    def cluster_linkage_matrix(
            Z, cutoff, criterion="distance", min_cluster_size=1, max_cluster_size=None,
            pre_cluster_labels=None, n_centroids: int = None
            ):
        
        valid_criterion = ('inconsistent', 'distance', 'monocrit', 'maxclust', 'maxclust_monocrit')
        if criterion not in valid_criterion:
            raise ValueError(
                    f"criterion has to be one of: {valid_criterion}. "
                    f"For more guidance see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html"
                    )
        
        cluster_labels = fcluster(Z, t=cutoff, criterion=criterion)
        
        if pre_cluster_labels is not None:
            
            cluster_map = {i: cluster_labels[i] for i in range(n_centroids)}
            cluster_labels = np.array([cluster_map[label] for label in pre_cluster_labels])
        
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
    def calculate_barycenters(
            self, clusters, cluster_labels, init_fraction=0.1, max_it=100, thr=1e-5, penalty=0, psi=None,
            show_progress=True
            ):
        
        """ Calculate consensus trace (barycenter) for each cluster"""
        
        events = self.events
        
        if not isinstance(events, pd.DataFrame):
            events = events.events
        
        traces = events.trace.tolist()
        indices = events.index.tolist()
        
        c_idx_, c_bc, c_num, c_cluster = list(), list(), list(), list()
        iterator = tqdm(
                enumerate(clusters.index), total=len(clusters), desc="barycenters:"
                ) if show_progress else enumerate(clusters.index)
        for i, cl in iterator:
            idx_ = np.where(cluster_labels == cl)[0]
            sel = [np.array(traces[id_], dtype=np.double) for id_ in idx_]
            idx = [indices[id_] for id_ in idx_]
            
            nb_initial_samples = len(sel) if len(sel) < 11 else int(init_fraction * len(sel))
            bc = dtw_barycenter.dba_loop(
                    sel, c=None, nb_initial_samples=nb_initial_samples, max_it=max_it, thr=thr, use_c=True,
                    penalty=penalty,
                    psi=psi
                    )
            
            c_idx_ += [idx]
            c_bc += [bc]
            c_num += [clusters.iloc[i]]
            c_cluster += [cl]
        
        barycenters = pd.DataFrame({"idx": c_idx_, "bc": c_bc, "num": c_num, "cluster": c_cluster})
        
        return barycenters
    
    @staticmethod
    def plot_cluster_fraction_of_retention(
            Z, cutoff, criterion='distance', min_cluster_size=None, ax=None, save_path=None
            ):
        
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
            fraction = np.zeros((len(mcs), len(zs)), dtype=float)
            for i, mc_ in enumerate(tqdm(mcs)):
                for j, z_ in enumerate(zs):
                    cluster_labels = fcluster(Z, z_, criterion=criterion)
                    clusters = pd.Series(cluster_labels).value_counts().sort_index()
                    
                    clusters = clusters[clusters > mc_]
                    
                    fraction[i, j] = clusters.sum() / len(cluster_labels) * 100
            
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
            if cutoff is None:
                x_ = None
            
            else:
                x_ = 0
                for z_ in zs:
                    if cutoff > z_:
                        x_ += 1
            
            if min_cluster_size is None:
                y_ = 0
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
    
    def __init__(self, events: Union[pd.DataFrame, Events] = None, cache_path: Union[str, Path] = None,
                 logging_level=logging.INFO):
        super().__init__(cache_path=cache_path, logging_level=logging_level, logger_name="Distance")
        
        if isinstance(events, pd.DataFrame):
            self.events = events
            self.my_hash = helper.hash_events_dataframe(events) if cache_path is not None else None
        else:
            self.events = events.events
            self.my_hash = hash(events) if cache_path is not None else None
    
    def __hash__(self):
        return self.my_hash
    
    @staticmethod
    def _clean_distance_matrix(distance_matrix: np.ndarray, replace_value: Union[Literal['max'], float, int] = 'max'):
        
        x, y = np.where(np.isnan(distance_matrix))
        xx, yy = np.where(np.isinf(distance_matrix))
        
        x = np.concatenate([x, xx])
        y = np.concatenate([y, yy])
        
        if replace_value == 'max':
            replace_value = np.nanmax(distance_matrix[x, y]) + 1
        
        for i in range(len(x)):
            
            x_ = x[i]
            y_ = y[i]
            
            distance_matrix[x_, y_] = replace_value
        
        return distance_matrix
    
    @staticmethod
    def _fix_invalid_distance_matrix(distance_matrix: np.ndarray, traces: List[np.ndarray],
                                     max_tries=3):
        
        # troubleshooting
        idx_nan = np.isnan(distance_matrix)
        idx_inf = np.isinf(distance_matrix)
        
        if idx_nan.any() or idx_inf.any():
            
            tries = 0
            while True:
                
                idx_nan = np.where(idx_nan)
                idx_inf = np.where(idx_inf)
                
                idx_x = np.concatenate([idx_nan[0], idx_inf[0]], axis=0)
                idx_y = np.concatenate([idx_nan[1], idx_inf[1]], axis=0)
                
                if tries == 0:
                    logging.warning(f"invalid distances identified ({len(idx_x)}, "
                                    f"{len(idx_x) / len(distance_matrix) ** 2 * 100:.1f}%). "
                                    f"Trying to fix automatically.")
                else:
                    logging.warning(f"Still invalid distances found ({len(idx_x)}). "
                                    f"Trying {tries}/{max_tries}")
                
                for i in tqdm(range(len(idx_x))):
                    x = idx_x[i]
                    y = idx_y[i]
                    
                    trace_x = np.array(traces[x]).astype(np.double)
                    trace_y = np.array(traces[y]).astype(np.double)
                    
                    distance_matrix[x, y] = dtw.distance_fast(trace_x, trace_y, use_pruning=False)
                
                idx_nan = np.isnan(distance_matrix)
                idx_inf = np.isinf(distance_matrix)
                
                if not idx_nan.any() and not idx_inf.any():
                    break
                
                elif tries >= max_tries:
                    logging.warning(f"max tries exceeded. Setting invalid values to 1e20")
                    max_distance = np.nanmax(distance_matrix)
                    distance_matrix[idx_nan] = max_distance
                    distance_matrix[idx_inf] = max_distance
        
        return distance_matrix
    
    @staticmethod
    def distance_to_similarity(distance_matrix, r=None, a=None, method='exponential', return_params=False,
                               cover_quantile=False):
        """Transform a distance matrix to a similarity matrix.

        The available methods are:
        - Exponential: e^(-D / r)
          r is max(D) if not given
        - Gaussian: e^(-D^2 / r^2)
          r is max(D) if not given
        - Reciprocal: 1 / (r + D*a)
          r is 1 if not given
        - Reverse: r - D
          r is min(D) + max(D) if not given

        All of these methods are monotonically decreasing transformations. The order of the
        distances thus remains unchanged (only the direction).

        Example usage::

            dist_matrix = dtw.distance_matrix(series)
            sim_matrix = distance_to_similarity(dist_matrix)


        :param distance_matrix: The distance matrix
        :param r: A scaling or smoothing parameter.
        :param method: One of 'exponential', 'gaussian', 'reciprocal', 'reverse'
        :param return_params: Return the value used for parameter r
        :param cover_quantile: Compute r such that the function covers the `cover_quantile` fraction of the data.
            Expects a value in [0,1]. If a tuple (quantile, value) is given, then r (and a) are set such that
            at the quantile the given value is reached (if not given, value is 1-quantile).
        :return: Similarity matrix S
        """
        if cover_quantile is not False:
            if type(cover_quantile) in [tuple, list]:
                cover_quantile, cover_quantile_target = cover_quantile
            else:
                cover_quantile_target = 1 - cover_quantile
        else:
            cover_quantile_target = None
        method = method.lower()
        if method == 'exponential':
            if r is None:
                if cover_quantile is False:
                    r = np.max(distance_matrix)
                else:
                    r = -np.quantile(distance_matrix, cover_quantile) / np.log(cover_quantile_target)
            S = np.exp(-distance_matrix / r)
        elif method == 'gaussian':
            if r is None:
                if cover_quantile is False:
                    r = np.max(distance_matrix)
                else:
                    r = np.sqrt(-np.quantile(distance_matrix, cover_quantile) ** 2 / np.log(cover_quantile_target))
            S = np.exp(-np.power(distance_matrix, 2) / r ** 2)
        elif method == 'reciprocal':
            if r is None:
                r = 1
            if a is None:
                if cover_quantile is False:
                    a = 1
                else:
                    a = (1 - cover_quantile_target * r) / (
                            cover_quantile_target * np.quantile(distance_matrix, cover_quantile))
            S = 1 / (r + distance_matrix * a)
        elif method == 'reverse':
            if r is None:
                r = np.min(distance_matrix) + np.max(distance_matrix)
            S = (r - distance_matrix) / r
        else:
            raise ValueError("method={} is not supported".format(method))
        if return_params:
            return S, r
        else:
            return S
    
    @staticmethod
    def plot_compare_correlated_events(
            corr, events, event_ids=None, event_index_range=(0, -1), z_range=None, corr_mask=None,
            corr_range=None, ev0_color="blue", ev1_color="red", ev_alpha=0.5, spine_linewidth=3, ax=None,
            figsize=(20, 3), title=None, trace_column: str = "trace"
            ):
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
            if trace_column not in events.columns:
                raise ValueError(f"'events' dataframe is expected to have the column: {trace_column}")
            
            events = np.array(events[trace_column].tolist())
        
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
                logging.warning(
                        "Thresholding the correlation array may take a long time. Consider precalculating the 'corr_mask' with eg. 'np.where(np.logical_and(corr >= corr_min, corr <= corr_max))'"
                        )
                
                rand_index = np.random.randint(0, corr_mask.shape[1])
                ev0, ev1 = corr_mask[:, rand_index]
        
        else:
            ev0, ev1 = event_ids
        
        if isinstance(ev0, np.ndarray):
            ev0 = ev0[0]
            ev1 = ev1[0]
        
        # Choose z range
        trace_0 = events[ev0]
        trace_1 = events[ev1]
        
        if isinstance(trace_0, da.Array):
            trace_0 = trace_0.compute()
            trace_1 = trace_1.compute()
        
        trace_0 = np.squeeze(trace_0).astype(float)
        trace_1 = np.squeeze(trace_1).astype(float)
        
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
    
    @wrapper_local_cache
    def get_pearson_correlation(self, dtype=np.single, trace_column: str = "trace") -> np.ndarray:
        """ Computes the correlation matrix of events.

        Args:
            dtype: Data type of the correlation matrix. Defaults to np.single.
            trace_column: column that contains the signal

        Raises:
            ValueError: If events is not one of (np.ndarray, da.Array, pd.DataFrame).
            ValueError: If events DataFrame does not have a 'trace' column.
        """
        
        events = self.events
        
        if isinstance(events, pd.DataFrame):
            if trace_column not in events.columns:
                raise ValueError(f"Events DataFrame is expected to have a trace column: {trace_column}")
            
            traces = events[trace_column].tolist()
            traces_are_ragged = is_ragged(traces)
            
            if traces_are_ragged:
                traces = np.array(traces, dtype=object)
            
            else:
                traces = np.array(traces, dtype=float)
        
        elif isinstance(events, (np.ndarray, da.Array)):
            traces = events
            traces_are_ragged = is_ragged(traces)
        else:
            raise ValueError(f"events must be of type Events, pd.DataFrame, np.ndarray or da.Array; not {type(events)}")
        
        if traces_are_ragged:
            
            logging.warning(f"Events are ragged (unequal length), default to slow correlation calculation.")
            
            # todo find an elegant solution
            #  dask natively cannot handle awkward arrays well. np.mean, map_blocks, etc. don't seem to work
            #  there is a dask-awkward library, but it is not very mature
            if isinstance(traces, da.Array):
                traces = traces.compute()
            
            N = len(traces)
            corr = np.zeros((N, N), dtype=dtype)
            for x in tqdm(range(N)):
                for y in range(N):
                    
                    if corr[y, x] == 0:
                        
                        ex = traces[x]
                        ey = traces[y]
                        
                        if np.isnan(ex).any() or np.isnan(ey).any():
                            logging.warning(
                                    f"Traces contain NaN values. Masking array to avoid incorrect distance measurement."
                                    f"This will slow processing significantly.")
                            
                            ex = np.ma.masked_invalid(ex)
                            ey = np.ma.masked_invalid(ey)
                            
                            ex -= np.nanmean(ex)
                            ey -= np.nanmean(ey)
                            
                            c = np.ma.correlate(ex, ey, mode='valid')
                            c = c.data
                        
                        else:
                            
                            ex = ex - np.mean(ex)
                            ey = ey - np.mean(ey)
                            
                            c = np.correlate(ex, ey, mode="valid")
                        
                        # ensure result between -1 and 1
                        c = np.max(c)
                        c /= (max(len(ex), len(ey) * np.std(ex) * np.std(ey)))
                        
                        corr[x, y] = c
                    
                    else:
                        corr[x, y] = corr[y, x]
        else:
            
            if np.isnan(traces).any():
                logging.warning(f"Traces contain NaN values. Masking array to avoid incorrect distance measurement."
                                f"This will slow processing significantly.")
                traces = np.ma.masked_invalid(traces)
                corr = np.ma.corrcoef(traces).data
                corr = corr.data
            else:
                corr = np.corrcoef(traces)
            
            corr = corr.astype(dtype)
            corr = np.tril(corr)
        
        return corr
    
    @wrapper_local_cache
    def get_dtw_correlation(
            self, use_mmap: bool = False, block: int = 10000,
            parallel: bool = True, compact: bool = False, only_triu: bool = False, use_pruning: bool = False,
            return_similarity: bool = True, show_progress: bool = True, trace_column: str = "trace",
            max_tries: int = 3
            ):
        """
                    Computes the dynamic time warping (DTW) correlation matrix for a set of time series.

                    This function calculates the pairwise DTW distances between time series data,
                    represented by the `events` object. It uses a fast DTW computation method and can handle
                    large datasets by optionally utilizing memory mapping (mmap).

                    .. error:

                        This function will not work on most systems with MacOS. Please use the `dtw_parallel` function instead.

                    Args:
                        use_mmap:
                            If set to `True`, the function uses memory-mapped files to store the distance matrix.
                            This approach is beneficial when dealing with large datasets as it avoids excessive memory usage.
                            However, it may result in a temporary file being created in the working directory.
                        block:
                            The size of the block used to split the computation of the DTW distance matrix.
                            A smaller block size reduces memory usage but may increase computational time.
                        show_progress:
                            If set to `True`, displays a progress bar indicating the computation progress of the distance matrix.

                    Returns:
                        np.ndarray
                            A 1-D array representing the upper triangular part of the computed DTW distance matrix.
                            The matrix is compacted into a 1-D array where each entry represents the distance between
                            a pair of time series.

                    """
        
        events = self.events
        
        traces = [np.array(t) for t in events[trace_column].tolist()]
        N = len(traces)
        
        if not use_mmap:
            distance_matrix = dtw.distance_matrix_fast(
                    traces, use_pruning=use_pruning, parallel=parallel, compact=compact, only_triu=only_triu
                    )
            
            distance_matrix = np.array(distance_matrix)
        
        else:
            
            logging.info("creating mmap of shape ({}, 1)".format(int((N * N - N) / 2)))
            
            tmp = tempfile.TemporaryFile()  # todo might not be a good idea to drop a temporary file in the working directory
            distance_matrix = np.memmap(tmp, dtype=float, mode="w+", shape=(int((N * N - N) / 2)))
            
            iterator = range(0, N, block) if not show_progress else tqdm(range(0, N, block), desc="distance matrix:")
            
            i = 0
            for x0 in iterator:
                x1 = min(x0 + block, N)
                
                dm_ = dtw.distance_matrix_fast(
                        traces, block=((x0, x1), (0, N)), use_pruning=False, parallel=parallel, compact=True,
                        only_triu=only_triu
                        )
                
                distance_matrix[i:i + len(dm_)] = dm_
                distance_matrix.flush()
                
                i += len(dm_)
                
                del dm_
        
        if not compact:
            distance_matrix = dtw.distances_array_to_matrix(distance_matrix, N, block=None, only_triu=only_triu)
        
        distance_matrix = self._fix_invalid_distance_matrix(distance_matrix=distance_matrix, traces=traces,
                                                            max_tries=max_tries)
        
        if return_similarity:
            distance_matrix = self.distance_to_similarity(distance_matrix)
        
        return distance_matrix
    
    @wrapper_local_cache
    def get_dtw_parallel_correlation(
            self, local_dissimilarity: Literal[
                "square_euclidean_distance", "gower", "norm1", "norm2", "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jensenshannon", "minkowski", "sqeuclidean"] = "norm1",
            type_dtw: Literal["d", "i"] = "d", constrained_path_search: Literal["itakura", "sakoe_chiba"] = None,
            itakura_max_slope: float = None, sakoe_chiba_radius: int = None, sigma_kernel: int = 1,
            dtw_to_kernel: bool = False, multivariate: bool = True, use_mmap: bool = False, trace_column: str = "trace"
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        """
                Computes the dynamic time warping (DTW) correlation between time series, with various options to customize the
                computation. This function is particularly useful for analyzing time series data with options to handle
                multivariate series, apply constraints, and transform DTW distances into kernel-based similarities. For more
                information see Escudero-Arnanz et al. 2023 [#escudero_arnanz_2023]_.


                Args:
                    sigma_kernel:
                        Sets the width of the exponential kernel used in transforming the DTW distance matrix into a similarity matrix.
                        The parameter sigma controls the rate at which similarity values decrease with increasing DTW distances.
                        A smaller sigma value leads to a rapid decrease in similarity for small increases in distance,
                        creating a sharper distinction in the similarity matrix. Conversely, a larger sigma value results in a more
                        gradual change, which can be useful when smaller DTW distance differences are significant.
                        The choice of sigma should be based on the specific application and the DTW distance scale.
                    dtw_to_kernel:
                        When set to `True`, activates the conversion of the DTW distance matrix into a similarity matrix through
                        an exponential kernel transformation. This transformation applies the exponential kernel function to each
                        entry in the distance matrix, converting distances into similarity values between 0 and 1.
                        Higher similarity (closer to 1) indicates lower distance, and vice versa. This transformation is useful
                        for tasks where similarity measures are more intuitive or beneficial than distance measures, such as in clustering,
                        classification, or visualization. The transformation adds computational overhead and should be considered
                        when dealing with large datasets or when computational efficiency is a priority.
                    type_dtw:
                        Specifies the type of DTW scheme to use for multivariate time series (MTS) based on Shokoohi-Yekta et al. 2017 [#shokoohi_yekta_2017]_.
                        The options are:
                        .. list-table:: Multivariate time series DTW schemes
                            :widths: 15 15
                            :header-rows: 1
                            * - Scheme name
                              - Description
                            * - "d" (Dependent DTW)
                              - In this scheme, DTW is calculated in a way similar to the single-dimensional DTW, but it redefines the distance calculation to consider the cumulative squared Euclidean distances across multiple data points of the MTS, instead of a single data point. This approach treats all dimensions of the MTS collectively, considering their cumulative behavior.
                            * - "i" (Independent DTW)
                              - This variant computes the DTW distances for all dimensions independently. Each dimension is treated separately, allowing for individual alignment without considering the interdependence among them. This method is suitable when the dimensions of the MTS are independent, or when analyzing each dimension separately is more meaningful.

                    constrained_path_search:
                        Specifies the method for constraining the search path in dynamic time warping calculations.
                        This parameter offers the option to choose between two well-known constraint schemes:
                        the Itakura parallelogram and the Sakoe-Chiba band. Constraining the path search is advantageous
                        for several reasons: it reduces computational overhead by limiting the area of the cost matrix that
                        needs to be explored, prevents non-intuitive and overly flexible warping, and can lead to more
                        meaningful alignments in specific application contexts. For instance, when prior knowledge about
                        the expected warping in the sequences is available, these constraints can significantly enhance
                        the efficiency and relevance of the DTW analysis. The choice between 'itakura' and 'sakoe_chiba'
                        depends on the specific requirements of the analysis and the nature of the data. Please refer to
                        `itakura_max_slope` and `sakoe_chiba_radius` for more information.
                        .. warning:

                            When using `itakura` to constrain ragged arrays an error might occur when `itakaru_max_slope=None`.
                            Try choosing a large `itakaru_max_slope` to test if this fixes the error.
                    itakura_max_slope
                        The maximum slope for the Itakura parallelogram, used as a constraint in dynamic time warping calculations.
                        This parameter controls the slope of the parallelogram sides, effectively constraining the DTW path
                        within a parallelogram-shaped area. A smaller value results in a more restricted warping path, while a larger
                        value allows for more flexibility in the path. Setting it to `None` removes this constraint, allowing for a full
                        search within the cost matrix. The Itakura parallelogram is beneficial for reducing computational overhead
                        and preventing non-intuitive, extreme warping. It is particularly effective when there is prior knowledge
                        about the maximum expected warping between the sequences being compared. Itakura et al. [#itakura_1975]_.
                    sakoe_chiba_radius:
                        The radius of the Sakoe-Chiba band, used as a constraint in dynamic time warping calculations.
                        This parameter specifies the maximum allowed deviation from the diagonal path in the DTW cost matrix.
                        A smaller value of the radius results in less flexibility for time series to warp, while a larger value
                        allows more warping. Setting it to `None` applies no constraint, allowing the warping path to explore
                        the full matrix. The Sakoe-Chiba band is particularly useful for reducing computational complexity
                        and avoiding overly flexible warping, which can lead to non-intuitive alignments. It's generally set
                        as a percentage of the time series length. As per Keogh and Ratanamahatana (2004), a common setting
                        is 10% of the sequence length, which has been found to be effective across various applications.
                        Sakoe 1978 et al. [#sakoe_1978]_.
                    local_dissimilarity:
                        Specifies the method to compute local dissimilarity between two vectors in dynamic time warping.
                        This parameter allows selecting from a range of distance functions, each of which quantifies the
                        dissimilarity between two data points in a distinct manner. The choice of function can significantly
                        influence the warping path and is typically selected based on the nature of the data and the specific
                        requirements of the analysis. The default setting is 'norm1' (Manhattan distance), but other options
                        provide flexibility for different scenarios, like handling mixed data types, scaling issues, or specific
                        geometric interpretations of distance.
                        .. list-table:: Local dissimilarity options
                            :widths: 15 15
                            :header-rows: 1
                        * - Distance measure
                          - Description
                        * - square_euclidean_distance
                          - Compute the squared Euclidean distance, useful for performance optimization as it avoids the square root calculation.
                        * - gower
                          - Compute the Gower distance, typically used for mixed data types or when variables are on different scales.
                        * - norm1 (Manhattan Distance)
                          - Compute the Manhattan (L1 norm) distance, which sums the absolute differences of their coordinates. Useful in grid-like pathfinding.
                        * - norm2 (Euclidean Distance)
                          - Compute the Euclidean (L2 norm) distance, representing the shortest distance between points. Common in spatial distance calculations.
                        * - braycurtis
                          - Compute the Bray-Curtis distance, a measure of dissimilarity based on the sum of absolute differences. Often used in ecological data analysis.
                        * - canberra
                          - Compute the Canberra distance, a weighted version of Manhattan distance, sensitive to small changes near zero. Useful in numerical analysis.
                        * - chebyshev
                          - Compute the Chebyshev distance, the maximum absolute difference along any coordinate dimension. Useful in chess-like applications.
                        * - cityblock (Manhattan Distance)
                          - Compute the City Block (Manhattan) distance, similar to norm1, effective in grid-based pathfinding.
                        * - correlation
                          - Compute the correlation distance, a measure of similarity between two variables based on their correlation coefficient.
                        * - cosine
                          - Compute the Cosine distance, measuring the cosine of the angle between vectors. Useful in text analysis and similarity of preferences.
                        * - euclidean
                          - Computes the Euclidean distance, the straight-line distance between two points in Euclidean space.
                        * - jensenshannon
                          - Compute the Jensen-Shannon distance, a method of measuring the similarity between two probability distributions.
                        * - minkowski
                          - Compute the Minkowski distance, a generalized metric that includes others (like Euclidean, Manhattan) as special cases.
                        * - sqeuclidean
                          - Compute the squared Euclidean distance, similar to square_euclidean_distance, often used in optimization problems.
                    use_mmap: Currently not implemented for this function.
                    multivariate:
                        Indicates whether the DTW algorithm should be applied in a multivariate context.
                        When set to `True`, the algorithm is designed to handle multivariate time series,
                        where each time series consists of multiple parallel sequences or dimensions.
                        .. caution:

                            Caution should be exercised when setting this parameter to `False`, as the algorithm
                            may not function as expected with univariate data or may lead to unexpected results.
                            While this parameter is included for completeness, its modification is generally not recommended
                            without a clear understanding of the specific context and requirements of the analysis. Users are
                            advised to leave this parameter set to `True` unless there is a compelling reason to alter it,
                            and thorough testing should be conducted if a change is made.

                Returns:
                    np.ndarray: The DTW distance matrix or a tuple of the distance matrix and the similarity matrix.

                .. rubric:: Footnotes

                .. [#itakura_1975] Itakura, F., 1975. Minimum prediction residual principle applied to speech recognition. IEEE Transactions on acoustics, speech, and signal processing, 23(1), pp.67-72.
                .. [#sakoe_1978] Sakoe, H. and Chiba, S., 1978. Dynamic programming algorithm optimization for spoken word recognition. IEEE transactions on acoustics, speech, and signal processing, 26(1), pp.43-49.
                .. [#shokoohi_yekta_2017] Shokoohi-Yekta, M., Hu, B., Jin, H., Wang, J. and Keogh, E., 2017. Generalizing DTW to the multi-dimensional case requires an adaptive approach. Data mining and knowledge discovery, 31, pp.1-31.
                .. [#escudero_arnanz_2023] Óscar Escudero-Arnanz, Antonio G. Marques, Cristina Soguero-Ruiz, Inmaculada Mora-Jiménez, Gregorio Robles, dtwParallel: A Python package to efficiently compute dynamic time warping between time series, SoftwareX, Volume 22, 2023, 101364, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2023.101364.

                """
        
        from dtwParallel import dtw_functions
        
        events = self.events
        
        traces = np.array(events[trace_column].tolist())
        
        if use_mmap:
            logging.warning(f"Currently the use of mmap files is not implemented.")
        
        if local_dissimilarity in ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
                                   "euclidean", "jensenshannon", "minkowski", "sqeuclidean"]:
            from scipy.spatial import distance
            
            try:
                # Retrieve the function from the scipy.spatial.distance module
                local_dissimilarity = getattr(distance, local_dissimilarity)
            
            except AttributeError:
                # Handle the case where the function name does not exist in the module
                raise ValueError(
                        f"Distance function '{local_dissimilarity}' is not available in scipy.spatial.distance."
                        )
        
        if not use_mmap:
            
            class Input:
                def __init__(self):
                    self.check_errors = False
                    self.type_dtw = type_dtw
                    self.constrained_path_search = constrained_path_search
                    self.MTS = multivariate
                    self.regular_flag = False
                    self.n_threads = -1
                    self.local_dissimilarity = local_dissimilarity
                    self.visualization = False
                    self.output_file = False
                    self.dtw_to_kernel = dtw_to_kernel
                    self.sigma_kernel = sigma_kernel
                    self.itakura_max_slope = itakura_max_slope
                    self.sakoe_chiba_radius = sakoe_chiba_radius
            
            input_obj = Input()
            
            distance_matrix = dtw_functions.dtw_tensor_3d(traces, traces, input_obj)
        
        else:
            
            """
            logging.info("creating mmap of shape ({}, 1)".format(int((N * N - N) / 2)))

            tmp = tempfile.TemporaryFile()  # todo might not be a good idea to drop a temporary file in the working directory
            distance_matrix = np.memmap(tmp, dtype=float, mode="w+", shape=(int((N * N - N) / 2)))

            iterator = range(0, N, block) if not show_progress else tqdm(range(0, N, block), desc="distance matrix:")

            i = 0
            for x0 in iterator:
                x1 = min(x0 + block, N)

                dm_ = dtw.distance_matrix_fast(
                    traces, block=((x0, x1), (0, N)), use_pruning=False, parallel=True, compact=True, only_triu=True
                )

                distance_matrix[i:i + len(dm_)] = dm_
                distance_matrix.flush()

                i = i + len(dm_)

                del dm_
            """
            
            raise NotImplementedError
        
        return distance_matrix
    
    @staticmethod
    def _clean_distance_matrix(distance_matrix: np.ndarray, replace_value: Union[Literal['max'], float, int] = 'max'):
        
        x, y = np.where(np.isnan(distance_matrix))
        xx, yy = np.where(np.isinf(distance_matrix))
        
        x = np.concatenate([x, xx])
        y = np.concatenate([y, yy])
        
        if replace_value == 'max':
            replace_value = np.nanmax(distance_matrix[x, y]) + 1
        
        for i in range(len(x)):
            
            x_ = x[i]
            y_ = y[i]
            
            distance_matrix[x_, y_] = replace_value
        
        return distance_matrix
    
    def get_correlation(
            
            self, events=None, correlation_type: Literal['pearson', 'dtw', 'dtw_parallel'] = "pearson",
            correlation_param: dict = None, trace_column: str = "trace",
            clean_matrix: bool = True, clean_replace_value: Union[Literal['max'], float, int] = 'max',
            ):
        
        if events is None:
            events = self.events
        
        if events is None:
            raise ValueError(f"Please provide data using the 'events' flag.")
        
        if correlation_param is None:
            correlation_param = {}
        
        correlation_param["trace_column"] = trace_column
        
        funcs = {"pearson":      self.get_pearson_correlation,
                 "dtw":          self.get_dtw_correlation,
                 "dtw_parallel": self.get_dtw_parallel_correlation}
        
        if correlation_type not in funcs.keys():
            raise ValueError(f"cannot find correlation type. Choose one of: {funcs.keys()}")
        else:
            corr_func = funcs[correlation_type]
        
        corr = corr_func(**correlation_param)
        
        if clean_matrix:
            self.logger.info(f"cleaning distance matrix")
            corr = self._clean_distance_matrix(corr, replace_value=clean_replace_value)
        
        return corr
    
    def _get_correlation_histogram(
            self, corr=None, events=None, correlation_type="pearson", correlation_param={}, start=-1, stop=1,
            num_bins=1000, density=False
            ):
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
    
    def get_spatial_tree(self, leafsize: int = 10, compact_nodes: bool = True, copy_data: bool = False,
                         balanced_tree: bool = True):
        
        events = self.events
        cx = events.cx.values
        cy = events.cy.values
        
        data = np.array([cx, cy], dtype=float).transpose()
        
        kdtree = KDTree(data=data, leafsize=leafsize, compact_nodes=compact_nodes, copy_data=copy_data,
                        balanced_tree=balanced_tree)
        return kdtree
    
    def plot_correlation_characteristics(
            self, corr=None, events=None, ax=None, perc=(5e-5, 5e-4, 1e-3, 1e-2, 0.05), bin_num=50, log_y=True,
            figsize=(10, 3)
            ):
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


class Modules(CachedClass):
    
    def __init__(self, events, cache_path=None):
        
        if cache_path is None:
            cache_path = events.cache_path
        
        super().__init__(cache_path=cache_path)
        
        if events._is_multi_subject():
            logging.warning(
                    "multiple values for 'subject_id' were found in the events table. "
                    "The module class expects all events to belong to a single recording."
                    )
        
        self.events = events
    
    def __hash__(self):
        return hash(self.events)
    
    @wrapper_local_cache
    def _create_node_edge_tables(self, correlation: np.ndarray, correlation_boundaries=(0.98, 1),
                                 max_distance: float = None):
        
        if isinstance(self.events, Events):
            events = self.events.events
        else:
            events = self.events
        
        if len(events) != len(correlation):
            raise ValueError(f"Number of dimensions ({correlation.shape}) does not match events length ({len(events)})")
        
        # select correlations within given boundaries
        lower_bound, upper_bound = correlation_boundaries
        selected_correlations = np.where(np.logical_and(correlation >= lower_bound, correlation < upper_bound))
        
        # deal with compact correlation matrices
        if len(selected_correlations) == 1:
            raise NotImplementedError(f"Node edge table has to be square matrix")
            
            # selected_correlations = selected_correlations[0]
            #
            # if len(selected_correlations.shape) == 1:
            #     triu_indices = np.array(np.triu_indices(len(events)))
            #     selected_correlations = triu_indices[:, selected_correlations].squeeze()
        
        selected_x_corr, selected_y_corr = selected_correlations
        logging.info(f"Remaining connections ({lower_bound:.2f} <= c < {upper_bound:.2f}): "
                     f"{len(selected_x_corr)}/{len(events) ** 2} "
                     f"({len(selected_x_corr) / len(events) ** 2 * 100:.2f}%)")
        
        if max_distance is not None:
            
            dist = Distance(events)
            tree = dist.get_spatial_tree()
            pairs = tree.query_pairs(r=max_distance)
            
            x_corr, y_corr = [], []
            for idx_x, idx_y in tqdm(list(zip(selected_x_corr, selected_y_corr)), desc='Filter by distance'):
                if (idx_x, idx_y) in pairs:
                    x_corr.append(idx_x)
                    y_corr.append(idx_y)
            
            logging.info(f"Remaining connections (distance <= {max_distance}): "
                         f"{len(x_corr)}/{len(selected_x_corr) ** 2} "
                         f"({len(x_corr) / len(selected_x_corr) ** 2 * 100:.2f}% / "
                         f"{len(x_corr) / len(events) ** 2 * 100:.2f}%)")
            
            selected_x_corr, selected_y_corr = np.array(x_corr), np.array(y_corr)
        
        # create nodes table
        selected_xy_corr = np.unique(np.concatenate([selected_x_corr, selected_y_corr], axis=0))
        selected_events = events.iloc[selected_xy_corr]
        nodes = pd.DataFrame(
                {"i_idx":     selected_xy_corr,
                 "x":         selected_events.cx, "y": selected_events.cy,
                 "trace_idx": selected_events.index}
                )
        
        # create edges table
        edges = pd.DataFrame({"source": selected_x_corr, "target": selected_y_corr})
        
        # convert edge indices from iidx in correlation array
        # to row_idx in nodes table
        lookup = dict(zip(nodes.i_idx.tolist(), nodes.trace_idx.tolist()))
        edges.source = edges.source.map(lookup)
        edges.target = edges.target.map(lookup)
        
        return nodes, edges
    
    @wrapper_local_cache
    def create_graph(self, correlation, correlation_boundaries=(0.98, 1), exclude_out_of_cluster_connection=True,
                     return_graph: bool = False, color_palette: str = None, palette=None, max_distance: float = None):
        
        nodes, edges = self._create_node_edge_tables(correlation, correlation_boundaries=correlation_boundaries,
                                                     max_distance=max_distance)
        logging.info(f"#nodes: {len(nodes)}, #edges: {len(edges)}")
        
        if len(nodes) < 1:
            logging.warning(f"No nodes were recovered. Aborting graph creation")
            
            if return_graph:
                return None, None, None
            else:
                return None, None, None
        
        # create graph and populate with edges
        G = nx.Graph()
        for _, edge in tqdm(edges.iterrows(), total=len(edges)):
            G.add_edge(*edge)
        
        # calculate modularity
        communities = community.greedy_modularity_communities(G)
        
        # assign modules
        nodes["module"] = -1
        n_mod = 0
        for module in tqdm(communities, desc='Updating modules'):
            for m in module:
                nodes.loc[m, "module"] = n_mod
            n_mod += 1
        
        # add color
        if color_palette or palette is not None:
            from astrocast.analysis import Plotting
            
            p = Plotting()
            lut = p.get_group_color(df=nodes, palette=color_palette, group_column="module")
            nodes["color"] = nodes.module.map(lut)
            
            for idx, node in nodes.iterrows():
                if not isinstance(node.color, tuple):
                    nodes[idx] = "black"
        
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
        
        if color_palette or palette is not None:
            edges["color"] = edges.module.map(lut)
            
            for idx, edge in edges.iterrows():
                if not isinstance(edge.color, tuple):
                    edges[idx] = "black"
        
        if return_graph:
            return G, nodes, edges, lookup_cluster_table
        else:
            return nodes, edges, lookup_cluster_table
    
    @staticmethod
    def summarize_modules(nodes):
        
        from pointpats import centrography
        
        summary = {}
        
        funcs = {"mean_center":     lambda module: None if len(module) < 1 else centrography.mean_center(
                module[["x", "y"]].astype(float)
                ), "median_center": lambda module: None if len(module) < 1 else centrography.euclidean_median(
                module[["x", "y"]].astype(float)
                ), "std_distance":  lambda module: None if len(module) < 1 else centrography.std_distance(
                module[["x", "y"]].astype(float)
                ), "coordinates":   lambda module: module[["x", "y"]].astype(float).values,
                 "num_events":      lambda module: len(module), }
        
        for func_key in funcs.keys():
            func = funcs[func_key]
            summary[func_key] = nodes.groupby("module").apply(func)
        
        return pd.DataFrame(summary)
    
    @staticmethod
    def plot_correlation_heatmap(correlation: np.ndarray, correlation_boundary: Tuple[float, float] = None,
                                 figsize=(7, 5), ax: plt.Axes = None, **kwargs):
        
        if correlation_boundary is not None:
            masked_corr = np.zeros(correlation.shape)
            
            lower_bound, upper_bound = correlation_boundary
            indices = np.where(np.logical_and(lower_bound <= correlation, correlation < upper_bound))
            masked_corr[indices] = correlation[indices]
            
            logging.info(f"{len(indices[0])}/{len(masked_corr)}")
        
        else:
            masked_corr = correlation
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        sns.heatmap(data=masked_corr, ax=ax, **kwargs)
    
    @staticmethod
    def plot_network(G: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame,
                     node_size=25, font_size=8, width=2,
                     node_color="blue", edge_color="black",
                     show_grid: bool = False, with_labels: bool = False,
                     figsize=(10, 10), ax: plt.Axes = None, **kwargs):
        
        pos = dict(zip(nodes.index, [(row.x, row.y) for _, row in nodes.iterrows()]))
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if edge_color == "edge":
            edge_color = edges.color
        
        nx.draw_networkx(G, pos=pos,
                         node_size=node_size, font_size=font_size, width=width,
                         node_color=node_color, edge_color=edge_color, with_labels=with_labels,
                         ax=ax, **kwargs)
        
        if not show_grid:
            ax.grid(None)


class Discriminator(CachedClass):
    
    def __init__(self, events, cache_path=None):
        super().__init__(cache_path=cache_path)
        
        self.events = events
        self.X_test = None
        self.Y_test = None
        self.X_train = None
        self.Y_train = None
        self.indices_train = None
        self.indices_test = None
        self.clf = None
    
    @staticmethod
    def get_available_models():
        
        available_models = []
        for module in [cluster, ensemble, gaussian_process, linear_model, neighbors, neural_network, tree]:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    available_models.append(name)
        
        return available_models
    
    def train_classifier(
            self, embedding=None, category_vector=None, indices=None, split=0.8, classifier="RandomForestClassifier",
            balance_training_set: bool = False, balance_test_set: bool = False, **kwargs
            ):
        
        # split into training and validation dataset
        if self.X_train is None or self.Y_train is None:
            self.split_dataset(embedding, category_vector, split=split, indices=indices,
                               balance_training_set=balance_training_set, balance_test_set=balance_test_set)
        
        # available models
        available_models = self.get_available_models()
        if classifier not in available_models:
            raise ValueError(f"unknown classifier {classifier}. Choose one of {available_models}")
        
        # fit model
        clf = None
        for module in [cluster, ensemble, gaussian_process, linear_model, neighbors, neural_network, tree]:
            
            try:
                class_ = getattr(module, classifier, None)
                clf = class_(**kwargs)
            except TypeError:
                pass
            except Exception as err:
                print(f"cannot load from {module} with error: {err}\n\n")
                traceback.print_exc()
                pass
        
        if clf is None:
            raise ValueError(f"could not load classifier. Please try another one")
        
        clf.fit(self.X_train, self.Y_train)
        
        self.clf = clf
        return clf
    
    def predict(self, X, normalization_instructions=None):
        
        if normalization_instructions is not None:
            norm = Normalization(X, inplace=True)
            norm.run(normalization_instructions)
        
        return self.clf.predict(X)
    
    @staticmethod
    def compute_scores(true_labels, predicted_labels, scoring: Literal['classification', 'clustering'] =
    'classification'):
        """Compute performance metrics between true and predicted labels.

        Args:
          true_labels: Ground truth (correct) labels.
          predicted_labels: Predicted labels, as returned by a classifier.
          scoring: type of scoring

        Returns:
          A dictionary with accuracy, precision, recall, and F1 score.
        """
        
        if scoring == 'classification':
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            return {
                'accuracy':  accuracy_score(true_labels, predicted_labels),
                'precision': precision_score(true_labels, predicted_labels, average='macro'),
                'recall':    recall_score(true_labels, predicted_labels, average='macro'),
                'f1':        f1_score(true_labels, predicted_labels, average='macro')
                }
        
        elif scoring == 'clustering':
            
            from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, rand_score
            
            return {
                'adjusted_mutual_info_score': adjusted_mutual_info_score(true_labels, predicted_labels),
                'adjusted_rand_score':        adjusted_rand_score(true_labels, predicted_labels),
                'homogeneity_score':          homogeneity_score(true_labels, predicted_labels),
                'rand_score':                 rand_score(true_labels, predicted_labels)
                }
        
        else:
            raise ValueError(f"for scoring choose one of: classifcation, clustering")
    
    def evaluate(self, regression=False, cutoff=0.5, normalize=None, show_plot: bool = True,
                 title: str = None, figsize=(10, 5), axx=None):
        
        fig = None
        if show_plot and axx is None:
            fig, axx = plt.subplots(1, 2, figsize=figsize)
        
        evaluations = {}
        for i, (X, Y, lbl) in enumerate([(self.X_train, self.Y_train, "train"), (self.X_test, self.Y_test, "test")]):
            
            results = {}
            pred = np.squeeze(self.clf.predict(X))
            
            if pred.dtype != int and not regression and cutoff is not None:
                logging.warning(f"assuming probability prediction. thresholding at {cutoff}")
                Y = Y >= cutoff
            
            if regression:
                score = self.clf.score(X, Y)
                evaluations[lbl] = {"score": score}
            
            else:
                
                # create confusion matrix
                cm = confusion_matrix(pred, Y, normalize=normalize)
                results["cm"] = cm
                
                # create score
                for k, v in self.compute_scores(Y, pred).items():
                    results[k] = v
                
                # plot result
                if show_plot:
                    ax = axx[i]
                    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix ({lbl.capitalize()})")
                
                evaluations[lbl] = results
        
        if show_plot and title is not None:
            if fig is None:
                fig = axx[0].get_figure()
            fig.suptitle(title)
        
        return evaluations
    
    def split_dataset(self, embedding, category_vector, indices=None,
                      split=0.8, balance_training_set=False, balance_test_set=False,
                      encode_category=None, normalization_instructions=None
                      ):
        
        # get data
        X = embedding
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # get category to predict
        if isinstance(category_vector, list):
            Y = np.array(category_vector)
        elif isinstance(category_vector, np.ndarray):
            Y = category_vector
        elif isinstance(category_vector, pd.Series):
            Y = category_vector.values
        elif isinstance(category_vector, str):
            if encode_category is None:
                Y = np.array(self.events.events[category_vector].tolist())
            else:
                Y = np.array(self.events.events[category_vector].map(encode_category).tolist())
        else:
            raise ValueError(
                    f"unknown category vector format: {type(category_vector)}. "
                    f"Use one of list, np.ndarray, str"
                    )
        
        # check inputs
        if len(X) != len(Y):
            raise ValueError(
                    f"embedding and events must have the same length: "
                    f"len(embedding)={len(X)} vs. len(events)={len(Y)}"
                    )
        
        if np.sum(np.isnan(X)) > 0:
            raise ValueError(f"embedding cannot contain NaN values.")
        
        # normalize
        if normalization_instructions is not None:
            norm = Normalization(X, inplace=True)
            norm.run(normalization_instructions)
        
        # split X and Y
        split_idx = int(len(X) * split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        
        # ensure correct length of indices
        if indices is None:
            logging.warning(f"Assuming indices from event dataframe.")
            indices = self.events.events.index.tolist()
        elif len(indices) != len(X):
            raise ValueError(f"length of indices ({len(indices)}) does not match length of embedding ({len(X)})")
        
        indices_train, indices_test = indices[:split_idx], indices[split_idx:]
        
        # todo: this is too late, because sometimes Y_test is missing values due to the split in the section before
        # balancing
        if balance_training_set:
            X_train, Y_train, indices_train = self._balance_set(X_train, Y_train, indices_train)
        
        if balance_test_set:
            X_test, Y_test, indices_test = self._balance_set(X_test, Y_test, indices_test)
        
        # cache results
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.indices_train = indices_train
        self.indices_test = indices_test
    
    # @staticmethod
    # def _balance_set(X, Y, indices):
    #
    #     # identify the category with the fewest members
    #     count_category_members = pd.Series(Y).value_counts()
    #     min_category_count = count_category_members.min()
    #
    #     # choose random indices
    #     rand_indices = list()
    #     for category in count_category_members.index.unique():
    #         rand_indices.append(np.random.choice(np.where(Y == category)[0], size=min_category_count, replace=False))
    #
    #     rand_indices = np.array(rand_indices).astype(int)
    #     rand_indices = rand_indices.flatten()
    #
    #     # select randomly chosen rows
    #     X = X[rand_indices]
    #     Y = Y[rand_indices]
    #
    #     indices = np.array(indices)[rand_indices]
    #
    #     return X, Y, indices
    
    @staticmethod
    def _balance_set(X, Y, indices):
        
        X = np.array(X)
        Y = np.array(Y)
        
        unique_classes = np.unique(Y)
        min_class_samples = np.inf
        
        for cls in unique_classes:
            class_count = np.sum(Y == cls)
            if class_count < min_class_samples:
                min_class_samples = class_count
        
        if min_class_samples == np.inf or min_class_samples == 0:
            # Balancing is not feasible; you could raise an error or skip.
            raise ValueError("Balancing not feasible: One or more classes have zero samples.")
        
        balanced_X, balanced_Y, balanced_indices = [], [], []
        for cls in unique_classes:
            cls_indices = np.where(Y == cls)[0]
            chosen_indices = np.random.choice(cls_indices, min_class_samples, replace=False)
            chosen_indices = np.array(chosen_indices).flatten()
            
            balanced_X.append(X[chosen_indices])
            balanced_Y.append(Y[chosen_indices])
            balanced_indices.extend([indices[i] for i in chosen_indices])
        
        # Flattening the lists
        balanced_X = np.vstack(balanced_X)
        balanced_Y = np.concatenate(balanced_Y)
        # Ensure indices match the order of the balanced dataset
        balanced_indices = [indices[i] for i in np.argsort(balanced_Y)]
        
        return balanced_X, balanced_Y, balanced_indices


class CoincidenceDetection:
    
    def __init__(
            self, events, incidences, embedding, train_split=0.8, balance_training_set=False, balance_test_set=False,
            encode_category=None, normalization_instructions=None
            ):
        
        if len(events) != len(embedding):
            raise ValueError(
                    f"Number of events and embedding does not match: "
                    f"n(events):{len(events)} vs. n(embedding): {len(embedding)}"
                    )
        
        self.events = events
        self.incidences = incidences
        self.embedding = embedding
        
        self.train_split = train_split
        self.balance_training_set = balance_training_set
        self.balance_test_set = balance_test_set
        self.encode_category = encode_category
        self.normalization_instructions = normalization_instructions
        
        # align incidences
        self.aligned = self._align_events_and_incidences()
    
    def _align_events_and_incidences(self):
        
        id_event_ = []
        num_events_ = []
        incidence_location_ = []
        incidence_location_relative_ = []
        for i, (idx, row) in enumerate(self.events.events.iterrows()):
            
            num_events = 0
            incidence_location = []
            incidence_location_relative = []
            for incidence in self.incidences:
                
                if row.z0 < incidence < row.z1:
                    num_events += 1
                    incidence_location.append(incidence - row.z0)
                    incidence_location_relative.append((incidence - row.z0) / row.dz)
            
            id_event_.append(i)
            num_events_.append(num_events)
            incidence_location_.append(incidence_location)
            incidence_location_relative_.append(incidence_location_relative)
        
        aligned = pd.DataFrame(
                {"id_event":                    id_event_, "num_incidences": num_events_,
                 "incidence_location":          incidence_location_,
                 "incidence_location_relative": incidence_location_relative_}
                )
        
        return aligned
    
    def _train(
            self, embedding, category_vector, classifier, regression=False, normalize_confusion_matrix=False,
            show_plot: bool = False,
            **kwargs
            ):
        
        discr = Discriminator(self.events)
        
        discr.split_dataset(
                embedding, category_vector, split=self.train_split, balance_training_set=self.balance_training_set,
                balance_test_set=self.balance_test_set, encode_category=self.encode_category,
                normalization_instructions=self.normalization_instructions
                )
        
        clf = discr.train_classifier(
                self, embedding, split=None, classifier=classifier, **kwargs
                )
        
        evaluation = discr.evaluate(cutoff=0.5, normalize=normalize_confusion_matrix, regression=regression,
                                    show_plot=show_plot)
        
        return clf, evaluation
    
    def predict_coincidence(
            self, binary_classification=True, classifier="RandomForestClassifier", normalize_confusion_matrix=None,
            show_plot: bool = False, **kwargs
            ):
        
        aligned = self.aligned.copy()
        aligned = aligned.reset_index()
        
        embedding = self.embedding.copy()
        
        if binary_classification:
            
            category_vector = aligned.num_incidences.apply(lambda x: x >= 1)
            category_vector = category_vector.astype(bool).values
        
        else:
            category_vector = aligned.num_incidences
            category_vector = category_vector.astype(int).values
        
        clf, confusion_matrix = self._train(
                embedding, category_vector, classifier, regression=False,
                normalize_confusion_matrix=normalize_confusion_matrix, show_plot=show_plot,
                **kwargs
                )
        
        return clf, confusion_matrix
    
    def predict_incidence_location(self, classifier="RandomForestRegressor", single_event_prediction=True,
                                   show_plot: bool = False, **kwargs):
        
        aligned = self.aligned.copy()
        aligned = aligned.reset_index()
        
        embedding = self.embedding.copy()
        
        # select event with coincidence
        selected = aligned[aligned.num_incidences > 0]
        
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.iloc[selected.id_event.tolist()].values
        elif isinstance(embedding, np.ndarray):
            embedding = embedding[selected.id_event.tolist()]
        else:
            raise ValueError(f"unknown embedding type: {type(embedding)}")
        
        if single_event_prediction:
            category_vector = selected.incidence_location_relative.apply(lambda x: x[0]).values
        else:
            raise ValueError(f"currently multi event prediction is not implemented.")
        
        clf, score = self._train(
                embedding, category_vector, classifier=classifier, regression=True,
                show_plot=False, **kwargs
                )
        
        return clf, score


class TeraHAC:
    
    def __init__(self, epsilon, threshold, logging_level=logging.INFO):
        self.epsilon = epsilon
        self.threshold = threshold
        
        self.log = MiniLogger(logger_name="TeraHAC", level=logging_level)
    
    def initialize_graph(self, graph):
        for node in graph.nodes:
            graph.nodes[node]['cluster_size'] = 1
            graph.nodes[node]['min_merge_similarity'] = float('inf')
            graph.nodes[node]['max_merge_similarity'] = 0
            graph.nodes[node]['max_weight'] = 0  # Initialize max_weight
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            graph.nodes[u]['max_weight'] = max(graph.nodes[u]['max_weight'], weight)
            graph.nodes[v]['max_weight'] = max(graph.nodes[v]['max_weight'], weight)
        
        return graph

    def _get_subgraph(self, G, method: Literal["connected_components", "louvain", "label_propagation"] = "louvain"):
        subgraphs = []
    
        if method == "louvain":
            import community as community_louvain
            partition = community_louvain.best_partition(G)
    
            for community_ in set(partition.values()):
                nodes = [node for node in partition.keys() if partition[node] == community_]
                subgraphs.append(G.subgraph(nodes).copy())
    
        elif method == "connected_components":
            for c in nx.connected_components(G):
                subgraphs.append(G.subgraph(c).copy())

        elif method == "label_propagation":
            communities = nx.algorithms.community.label_propagation_communities(G)

            for community in communities:
                subgraphs.append(G.subgraph(community).copy())
        
        return subgraphs
    
    def subgraph_hac(self, subgraph):
        pq = []
        for u, v, data in subgraph.edges(data=True):
            weight = data['weight']
            goodness = max(subgraph.nodes[u]['max_weight'], subgraph.nodes[v]['max_weight']) / weight
            heappush(pq, (goodness, u, v))
        
        merges = []
        while pq and pq[0][0] <= 1 + self.epsilon:
            _, u, v = heappop(pq)
            if subgraph.has_edge(u, v):
                # new_node, merge_distance, new_cluster_size = self.merge_clusters(subgraph, u, v)
                # merges.append((u, v, merge_distance))  # Include merge distance
                
                self.merge_clusters(subgraph, u, v)
                merges.append((u, v))
        
        self.log.log(f"Merges performed: {merges}", level=logging.DEBUG)
        return merges
    
    @staticmethod
    def merge_clusters(graph, u, v):
        new_node = max(graph.nodes) + 1  # Ensure new_node is an integer
        graph.add_node(new_node)
        
        graph.nodes[new_node]['cluster_size'] = graph.nodes[u]['cluster_size'] + graph.nodes[v]['cluster_size']
        graph.nodes[new_node]['min_merge_similarity'] = min(graph.nodes[u]['min_merge_similarity'],
                                                            graph.nodes[v]['min_merge_similarity'],
                                                            graph.edges[u, v]['weight'])
        graph.nodes[new_node]['max_merge_similarity'] = max(graph.nodes[u]['max_merge_similarity'],
                                                            graph.nodes[v]['max_merge_similarity'],
                                                            graph.edges[u, v]['weight'])
        
        # Calculate max weight
        max_weight = 0
        for neighbor in list(graph.neighbors(u)) + list(graph.neighbors(v)):
            # if neighbor not in {u, v} and graph.has_edge(u, neighbor) and graph.has_edge(v, neighbor):
            if neighbor not in {u, v}:
                
                weight = 0
                if graph.has_edge(u, neighbor):
                    weight = max(weight, graph[u][neighbor]['weight'])
                if graph.has_edge(v, neighbor):
                    weight = max(weight, graph[v][neighbor]['weight'])
                
                graph.add_edge(new_node, neighbor, weight=weight)
                max_weight = max(max_weight, weight)
        
        graph.nodes[new_node]['max_weight'] = max_weight  # Update max_weight for new_node
        
        # # Calculate the average linkage distance
        # total_weight = graph.edges[u, v]['weight']
        # count = 1
        #
        # for neighbor in list(graph.neighbors(u)) + list(graph.neighbors(v)):
        #     if neighbor not in {u, v}:
        #         if graph.has_edge(u, neighbor):
        #             weight = graph[u][neighbor]['weight']
        #             total_weight += weight
        #             count += 1
        #         if graph.has_edge(v, neighbor):
        #             weight = graph[v][neighbor]['weight']
        #             total_weight += weight
        #             count += 1
        #         graph.add_edge(new_node, neighbor, weight=weight)
        #
        # # Update the distance for the new cluster
        # merge_distance = total_weight / count
        #
        # graph.nodes[new_node]['max_weight'] = merge_distance  # This is now the merge distance
        
        graph.remove_node(u)
        graph.remove_node(v)
        
        return new_node
        
        # return new_node, merge_distance, new_cluster_size
    
    def run_terahac(self, similarity_matrix: np.ndarray, similarity_threshold: float = 0.5,
                    stop_after=10e6, parallel: bool = False,
                    zero_similarity_decrease: float = 0.9,
                    subgraph_approach: Literal["connected_components", "louvain", "label_propagation"] = "label_propagation",
                    distance_conversion: Literal["inverse", "reciprocal", "inverse_logarithmic"] = None,
                    plot_intermediate: bool = False, n_colors=10, color_palette="pastel"):
                        
        graph = self.create_similarity_graph(similarity_matrix, threshold=similarity_threshold)
        self.log.log(f"Created graph from similarity matrix.")
        
        graph = self.initialize_graph(graph)
        self.log.log(f"Initialized graph.")
        
        if plot_intermediate:
            color_palette = sns.color_palette(color_palette, n_colors)
            palette = itertools.cycle(color_palette)
        
        counter = 0
        linkage_matrix = []
        pruned = []
        # dendrogram_nodes = []
        # last_merge_distance = 0
        while graph.number_of_edges() > 0 and counter < stop_after:
            
            self.log.log(f"#nodes: {len(graph.nodes):,d} #edges: {len(graph.edges):,d}", level=logging.INFO)
            
            if plot_intermediate:
                fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 3))
                node_color = next(palette)
                self.plot_graph(graph, title=f"full", ax=[ax0], iteration=counter, color=node_color)
            
            # todo: replace by Bateni et al. affinity clustering algorithm
            subgraphs = self._get_subgraph(graph, method = subgraph_approach)
            self.log.log(f"#{counter}: Collected {len(subgraphs)} sub graphs.")
            
            if plot_intermediate:
                self.plot_graph(subgraphs, title=f"SG", color=node_color)
            
            # Merge sub graphs
            merges = []
            if not parallel:
                for subgraph in tqdm(subgraphs):
                    merges.extend(self.subgraph_hac(subgraph))
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.subgraph_hac, subgraph) for subgraph in subgraphs]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        merges.extend(future.result())
            
            self.log.log(f"#{counter}: Performed {len(merges)} merges.")
            
            # for u, v, merge_distance in merges:
            for u, v in merges:
                if graph.has_edge(u, v):
                    
                    n = self.merge_clusters(graph, u, v)
                    
                    new_node = graph.nodes[n]
                    linkage_matrix.append([float(u), float(v), new_node["max_weight"], new_node["cluster_size"]])
            
            if plot_intermediate:
                self.plot_graph(graph, title=f"post merge", ax=[ax1], iteration=counter, color=node_color)
            
            # Prune nodes
            to_prune = []
            eps_threshold = self.threshold / (1 + self.epsilon)
            for n in graph.nodes:
                node = graph.nodes[n]
                edges = graph.edges(n)
                
                if len(edges) < 1 or node['max_weight'] < eps_threshold:
                    to_prune.append(n)
            
            pruned += to_prune
            graph.remove_nodes_from(to_prune)
            
            if len(to_prune) > 0:
                self.log.log(f"#{counter}: Pruned {len(to_prune)} nodes.")
            
            # Break loop
            
            if plot_intermediate:
                self.plot_graph(graph, title=f"post remove", ax=[ax2], iteration=counter, color=node_color)
            
            if len(merges) == 0:
                self.log.log("No more merges are possible.", level=logging.INFO)
                break
            
            counter += 1
        
        if counter >= stop_after:
            self.log.log(f"Prematurely stopped run after {counter} iterations.", level=logging.WARNING)
        
        # Final merge step to ensure all nodes are merged
        # remaining_nodes = list(graph.nodes)
        # while len(remaining_nodes) > 1:
        #     new_node, merge_distance, new_cluster_size = self.merge_clusters(graph, remaining_nodes[0],
        #                                                                      remaining_nodes[1])
        #
        #     # %todo -->
        #
        #     merge_distance += last_merge_distance
        #
        #     # % todo <--
        #
        #     dendrogram_nodes.append((remaining_nodes[0], remaining_nodes[1], merge_distance, new_cluster_size))
        #     remaining_nodes = list(graph.nodes)
        #
        #     # %todo -->
        #
        #     last_merge_distance = merge_distance
        #
        #     # % todo <--
        
        if plot_intermediate:
            self.plot_graph(graph, title=f"Final")
        
        # convert linkage_matrix and remove zero values
        linkage_matrix = np.array(linkage_matrix, dtype=float)
        
        if zero_similarity_decrease is not None:
            distances = linkage_matrix[:, 2]
            mask = np.ones(distances.shape, dtype=bool)
            zero_entries = np.where(distances == 0)[0]
            mask[zero_entries] = 0
            
            new_min = np.min(linkage_matrix[mask, 2]) / 2 * zero_similarity_decrease
            
            for ze in zero_entries:
                linkage_matrix[ze, 2] = new_min
                new_min *= zero_similarity_decrease
        else:
            new_min = 0
        
        # add pruned
        skipped = 0
        if len(pruned) > 0:
            c_max = len(similarity_matrix) + len(linkage_matrix) - 1
            for i, pr in enumerate(pruned):
                
                if np.any(np.isin(linkage_matrix[:, :-2], [pr])):
                    skipped += 1
                    continue
                
                fix = [[pr, c_max + i - skipped, new_min, 0]]
                fix = np.array(fix, dtype=float)
                linkage_matrix = np.concatenate([linkage_matrix, fix])
                new_min *= zero_similarity_decrease
        
        # add missing
        all_clusters = np.unique(linkage_matrix[:, 0:2])
        c_max = len(similarity_matrix) + len(linkage_matrix) - 1
        should_exist = np.arange(0, c_max)
        
        missing_indices = np.isin(should_exist, all_clusters, invert=True)
        missing = should_exist[missing_indices]
        skipped = 0
        if len(missing) > 0:
            for i, m in enumerate(missing):
                
                if np.any(np.isin(linkage_matrix[:, :-2], [m])):
                    skipped += 1
                    continue
                
                fix = [[m, c_max + i - skipped, new_min, 0]]
                fix = np.array(fix, dtype=float)
                linkage_matrix = np.concatenate([linkage_matrix, fix])
                new_min *= zero_similarity_decrease
        
        # convert to distance
        if distance_conversion == "inverse":
            linkage_matrix[:, 2] = 1 - linkage_matrix[:, 2]
        elif distance_conversion == "reciprocal":
            linkage_matrix[:, 2] = 1 / linkage_matrix[:, 2]
        elif distance_conversion == "inverse_logarithmic":
            linkage_matrix[:, 2] = - np.log(linkage_matrix[:, 2])
        elif distance_conversion is None:
            pass
        else:
            self.log.log(f"Incorrect 'distance_conversion' input ({distance_conversion}. Must be one of "
                         f"inverse, reciprocal, inverse_logarithmic")
        
        # linkage_matrix = np.array(linkage_matrix, dtype=float)
        # # eps_threshold = self.threshold / (1 + self.epsilon)
        # # max_dist = similarity_threshold + eps_threshold
        # print(f"pruned: {pruned}")
        # for pr in pruned:
        #
        #     u = pr
        #     v = np.max(linkage_matrix[:, 0:2]) + 1
        #     sim = 0
        #     cluster_size = 0
        #
        #     row = np.array([[u, v, -1, sim, cluster_size]])
        #
        #     linkage_matrix = np.concatenate((linkage_matrix, row), axis=0)
        #     # max_dist += eps_threshold
        #
        return graph, linkage_matrix
        
        # self.log.log(f"Final dendrogram nodes: {dendrogram_nodes}", level=logging.DEBUG)
        #
        # # Convert dendrogram nodes to linkage matrix
        # linkage_matrix = self.convert_to_linkage_matrix(dendrogram_nodes, len(graph.nodes))
        #
        # return linkage_matrix
    
    @staticmethod
    def distance_to_similarity(distance_matrix, method='inverse', sigma=1.0):
        if method == 'inverse':
            similarity_matrix = 1 / distance_matrix
        elif method == 'gaussian':
            similarity_matrix = np.exp(-distance_matrix ** 2 / (2.0 * sigma ** 2))
        else:
            raise ValueError("Unknown method: choose 'inverse' or 'gaussian'")
        return similarity_matrix
    
    def create_similarity_graph(self, similarity_matrix, threshold=None, k=None):
        n = similarity_matrix.shape[0]
        G = nx.Graph()
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if threshold is not None and similarity >= threshold:
                    G.add_edge(i, j, weight=similarity)
                elif k is not None:
                    sorted_indices = np.argsort(similarity_matrix[i])[::-1]
                    for idx in sorted_indices[:k]:
                        if idx != i:
                            G.add_edge(i, idx, weight=similarity_matrix[i, idx])
        
        total_edges = (n ** 2 - n) / 2
        self.log.log(f"retained edges: {len(G.edges) / total_edges * 100:.1f}%")
        
        return G
    
    @staticmethod
    def convert_to_linkage_matrix(dendrogram_nodes, num_points):
        """
        Convert dendrogram nodes to a linkage matrix.

        Args:
        dendrogram_nodes (list of tuples): List of merges in the format (node1, node2, distance, cluster_size).
        num_points (int): Number of original data points.

        Returns:
        np.ndarray: Linkage matrix.
        """
        linkage_matrix = []
        current_cluster_index = num_points
        
        for node1, node2, distance, cluster_size in dendrogram_nodes:
            linkage_matrix.append([node1, node2, distance, cluster_size])  # Include cluster size
            current_cluster_index += 1
        
        return np.array(linkage_matrix, dtype=np.double)
    
    @staticmethod
    def plot_graph(G, ax=None, title: str = None, iteration: int = None, color="#1f78b4"):
        
        if not isinstance(G, list):
            G = [G]
        
        if ax is None:
            fig, axx = plt.subplots(1, len(G), figsize=(3 * len(G), 3))
        else:
            axx = ax
        
        if not isinstance(axx, (np.ndarray, list)):
            axx = [axx]
        
        for i, g in enumerate(G):
            
            ax = axx[i]
            
            layout = nx.spring_layout(g)
            
            for edge in g.edges(data="weight"):
                nx.draw_networkx_edges(g, layout, edgelist=[edge], alpha=(edge[2] / 5), ax=ax)
            
            nx.draw_networkx_nodes(g, layout, node_color=color, ax=ax)
            nx.draw_networkx_labels(g, layout, ax=ax)
            
            if title is not None:
                
                if iteration is None:
                    ax.set_title(f"{title} {i}: {len(g.nodes)}")
                else:
                    ax.set_title(f"{title} {iteration}: {len(g.nodes)}")
