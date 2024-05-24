import inspect
import itertools
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import pyinform.shannon
import pytest
import seaborn as sns

try:
    import umap
    import umap.plot
except ImportError:
    logging.error(f"Unable to import umap package. Some functionality will not work as expected. "
                  f"If you are using astroCAST on MacOS this is expected.")

from matplotlib import pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy

from astrocast.analysis import Events
from astrocast.helper import CachedClass, wrapper_local_cache, experimental


class FeatureExtraction(CachedClass):
    
    def __init__(self, events: Events, cache_path=None, logging_level=logging.INFO):
        """
        A class for extracting various statistical and signal processing features from time series data.

        This class is designed to work with time series events, providing a suite of methods for feature extraction. It supports caching of results to optimize repeated computations and includes a variety of statistical, geometric, harmonic, and other time series analysis methods.

        Key methods include statistical measures (mean, median, variance, standard deviation), geometric and harmonic means, sum and energy computations, temporal derivatives, spectral analysis, and several others designed to capture different aspects of time series data.

        The `all_features` method aggregates all available features, providing a comprehensive overview of the extracted features for each time series in the dataset.

        Features available for extraction:
        - Mean: Compute the statistical mean.
        - Median: Compute the statistical median.
        - Geometric Mean (gmean): Compute the geometric mean.
        - Harmonic Mean (hmean): Compute the harmonic mean.
        - Vector Sum (vec_sum): Compute the vector sum.
        - Absolute Sum (abs_sum): Compute the sum of absolute values.
        - Absolute Energy (abs_energy): Compute the absolute sum of squares.
        - Standard Deviation (std): Compute the standard deviation.
        - Variance (var): Compute the variance.
        - Median Absolute Deviation (median_absolute_deviation): Compute the median absolute deviation.
        - Coefficient of Variation (variation): Compute the coefficient of variation.
        - Minimum and Maximum (minimum, maximum): Compute the min and max values.
        - Skewness (skew) and Kurtosis (kurt): Compute the skewness and kurtosis.
        - Temporal Derivatives (mean_diff, means_abs_diff): Compute mean and mean absolute temporal derivatives.
        - Mean Spectral Energy (mse): Compute mean spectral energy.
        - Zero Crossing and Slope Sign Changes: Compute zero crossings and slope sign changes.
        - Waveform Length (waveform_length): Compute the cumulative length of the waveform.
        - Root Mean Square (root_mean_square): Compute the root-mean-square.

        Further methods include statistical tests and other specific feature computations that can be useful in time
        series analysis and machine learning contexts.

        Usage:
            fe = FeatureExtraction(events, cache_path=None)
            features_df = fe.all_features(dropna=False, drop_axis='columns')

        Args:
            events (Events): An instance of the Events class containing time series data.
            cache_path (str, optional): Path to the cache directory. Defaults to the cache path of the events.
            logging_level (logging.LEVEL, optional): Level of logging. Defaults to logging.INFO.
        """
        
        if cache_path is None:
            cache_path = events.cache_path
        
        super().__init__(cache_path=cache_path, logging_level=logging_level)
        
        self.events = events
    
    @wrapper_local_cache
    def all_features(self, dropna: bool = False, drop_axis: Literal['rows', 'columns'] = 'columns') -> pd.DataFrame:
        """ Returns dictionary of all features in the module

        Args:
            dropna: Whether to drop NaN values from the dataframe
            drop_axis: Whether to drop rows or columns that contain NaN values.

        """
        
        # Using inspect to get only the functions
        exclusion = ['__hash__', '__init__', 'all_features', 'get_features', 'print_cache_path',
                     '_get_length_sequences_where']
        functions_list = [attr for attr, _ in inspect.getmembers(FeatureExtraction, inspect.isfunction) if
                          attr not in exclusion]
        
        features = {name: getattr(self, name) for name in functions_list}
        
        summary = {}
        for k, func in features.items():
            
            summ_values = []
            for trace in self.events.events.trace.tolist():
                try:
                    s = func(trace)
                except:
                    s = None
                summ_values += [s]
            
            summary[f"v_{k}"] = summ_values
        
        summary = pd.DataFrame(summary, index=self.events.events.index)
        
        for col in summary.columns:
            unique = summary[col].unique()
            if (unique[0] is None) and (len(unique) == 1):
                del summary[col]
        
        if dropna:
            num_before = len(summary)
            col_before = len(summary.columns)
            
            summary = summary.dropna(axis=drop_axis)
            
            if drop_axis == 'rows':
                logging.warning(f"Dropped {num_before - len(summary)} rows "
                                f"({len(summary) / num_before * 100:.1f}%)")
            else:
                logging.warning(f"Number of columns dropped: {col_before - len(summary.columns)}")
        
        return summary
    
    @staticmethod
    def mean(x):
        """ statistical mean for each variable in a segmented time series """
        return np.mean(x)
    
    @staticmethod
    def median(x):
        """ statistical median for each variable in a segmented time series """
        return np.median(x)
    
    @staticmethod
    def gmean(x):
        """ geometric mean for each variable """
        return stats.gmean(x)
    
    @staticmethod
    def hmean(x):
        """ harmonic mean for each variable """
        return stats.hmean(x)
    
    @staticmethod
    def vec_sum(x):
        """ vector sum of each variable """
        return np.sum(x)
    
    @staticmethod
    def abs_sum(x):
        """ sum of absolute values """
        return np.sum(np.abs(x))
    
    @staticmethod
    def abs_energy(x):
        """ absolute sum of squares for each variable """
        return np.sum(x * x)
    
    @staticmethod
    def std(x):
        """ statistical standard deviation for each variable in a segmented time series """
        return np.std(x)
    
    @staticmethod
    def var(x):
        """ statistical variance for each variable in a segmented time series """
        return np.var(x)
    
    @staticmethod
    def median_absolute_deviation(x):
        """ median absolute deviation for each variable in a segmented time series """
        if hasattr(stats, 'median_abs_deviation'):
            return stats.median_abs_deviation(x)
        else:
            return stats.median_absolute_deviation(x)
    
    @staticmethod
    def variation(x):
        """ coefficient of variation """
        return stats.variation(x)
    
    @staticmethod
    def minimum(x):
        """ minimum value for each variable in a segmented time series """
        return np.min(x)
    
    @staticmethod
    def maximum(x):
        """ maximum value for each variable in a segmented time series """
        return np.max(x)
    
    @staticmethod
    def skew(x):
        """ skewness for each variable in a segmented time series """
        return stats.skew(x)
    
    @staticmethod
    def kurt(x):
        """ kurtosis for each variable in a segmented time series """
        return stats.kurtosis(x)
    
    @staticmethod
    def mean_diff(x):
        """ mean temporal derivative """
        return np.mean(np.diff(x))
    
    @staticmethod
    def means_abs_diff(x):
        """ mean absolute temporal derivative """
        return np.mean(np.abs(np.diff(x)))
    
    @staticmethod
    def mse(x):
        """ computes mean spectral energy for each variable in a segmented time series """
        return np.mean(np.square(np.abs(np.fft.fft(x))))
    
    @staticmethod
    def mean_crossings(x):
        """ Computes number of mean crossings for each variable in a segmented time series """
        x = np.atleast_3d(x)
        N = x.shape[0]
        D = x.shape[2]
        mnx = np.zeros(N, D)
        for i in range(D):
            pos = x[:, :, i] > 0
            npos = ~pos
            c = (pos[:, :-1] & npos[:, 1:]) | (npos[:, :-1] & pos[:, 1:])
            mnx[:, i] = np.count_nonzero(c)
        return mnx
    
    @staticmethod
    def mean_abs(x):
        """ statistical mean of the absolute values for each variable in a segmented time series """
        return np.mean(np.abs(x))
    
    @staticmethod
    def zero_crossing(x, threshold=0):
        """ number of zero crossings among two consecutive samples above a certain threshold for each
        variable in the segmented time series"""
        
        sign = np.heaviside(-1 * x[:, :-1] * x[:, 1:], 0)
        abs_diff = np.abs(np.diff(x))
        return np.sum(sign * abs_diff >= threshold, dtype=x.dtype)
    
    @staticmethod
    def slope_sign_changes(x, threshold=0):
        """ number of changes between positive and negative slope among three consecutive samples
        above a certain threshold for each variable in the segmented time series"""
        
        change = (x[:, 1:-1] - x[:, :-2]) * (x[:, 1:-1] - x[:, 2:])
        return np.sum(change >= threshold, dtype=x.dtype)
    
    @staticmethod
    def waveform_length(x):
        """ cumulative length of the waveform over a segment for each variable in the segmented time
        series """
        return np.sum(np.abs(np.diff(x)))
    
    @staticmethod
    def root_mean_square(x):
        """ root mean square for each variable in the segmented time series """
        segment_width = x.shape[1]
        return np.sqrt(np.sum(x * x) / segment_width)
    
    @staticmethod
    def emg_var(x):
        """ variance (assuming a mean of zero) for each variable in the segmented time series
        (equals abs_energy divided by (seg_size - 1)) """
        segment_width = x.shape[1]
        return np.sum(x * x) / (segment_width - 1)
    
    @staticmethod
    def willison_amplitude(x, threshold=0):
        """ the Willison amplitude for each variable in the segmented time series """
        return np.sum(np.abs(np.diff(x)) >= threshold)
    
    @staticmethod
    def shannon_entropy(x, b=2):
        return pyinform.shannon.entropy(x, b=b)
    
    @staticmethod
    def cid_ce(x, normalize=True):
        """
                This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
                valleys etc.). It calculates the value of

                .. math::

                    \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

                .. rubric:: References

                |  [1] Batista, Gustavo EAPA, et al (2014).
                |  CID: an efficient complexity-invariant distance for time series.
                |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

                :param x: the time series to calculate the feature of
                :type x: numpy.ndarray
                :param normalize: should the time series be z-transformed?
                :type normalize: bool

                :return: the value of this feature
                :return type: float
                """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        if normalize:
            s = np.std(x)
            if s != 0:
                x = (x - np.mean(x)) / s
            else:
                return 0.0
        
        x = np.diff(x)
        return np.sqrt(np.dot(x, x))
    
    @staticmethod
    def large_standard_deviation(x, r=0.5):
        """
                Does time series have *large* standard deviation?

                Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
                min of x. Hence it checks if

                .. math::

                    std(x) > r * (max(X)-min(X))

                According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

                :param x: the time series to calculate the feature of
                :type x: numpy.ndarray
                :param r: the percentage of the range to compare with
                :type r: float
                :return: the value of this feature
                :return type: bool
                """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.std(x) > (r * (np.max(x) - np.min(x)))
    
    @staticmethod
    def _get_length_sequences_where(x):
        
        if len(x) == 0:
            return [0]
        else:
            res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
            return res if len(res) > 0 else [0]
    
    def longest_strike_above_mean(self, x):
        """
        Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.max(self._get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0
    
    def longest_strike_below_mean(self, x):
        """
        Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.max(self._get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0
    
    @staticmethod
    def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
        """
        Returns the percentage of non-unique data points. Non-unique means that they are
        contained another time in the time series again.

            # of data points occurring more than once / # of all data points

        This means the ratio is normalized to the number of data points in the time series,
        in contrast to the percentage_of_reoccurring_values_to_all_values.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if len(x) == 0:
            return np.nan
        
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        
        value_counts = x.value_counts()
        reoccuring_values = value_counts[value_counts > 1].sum()
        
        if np.isnan(reoccuring_values):
            return 0
        
        return reoccuring_values / x.size
    
    @staticmethod
    def symmetry_looking(x, r=0.5):
        """
                Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

                .. math::

                    | mean(X)-median(X)| < r * (max(X)-min(X))

                :param x: the time series to calculate the feature of
                :type x: numpy.ndarray
                :param param: contains dictionaries {"r": x} with x (float) is the percentage of the range to compare with
                :type param: list
                :return: the value of this feature
                :return type: bool
                """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        mean_median_difference = np.abs(np.mean(x) - np.median(x))
        max_min_difference = np.max(x) - np.min(x)
        return mean_median_difference < r * max_min_difference
    
    @staticmethod
    def variance_larger_than_standard_deviation(x):
        """
        Is variance higher than the standard deviation?

        Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
        being larger than 1

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: bool
        """
        y = np.var(x)
        return y > np.sqrt(y)
    
    def __hash__(self):
        return hash(self.events)


class UMAP:
    
    def __init__(self, n_neighbors=30, min_dist=0, n_components=2, metric="euclidean", ):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    
    def train(self, data) -> np.ndarray:
        return self.reducer.fit_transform(data)
    
    def embed(self, data):
        return self.reducer.transform(data)
    
    @staticmethod
    def _plot_napari(data: np.ndarray, labels: np.ndarray = None, size: float = 0.1):
        """Plot data using napari.

        Args:
          data: The data to plot.
          labels: Optional labels for the data points.
          size: The size of the points.

        Returns:
          A napari viewer instance displaying the plot.
        """
        napari = pytest.importorskip("napari")
        viewer = napari.Viewer()
        if labels is None:
            viewer.add_points(data, size=size)
        else:
            labels_ = labels / np.max(labels)
            viewer.add_points(data, properties={'labels': labels_}, face_color='labels', face_colormap='viridis',
                              size=size)
        return viewer
    
    @staticmethod
    def _plot_matplotlib(data: np.ndarray, alpha=0.1, ax: plt.Axes = None, labels: np.ndarray = None,
                         true_labels: np.ndarray = None,
                         size: float = 0.1) -> plt.Axes:
        """Plot data using UMAP and matplotlib.

        Args:
          data: The data to plot.
          ax: The matplotlib axes to plot on.
          labels: Optional labels for the data points.
          size: The size of the points.

        Returns:
          The matplotlib axes with the plot.
        """
        
        df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1], c=labels, s=true_labels))
        sns.scatterplot(data=df, x="x", y="y", alpha=alpha, size=size, style="s", hue="c", ax=ax)
        
        return ax
    
    def _plot_umap(self, ax: plt.Axes, labels: np.ndarray = None):
        """ Method to use umap package plotting

        Args:
          ax: The matplotlib axes to plot on.
          labels: Optional labels for the data points.

        Returns:
          The matplotlib axes with the plot.
        """
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        umap.plot.points(self.reducer, labels=labels, ax=ax)
    
    @experimental
    def plot(self, data: np.ndarray = None, ax: plt.Axes = None,
             labels: np.ndarray = None, true_labels: np.ndarray = None,
             size: float = 0.1, plot_type: str = 'napari', alpha: float = 0.1,
             figsize: Tuple[int, int] = (10, 10)):
        """Plot data using specified plotting type.

        Supports plotting with napari, UMAP via matplotlib, or matplotlib directly.

        Args:
          data: The data to plot. Must be provided if `plot_type` is not 'napari'.
          ax: The matplotlib axes to plot on, required if `plot_type` is 'matplotlib' or 'umap'.
          labels: Optional labels for the data points.
          true_labels: Optional true labels
          size: The size of the points.
          plot_type: The type of plot to produce, options are 'napari', 'umap', 'matplotlib'.
          figsize: Size of the figure.
        
        Returns:
          A napari viewer or matplotlib axes displaying the plot, depending on `plot_type`.

        Raises:
          ValueError: If required arguments are missing for the chosen `plot_type`.
        """
        
        if plot_type in ['napari', 'matplotlib'] and data is None:
            raise ValueError("Please provide the data attribute for plotting with napari and matplotlib.")
        
        if plot_type == 'napari':
            return self._plot_napari(data, labels, size)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if plot_type == 'umap':
            return self._plot_umap(ax=ax, labels=labels)
        
        elif plot_type == 'matplotlib':
            return self._plot_matplotlib(data=data, ax=ax, alpha=alpha, labels=labels, true_labels=true_labels,
                                         size=size)
    
    def save(self, path):
        
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_dir():
            path = path.with_name("umap.p")
            logging.info(f"saving umap to {path}")
        
        assert not path.is_file(), f"file already exists: {path}"
        pickle.dump(self.reducer, open(path, "wb"))
    
    def load(self, path):
        
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_dir():
            path = path.with_name("umap.p")
            logging.info(f"loading umap from {path}")
        
        assert path.is_file(), f"can't find umap: {path}"
        self.reducer = pickle.load(open(path, "rb"))


class ClusterTree:
    """ converts linkage matrix to searchable tree"""
    
    def __init__(self, Z, names: List = None):
        self.tree = hierarchy.to_tree(Z)
        self.root_id = np.max(Z[:, :2]) + 1
        
        self.leaves = None
        self.nodes = None
        
        if names is not None:
            value_dictionary = {l: n for l, n in list(zip(range(len(names)), names))}
            self._set_attributes("idx", value_dictionary)
    
    def get_root(self):
        return self.get_node(self.root_id)
    
    def get_node(self, id_):
        return self.search(self.tree, id_)
    
    def get_leaves(self, tree=None, attribute: str = "id"):
        
        if tree is None:
            if self.leaves is None:
                self.leaves = self.get_leaves(self.get_root())
            return self.leaves
        
        if tree.is_leaf():
            return [getattr(tree, attribute)]
        
        left = self.get_leaves(tree.get_left())
        right = self.get_leaves(tree.get_right())
        
        return left + right
    
    def get_nodes(self, tree=None):
        
        if tree is None:
            if self.nodes is None:
                self.nodes = self.get_nodes(self.get_root())
            return self.nodes
        
        if tree.is_leaf():
            return []
        
        left = self.get_nodes(tree.get_left())
        right = self.get_nodes(tree.get_right())
        
        return [tree.id] + left + right
    
    def get_count(self, tree):
        
        if tree.is_leaf():
            return 1
        
        left = self.get_count(tree.get_left())
        right = self.get_count(tree.get_right())
        
        return left + right
    
    def get_attribute(self, id_, attribute: str, tree=None):
        
        if tree is None:
            tree = self.get_root()
        
        node = self.search(tree, id_)
        return getattr(node, attribute)
    
    def get_attributes(self, attribute: str, ids: List, tree=None):
        attributes = {}
        for id_ in ids:
            attr = self.get_attribute(id_, attribute, tree=tree)
            attributes[id_] = attr
        
        return attributes
    
    def _set_attribute(self, id_, attribute: str, value):
        
        node = self.search(self.get_root(), id_=id_)
        setattr(node, attribute, value)
    
    def _set_attributes(self, attribute: str, value_dictionary: dict):
        
        for id_, value in value_dictionary.items():
            self._set_attribute(id_, attribute, value)
    
    def search(self, tree, id_):
        
        if tree is None:
            return None
        
        if tree.id == id_:
            return tree
        
        left = self.search(tree.get_left(), id_)
        if left is not None:
            return left
        
        right = self.search(tree.get_right(), id_)
        if right is not None:
            return right
        
        return None
    
    def is_leaf(self):
        """
        Determines if the given node is a leaf in the tree.

        Args:
            tree (ClusterNode): The node to check.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self.tree.get_left() is None and self.tree.get_right() is None
    
    def _get_clusters_by_leaf_num(self, root_node, max_leaves: int, min_leaves: int = 1):
        """ Recursive function to get clusters by number of leaf nodes.

        Args:
            root_node: Node to start searching for clusters.
            max_leaves: Maximum number of leaf nodes below each cluster node.
            min_leaves: Minimum number of leaf nodes below each cluster node.

        Returns:
            # todo: what would make sense to return? Ultimately we need a mapping from lead_id to cluster_name

        """
        
        leaf_count = root_node.get_count()
        
        if min_leaves <= leaf_count < max_leaves:
            return [root_node]
        
        elif leaf_count >= max_leaves:
            left_cluster_nodes = self._get_clusters_by_leaf_num(root_node.get_left(), max_leaves=max_leaves,
                                                                min_leaves=min_leaves)
            right_cluster_nodes = self._get_clusters_by_leaf_num(root_node.get_right(), max_leaves=max_leaves,
                                                                 min_leaves=min_leaves)
            return left_cluster_nodes + right_cluster_nodes
        
        elif leaf_count < min_leaves:
            return []
        
        else:
            logging.error(f"Implementation should not be able to reach this code section. "
                          f"leaf_count: {leaf_count} ({max_leaves}, {min_leaves})")
            return []
    
    def get_clusters(self, max_leaves: int, min_leaves: int = 1, default_value=-1):
        
        if isinstance(max_leaves, float):
            max_leaves = int(max_leaves * self.get_count(self.get_root()))
        
        if isinstance(min_leaves, float):
            min_leaves = int(min_leaves * self.get_count(self.get_root()))
        
        cluster_nodes = self._get_clusters_by_leaf_num(root_node=self.get_root(),
                                                       max_leaves=max_leaves, min_leaves=min_leaves)
        
        mapping = defaultdict(lambda: default_value)
        
        cluster_id = 0
        for cn in cluster_nodes:
            leaves = self.get_leaves(cn)
            for leaf in leaves:
                mapping[leaf] = cluster_id
            
            cluster_id += 1
        
        return mapping
    
    def calculate_distance(self, node1, node2):
        """
        Calculate the distance between two nodes. This should be implemented based on
        your specific distance metric.

        Args:
            node1 (ClusterNode): The first node.
            node2 (ClusterNode): The second node.

        Returns:
            float: The distance between the two nodes.
        """
        # Placeholder for actual distance calculation
        # Implement this based on your distance metric (e.g., Euclidean distance)
        return np.linalg.norm(np.array(node1.dist) - np.array(node2.dist))
    
    def merge_trees(self, tree1, tree2, threshold):
        """
        Merges two trees based on a distance threshold.

        Args:
            tree1 (ClusterNode): The root node of the first tree.
            tree2 (ClusterNode): The root node of the second tree.
            threshold (float): The distance threshold for merging.

        Returns:
            ClusterNode: The root of the merged tree.
        """
        if self.calculate_distance(tree1, tree2) <= threshold:
            new_root = TreeNode(len(self.tree) + 1)
            new_root.children = [tree1, tree2]
            new_root.distance = self.calculate_distance(tree1, tree2)
            tree1.parent = new_root
            tree2.parent = new_root
            return new_root
        else:
            # If not mergeable, return a new root with tree1 and tree2 as separate children
            new_root = TreeNode(len(self.tree) + 1)
            new_root.children = [tree1, tree2]
            new_root.distance = self.calculate_distance(tree1, tree2)
            tree1.parent = new_root
            tree2.parent = new_root
            return new_root
    
    def tree_to_linkage(self):
        """
        Converts the tree back to a linkage matrix.

        Returns:
            np.ndarray: The linkage matrix.
        """
        
        root = self.get_root()
        leaves = self.get_leaves(root)
        
        def add_node(node, node_list):
            if node.is_leaf():
                return node.id
            left_id = add_node(node.get_left(), node_list)
            right_id = add_node(node.get_right(), node_list)
            node_id = len(node_list) + len(leaves)
            node_list.append([left_id, right_id, node.dist, node.count])
            return node_id
        
        node_list = []
        add_node(root, node_list)
        return np.array(node_list)
    
    def plot_dendogram(self, truncate_mode='level', p=4, figsize=(8, 3), ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        Z = self.tree_to_linkage()
        _ = hierarchy.dendrogram(Z=Z, truncate_mode=truncate_mode, p=p, ax=ax)
    
    def _calculate_overlap(self, other_tree):
        """
        Calculate the overlap between two nodes based on their leaf nodes.

        Args:
            node1 (ClusterNode): The first node.
            node2 (ClusterNode): The second node.

        Returns:
            float: The overlap metric between the two nodes.
        """
        attr1 = self.get_leaves(attribute="idx")
        attr2 = other_tree.get_leaves(attribute="idx")
        
        print(attr1)
        #
        # attr1 = [l.idx for l in leaves1]
        # attr2 = [l.idx for l in leaves2]
        #
        # attr1 = self.get_attributes("idx", ids=leaves1)
        # attr2 = other_tree.get_attributes("idx", ids=leaves2)
        
        # attr1 = set(attr1.values())
        # attr2 = set(attr2.values())
        
        attr1 = set(attr1)
        attr2 = set(attr2)
        
        intersection = len(attr1.intersection(attr2))
        union = len(attr1.union(attr2))
        return intersection / union
