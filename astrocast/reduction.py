import inspect
import itertools
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyinform.shannon
import pytest
import seaborn as sns
from typing import Tuple

from matplotlib.lines import Line2D

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
        
        if cache_path is None:
            cache_path = events.cache_path
        
        super().__init__(cache_path=cache_path, logging_level=logging_level)
        
        self.events = events
    
    @wrapper_local_cache
    def all_features(self, dropna: bool = False) -> pd.DataFrame:
        """ Returns dictionary of all features in the module

        Args:
            dropna: Whether to drop NaN values from the dataframe

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
            len_before = len(summary)
            summary = summary.dropna(axis="rows")
            logging.warning(f"Dropped {len_before - len(summary)} rows "
                            f"({len(summary) / len_before * 100:.1f}%)")
        
        return summary
    
    @staticmethod
    def mean(X):
        """ statistical mean for each variable in a segmented time series """
        return np.mean(X)
    
    @staticmethod
    def median(X):
        """ statistical median for each variable in a segmented time series """
        return np.median(X)
    
    @staticmethod
    def gmean(X):
        """ geometric mean for each variable """
        return stats.gmean(X)
    
    @staticmethod
    def hmean(X):
        """ harmonic mean for each variable """
        return stats.hmean(X)
    
    @staticmethod
    def vec_sum(X):
        """ vector sum of each variable """
        return np.sum(X)
    
    @staticmethod
    def abs_sum(X):
        """ sum of absolute values """
        return np.sum(np.abs(X))
    
    @staticmethod
    def abs_energy(X):
        """ absolute sum of squares for each variable """
        return np.sum(X * X)
    
    @staticmethod
    def std(X):
        """ statistical standard deviation for each variable in a segmented time series """
        return np.std(X)
    
    @staticmethod
    def var(X):
        """ statistical variance for each variable in a segmented time series """
        return np.var(X)
    
    @staticmethod
    def median_absolute_deviation(X):
        """ median absolute deviation for each variable in a segmented time series """
        if hasattr(stats, 'median_abs_deviation'):
            return stats.median_abs_deviation(X)
        else:
            return stats.median_absolute_deviation(X)
    
    @staticmethod
    def variation(X):
        """ coefficient of variation """
        return stats.variation(X)
    
    @staticmethod
    def minimum(X):
        """ minimum value for each variable in a segmented time series """
        return np.min(X)
    
    @staticmethod
    def maximum(X):
        """ maximum value for each variable in a segmented time series """
        return np.max(X)
    
    @staticmethod
    def skew(X):
        """ skewness for each variable in a segmented time series """
        return stats.skew(X)
    
    @staticmethod
    def kurt(X):
        """ kurtosis for each variable in a segmented time series """
        return stats.kurtosis(X)
    
    @staticmethod
    def mean_diff(X):
        """ mean temporal derivative """
        return np.mean(np.diff(X))
    
    @staticmethod
    def means_abs_diff(X):
        """ mean absolute temporal derivative """
        return np.mean(np.abs(np.diff(X)))
    
    @staticmethod
    def mse(X):
        """ computes mean spectral energy for each variable in a segmented time series """
        return np.mean(np.square(np.abs(np.fft.fft(X))))
    
    @staticmethod
    def mean_crossings(X):
        """ Computes number of mean crossings for each variable in a segmented time series """
        X = np.atleast_3d(X)
        N = X.shape[0]
        D = X.shape[2]
        mnx = np.zeros(N, D)
        for i in range(D):
            pos = X[:, :, i] > 0
            npos = ~pos
            c = (pos[:, :-1] & npos[:, 1:]) | (npos[:, :-1] & pos[:, 1:])
            mnx[:, i] = np.count_nonzero(c)
        return mnx
    
    @staticmethod
    def mean_abs(X):
        """ statistical mean of the absolute values for each variable in a segmented time series """
        return np.mean(np.abs(X))
    
    @staticmethod
    def zero_crossing(X, threshold=0):
        """ number of zero crossings among two consecutive samples above a certain threshold for each
        variable in the segmented time series"""
        
        sign = np.heaviside(-1 * X[:, :-1] * X[:, 1:], 0)
        abs_diff = np.abs(np.diff(X))
        return np.sum(sign * abs_diff >= threshold, dtype=X.dtype)
    
    @staticmethod
    def slope_sign_changes(X, threshold=0):
        """ number of changes between positive and negative slope among three consecutive samples
        above a certain threshold for each variable in the segmented time series"""
        
        change = (X[:, 1:-1] - X[:, :-2]) * (X[:, 1:-1] - X[:, 2:])
        return np.sum(change >= threshold, dtype=X.dtype)
    
    @staticmethod
    def waveform_length(X):
        """ cumulative length of the waveform over a segment for each variable in the segmented time
        series """
        return np.sum(np.abs(np.diff(X)))
    
    @staticmethod
    def root_mean_square(X):
        """ root mean square for each variable in the segmented time series """
        segment_width = X.shape[1]
        return np.sqrt(np.sum(X * X) / segment_width)
    
    @staticmethod
    def emg_var(X):
        """ variance (assuming a mean of zero) for each variable in the segmented time series
        (equals abs_energy divided by (seg_size - 1)) """
        segment_width = X.shape[1]
        return np.sum(X * X) / (segment_width - 1)
    
    @staticmethod
    def willison_amplitude(X, threshold=0):
        """ the Willison amplitude for each variable in the segmented time series """
        return np.sum(np.abs(np.diff(X)) >= threshold)
    
    @staticmethod
    def shannon_entropy(X, b=2):
        return pyinform.shannon.entropy(X, b=b)
    
    @staticmethod
    def cid_ce(X, normalize=True):
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
        if not isinstance(X, (np.ndarray, pd.Series)):
            X = np.asarray(X)
        if normalize:
            s = np.std(X)
            if s != 0:
                X = (X - np.mean(X)) / s
            else:
                return 0.0
        
        X = np.diff(X)
        return np.sqrt(np.dot(X, X))
    
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
    
    def train(self, data):
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
    
    def __init__(self, Z):
        self.tree = hierarchy.to_tree(Z)
    
    def get_node(self, id_):
        return self.search(self.tree, id_)
    
    def get_leaves(self, tree):
        
        if tree.is_leaf():
            return [tree.id]
        
        left = self.get_leaves(tree.get_left())
        right = self.get_leaves(tree.get_right())
        
        return left + right
    
    def get_count(self, tree):
        
        if tree.is_leaf():
            return 1
        
        left = self.get_count(tree.get_left())
        right = self.get_count(tree.get_right())
        
        return left + right
    
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
