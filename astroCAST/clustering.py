import logging
import pickle
import tempfile
from pathlib import Path

import awkward
import fastcluster
import hdbscan
import numpy as np
import pandas as pd
from dtaidistance import dtw_barycenter, dtw
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from tqdm import tqdm

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

    def __init__(self, local_cache=None, caching=True):

        if caching:
            assert local_cache is not None, "when enabling caching, please provide a 'local_cache' path"

            local_cache = Path(local_cache)
            if not local_cache.is_dir():
                local_cache.mkdir()

        self.lc_path=local_cache
        self.local_cache = caching

    def get_barycenters(self, traces, param_distance_matrix=None, param_linkage_matrix=None, param_barycenter=None):
        raise NotImplementedError

    def load_traces(self): # TODO delete
        traces_path = sd.joinpath("bary_traces.npy")
        raw_path = sd.joinpath("raw_traces.npy")
        df_path = sd.joinpath("events.csv")

        if not traces_path.is_file() or not df_path.is_file() or not raw_path.is_file():
            # load events
            print("loading events ...")
            uc = ra.UnbiasedClustering(wd, cache=False)
            events = uc.combine_omega_objects([subject])

            # prepare traces
            print("preparing traces ...")
            parameters = meta["trace_parameters"]
            parameters["data"] = events

            res = uc.prepare_traces(**parameters, caching=False,
                                        cache_dir=None, data_dir=Path(meta["data_dir"]), verbose=10)
            traces = np.array(res.trace.tolist(), dtype=object)
            raw = np.array(res.raw.tolist(), dtype=object)

            # save dataframe
            del res["trace"]
            del res["raw"]
            res.to_csv(df_path.as_posix(), index=False)
            del res

            # save traces separately
            np.save(traces_path.as_posix(), traces)
            traces = np.load(traces_path.as_posix(), allow_pickle=True)

            np.save(raw_path.as_posix(), raw)
            raw = np.load(raw_path.as_posix(), allow_pickle=True)

        else:

            traces = np.load(traces_path.as_posix(), allow_pickle=True)
            # raw = np.load(raw_path.as_posix(), allow_pickle=True)

        if not meta["max_events"] is None:
            warnings.warn(str("too many events found. Reducing to {}".format(meta["max_events"])))
            traces = traces[:meta["max_events"]]
            # raw = raw[:meta["max_events"]]

    @wrapper_local_cache
    def calculate_distance_matrix(self, events, use_mmap=False, block=10000, show_progress=True):

        traces = events.events.trace.tolist()

        if is_ragged(traces):
            traces = awkward.array(traces)
        else:
            traces = np.array(traces)

        logging.warning(f"traces.shape: {traces.shape}")
        N = len(traces)

        if not use_mmap:
            distance_matrix = dtw.distance_matrix_fast(traces,
                        use_pruning=False, parallel=True, compact=True, only_triu=True)

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

    @wrapper_local_cache
    def calculate_linkage_matrix(self, distance_matrix, method="average", metric="euclidean"):
        Z = fastcluster.linkage(distance_matrix, method=method, metric=metric, preserve_input=False)
        return Z

    @staticmethod
    def cluster_linkage_matrix(Z, z_threshold, criterion="distance"):

        cluster_labels = fcluster(Z, z_threshold, criterion=criterion)
        clusters = pd.Series(cluster_labels).value_counts().sort_index()

        # TODO do we need to return both of these? or would pd.Series(cluster_labels) be sufficient
        return clusters, cluster_labels

        #TODO filtering: clusters = clusters[clusters > min_cluster_size]

    @wrapper_local_cache
    def calculate_barycenters(self, clusters, cluster_labels, events,
                              max_it=100, thr=1e-5, penalty=0, psi=None,
                              show_progress=True):

        """ Calculate consensus trace (barycenter) for each cluster"""

        traces = events.events.trace.tolist()

        if is_ragged(traces):
            traces = awkward.array(traces)
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

            barycenters[cl] = {"idx":idx_, "bc":bc, "num":clusters.iloc[i]}

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
