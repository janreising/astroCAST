import pathlib
from pathlib import Path
import os
from typing import Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster
from dtaidistance import dtw, dtw_barycenter
import fastcluster
from collections import defaultdict
import argparse, os, logging
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from astroCAST.helper import wrapper_local_cache

class UnbiasedClustering():

    def __init__(self, working_directory:Path, local_cache:bool = True, cache_path:Path = None):
        
        if local_cache:
            assert cache_path is not None, "when enabling caching, please provide a 'local_cache' path" 
        
        cache_path = Path(cache_path)
        if not cache_path.is_dir():
            cache_path.mkdir()
        
        self.wd = Path(working_directory)
        self.lc_path = cache_path
        self.local_cache = local_cache

    #@staticmethod
    #def load_bary(subj):
    @wrapper_local_cache
    def load_bary(self, subj):
        """
        Load barycenter is a static method of the class UnbiasedClustering. 
        It Load barycenter data for a given subject.

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

    #@staticmethod
    #def bary_prep(subjects: list, mem_data = None, z_thr = 2, min_cluster_size = 15, load_bary = False,
    #             dtw_parameters = {"penalty": 0, "psi": None}, show_progress = False):
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

    #@staticmethod
    #def combine_barycenters(data: dict, z_thr = 2,
     #                       events = None, barycenters = None, Z = None,
     #                       add_cluster_to_events:bool = False, default_cluster = -1,
     #                       verbose = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
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
            lut = defaultdict(lambda: default_cluster)

            for _, row in barycenters.iterrows():
                lut.update({idx_: row.cluster for idx_ in row.trace_idx})

            events["cluster"] = events.idx.map(lut)

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

def parse_subjects(value):
    subjects = []
    for subject in value:
        path, condition = subject.split(",")
        subjects.append((Path(path.strip()), condition.strip()))
    return subjects

def parse_multiple_par(value):
    param = {}
    for kv in value:
        kv_parts = kv.split(",")
        k, v = kv_parts
        if v == 'None':
            v = None
        param[k] = v
    return param

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-wd", "--workingdir", required=True, default=None, help="Working directory path")
        parser.add_argument("--localcache", type=bool, required=False, default=True, help="Bool parameter, indicates if local_cache is to be used.")
        parser.add_argument("-lc", "--cachepath", required=False, default=None, help="Local cache directory")

        parser.add_argument("-s", "--subjects", required=True, nargs="+", action="store",
                            help="A list of paths of subjects data and corresponding condition. e.g. 'first/path/tofile, condition1' 'second/path/tofile, condition2'",
                            metavar=('DIR', 'CONDITION'), dest='subject_list', default=[])
        parser.add_argument("--loadbary", type=bool, default=False, help="Bool parameter, load bary")
        parser.add_argument("--showprogress", type=bool, default=False, help="Show progress")
        parser.add_argument("--dtwparam", type=str, nargs="+", default={'penalty': None, 'psi': None}, help="DTW parameters, space separated. e.g. 'penalty,None' 'psi,None'")
        parser.add_argument("--minclustsize", type=int, default=0, help="Minimum cluster size")
        parser.add_argument("--zthr", type=int, default=2, help="Z threshold")
        parser.add_argument("--combinepar", type=str, nargs="+",default={'z_thr': 4}, help="Combine parameters, space separater. e.g. 'z_thr,4'")

        args = parser.parse_args()

bary = UnbiasedClustering(working_directory = args.workingdir, local_cache = args.localcache, cache_path = args.cachepath)
data = bary.bary_prep(subjects = parse_subjects(args.subject_list), load_bary = args.loadbary,
                                    show_progress = args.showprogress, dtw_parameters = parse_multiple_par(args.dtwparam),
                                    min_cluster_size = args.minclustsize, z_thr = args.zthr)

comb_events, comb_barycenter, Z = bary.combine_barycenters(data, **parse_multiple_par(args.combinepar))

print("#events:\t{:,d}".format(len(comb_events)))
print("#barycenter:\t{:4,d}".format(len(comb_barycenter)))
print("reduction: {:.1f}%\n".format(len(comb_barycenter)/len(comb_events)*100))

print("#clusters: {:,d}".format(len(comb_barycenter.cluster.unique())))
print("reduction: {:.1f}% / {:.1f}%".format(
    len(comb_barycenter.cluster.unique())/len(comb_barycenter)*100, 
    len(comb_barycenter.cluster.unique())/len(comb_events)*100))
