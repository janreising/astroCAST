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



    #@staticmethod
    #def load_bary(subj):


    #@staticmethod
    #def bary_prep(subjects: list, mem_data = None, z_thr = 2, min_cluster_size = 15, load_bary = False,
    #             dtw_parameters = {"penalty": 0, "psi": None}, show_progress = False):

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
