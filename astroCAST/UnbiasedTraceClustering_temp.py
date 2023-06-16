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
