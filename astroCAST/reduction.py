import logging
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
import tsfresh

from astroCAST.helper import wrapper_local_cache

class FeatureExtraction:

    def __init__(self,local_cache=False, cache_path=None):

        self.local_cache = local_cache
        self.lc_path = cache_path

    def get_features(self, data, normalize=None, padding=None, n_jobs=-1, feature_only=False, show_progress=True):

        # calculate features for long traces
        logging.info("converting dataset to tsfresh format ...")

        if normalize is not None:

            # if min_max_normalize:
            #     trace = trace - trace[0]
            #     max_ = np.max(np.abs(trace))
            #     if max_ != 0:
            #         trace = trace / max_

            raise NotImplementedError

        if padding is not None:

            # if enforced_min is not None:
            #     if len(trace) < enforced_min:
            #         trace = np.pad(trace, (0, enforced_min-len(trace)), mode='edge', )

            raise NotImplementedError

        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        if isinstance(data, pd.DataFrame):
            iterator = data.trace.items()

        elif isinstance(data, pd.Series):
            iterator = data.items()

        elif isinstance(data, list):
            assert feature_only, f"when providing data as 'list' use 'feature_only=True'"
            iterator = list(zip(range(len(data)), data))

        elif isinstance(data, np.ndarray):
            assert feature_only, f"when providing data as 'np.ndarray' use 'feature_only=True'"
            iterator = list(zip(range(data.shape[0]), data.tolist()))

        iterator = tqdm(iterator, total=len(data)) if show_progress else iterator

        ids, times, dim_0s = [], [], []
        for id_, trace in iterator:

            if type(trace) != np.ndarray:
                trace = np.array(trace)

            # take care of NaN
            trace = np.nan_to_num(trace)

            ids = ids + [id_]*len(trace)
            times = times + list(range(len(trace)))
            dim_0s = dim_0s + list(trace)

        X = pd.DataFrame({"id":ids, "time":times, "dim_0":dim_0s})

        logging.info("extracting features")
        features = tsfresh.extract_features(X, column_id="id", column_sort="time", disable_progressbar=False,
                                            n_jobs=n_jobs)

        if feature_only:
            return features
        else:
            features.index = data.index
            return pd.concat([data, features], axis=1)
