import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import tsfresh

from astroCAST.helper import wrapper_local_cache

class FeatureExtraction:

    def __init__(self,
                 local_cache=False, cache_path=None):

        self.lc_path = cache_path

    def get_features(self, data, normalize=None, padding=None, n_jobs=-1, feature_only=False):

        # calculate features for long traces
        logging.info("converting dataset to tsfresh format ...")

        if normalize is not None:
            raise NotImplementedError

        if padding is not None:
            raise NotImplementedError

        ids, times, dim_0s = [], [], []
        for id_, row in tqdm(df.iterrows(), total=len(df)):

            trace = row.trace

            if type(trace) != np.ndarray:
                trace = np.array(trace)

            if min_max_normalize:
                trace = trace - trace[0]
                max_ = np.max(np.abs(trace))
                if max_ != 0:
                    trace = trace / max_

            if enforced_min is not None:
                if len(trace) < enforced_min:
                    trace = np.pad(trace, (0, enforced_min-len(trace)), mode='edge', )

            # take care of NaN
            trace = np.nan_to_num(trace)

            ids = ids + [id_]*len(trace)
            times = times + list(range(len(trace)))
            dim_0s = dim_0s + list(trace)

        X = pd.DataFrame({"id":ids, "time":times, "dim_0":dim_0s})

        logging.info("extracting features")
        features = tsfresh.extract_features(X, column_id="id", column_sort="time", disable_progressbar=False,
                                            n_jobs=n_jobs) # TODO dynamic
        features.index = data.index

        if feature_only:
            return features
        else:
            return pd.concat([data, features], axis=1)
