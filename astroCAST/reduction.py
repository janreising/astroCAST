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

    def convert_to_tsfresh(self, df, enforced_min=10, min_max_normalize=True):

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

        return pd.DataFrame({"id":ids, "time":times, "dim_0":dim_0s})

    def get_features(self, data, feature_only=False):

        # calculate features for long traces
        logging.info("creating tsfresh dataset ...")

        X = self.convert_to_tsfresh(data)

        logging.info("extracting features")
        features = tsfresh.extract_features(X, column_id="id", column_sort="time", disable_progressbar=False, n_jobs=12)
        features.index = data.index

        if feature_only:
            return features
        else:
            return pd.concat([data, features], axis=1)
