import logging
import multiprocessing
import pickle
from pathlib import Path

import keras.models
import numpy as np
import pandas as pd
import umap
import umap.plot
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Conv1D, MaxPooling1D, UpSampling1D, ActivityRegularization
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tsfresh

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

class UMAP:
    def __init__(self, n_neighbors=30, min_dist=0, n_components=2, metric="euclidean",):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)

    def train(self, data):
        return self.reducer.fit_transform(data)

    def embed(self, data):
        return self.reducer.transform(data)

    def plot(self, data=None, ax=None, labels=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        if data is None:
            umap.plot.points(self.reducer, labels=labels, ax=ax)

        else:

            if labels is not None:

                palette = sns.color_palette("husl", len(np.unique(labels)))
                ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1,
                           color=[palette[v] for v in labels])

            else:
                ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1)

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


class CNN:

    """ embeds data in a latent space of defined size
    """

    def __init__(self, encoder=None, autoencoder=None):

        self.encoder = encoder
        self.autoencoder = autoencoder

        self.history = None

    def train(self, data, train_split=0.9, validation_split=0.1,
              loss='mse', dropout=None, regularize_latent=None,
              epochs=50, batch_size=64, patience=5, min_delta=0.0005, monitor="val_loss"):

        assert isinstance(data, np.ndarray), f"please provide data in 'np.ndarray' format instead of {type(data)}"
        assert isinstance(data.dtype, object), f"please provide data in format other than 'object' type "
        assert not np.isnan(data).any(), "data contains NaN values. Please exclude data points or fill NaN values (eg. np.nan_to_num)"

        # TODO to_time_series_dataset(traces)

        if self.encoder is not None:
            logging.warning("encoder was provided during initialization. This function will override the 'encoder' attribute.")

        # split dataset
        split_index = int(data.shape[0]*train_split)
        X_train = data[:split_index, :]
        X_test = data[split_index:, :]

        # callbacks
        callbacks = [EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta)]

        # create model
        input_window = Input(shape=(X_train.shape[1], 1))

        x = input_window
        x = Dropout(dropout)(x) if dropout is not None else x
        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2, padding="same", )(x)
        x = Conv1D(16, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2, padding="same")(x)
        x = ActivityRegularization(l1=regularize_latent)(x) if regularize_latent is not None else x
        encoded = x

        x = Conv1D(16, 3, activation="relu", padding="same")(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(64, 3, activation='relu', padding="same")(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
        decoded = x

        encoder = Model(input_window, encoded)
        autoencoder = Model(input_window, decoded)

        logging.info("Model architecture:\n", autoencoder.summary)

        # train
        autoencoder.compile(optimizer='adam', loss=loss)
        history = autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        shuffle=True, verbose=0,
                        validation_split=validation_split)

        self.encoder = encoder
        self.autoencoder = autoencoder
        self.history = history

        # TODO save model

        # quality control
        # TODO for some reason this is an array and not a float?!
        Y_test = autoencoder.predict(X_test)
        MSE = mean_squared_error(np.squeeze(X_test), np.squeeze(Y_test))
        logging.info(f"Quality of encoding > MSE: {MSE}") # :.4f

        return history, X_test, Y_test, MSE

    def embed(self, data):

        assert self.encoder is not None, "please provide 'encoder' at initialization or use CNN.train() function."

        assert isinstance(data, np.ndarray), f"please provide data in 'np.ndarray' format instead of {type(data)}"
        assert isinstance(data.dtype, object), f"please provide data in format other than 'object' type "
        assert not np.isnan(data).any(), "data contains NaN values. Please exclude data points or fill NaN values (eg. np.nan_to_num)"

        # predicting
        latent = self.encoder.predict(data)
        latent = np.reshape(latent, (latent.shape[0], int(latent.shape[1]*latent.shape[2])))

        latent = np.squeeze(latent)

        return latent

    def plot_history(self, history=None, figsize=(4, 2)):

        if history is None:
            history = self.history

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

        ax0.plot(history.history["loss"])
        ax0.set_title("Train loss")

        ax1.plot(history.history["val_loss"])
        ax1.set_title("Validation loss")

        plt.tight_layout()

    def plot_examples(self, X_test, Y_test=None, num_samples=9, figsize=(10, 3)):

        assert (Y_test is not None) or (self.autoencoder), "please train autoencoder or provide 'Y_test' argument"

        if Y_test is None:
            Y_test = self.autoencoder.predict(X_test)

        X_test = np.squeeze(X_test)
        Y_test = np.squeeze(Y_test)

        if type(num_samples) == int:
            num_rounds = 1

        else:
            num_rounds, num_samples = num_samples

        for nr in range(num_rounds):

            fig, axx = plt.subplots(2, num_samples, figsize=figsize, sharey=True)

            for i, idx in enumerate([np.random.randint(0, len(X_test)-1) for n in range(num_samples)]):

                inp = X_test[idx, :]
                out = Y_test[idx, :]

                inp = np.trim_zeros(inp, trim="b")
                out = out[0:len(inp)]

                axx[0, i].plot(inp, alpha=0.75, color="black")
                axx[0, i].plot(out, alpha=0.75, linestyle="--", color="darkgreen")
                axx[1, i].plot(inp-out)

                axx[0, i].get_xaxis().set_visible(False)
                axx[1, i].get_xaxis().set_visible(False)

                if i != 0:

                    axx[0, i].get_yaxis().set_visible(False)
                    axx[1, i].get_yaxis().set_visible(False)

            axx[0, 0].set_ylabel("IN/OUT", fontweight=600)
            axx[1, 0].set_ylabel("error", fontweight=600)

    def save_model(self, path, model=None):

        if model is None:
            assert self.encoder is not None, "please provide a 'model' or train a new one 'train()'"

        if isinstance(path, str):
            path = Path(path)

        encoder_path = path if path.suffix == ".h5" else path.joinpath("encoder.h5")
        assert not encoder_path.is_file(), f"output file exists. Please delete or provide different path: {encoder_path}"
        self.encoder.save(encoder_path.as_posix())
        logging.info(f"saved encoder model to {encoder_path}")

        autoencoder_path = path if path.suffix == ".h5" else path.joinpath("autoencoder.h5")
        assert not autoencoder_path.is_file(), f"output file exists. Please delete or provide different path: {autoencoder_path}"
        self.encoder.save(autoencoder_path.as_posix())
        logging.info(f"saved autoencoder model to {autoencoder_path}")

    def load_model(self, path, loading_encoder=True):

        if isinstance(path, str):
            path = Path(path)

        if loading_encoder:
            model_path = path if path.suffix == ".h5" else path.joinpath("encoder.h5")
            assert model_path.is_file(), f"Can't find model: {model_path}"
            self.encoder = keras.models.load_model(model_path.as_posix())

        else:
            model_path = path if path.suffix == ".h5" else path.joinpath("autoencoder.h5")
            assert model_path.is_file(), f"Can't find model: {model_path}"
            self.encoder = keras.models.load_model(model_path.as_posix())
            self.autoencoder = keras.models.load_model(model_path.as_posix())

class Barycenter_OR_TIMECLUSTERING:
    def __init__(self):
        pass

class ClusterTree:
    def __init__(self):
        pass
