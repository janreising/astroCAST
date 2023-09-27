import inspect
import itertools
import logging
import math
import multiprocessing
import pickle
from pathlib import Path

import keras.models
import napari
import numpy as np
import pandas as pd
import pyinform.shannon
import umap
import umap.plot
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Conv1D, MaxPooling1D, UpSampling1D, ActivityRegularization
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

from astrocast.analysis import Events
from astrocast.helper import CachedClass, wrapper_local_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim


class FeatureExtraction(CachedClass):

    def __init__(self, events:Events, cache_path=None, logging_level=logging.INFO):
        super().__init__(cache_path=cache_path, logging_level=logging_level)

        self.events = events

    @wrapper_local_cache
    def all_features(self):
        """ Returns dictionary of all features in the module

        """

        # Using inspect to get only the functions
        exclusion = ['__hash__', '__init__', 'all_features', 'get_features',
                     'print_cache_path', '_get_length_sequences_where']
        functions_list = [attr for attr, _ in inspect.getmembers(FeatureExtraction, inspect.isfunction) if attr not in exclusion]

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

        return summary

    def mean(self, X):
        """ statistical mean for each variable in a segmented time series """
        return np.mean(X)

    def median(self, X):
        """ statistical median for each variable in a segmented time series """
        return np.median(X)

    def gmean(self, X):
        """ geometric mean for each variable """
        return stats.gmean(X)

    def hmean(self, X):
        """ harmonic mean for each variable """
        return stats.hmean(X)

    def vec_sum(self, X):
        """ vector sum of each variable """
        return np.sum(X)

    def abs_sum(self, X):
        """ sum of absolute values """
        return np.sum(np.abs(X))

    def abs_energy(self, X):
        """ absolute sum of squares for each variable """
        return np.sum(X * X)

    def std(self, X):
        """ statistical standard deviation for each variable in a segmented time series """
        return np.std(X)

    def var(self, X):
        """ statistical variance for each variable in a segmented time series """
        return np.var(X)

    def median_absolute_deviation(self, X):
        """ median absolute deviation for each variable in a segmented time series """
        if hasattr(stats, 'median_abs_deviation'):
            return stats.median_abs_deviation(X)
        else:
            return stats.median_absolute_deviation( X)

    def variation(self, X):
        """ coefficient of variation """
        return stats.variation(X)

    def minimum(self, X):
        """ minimum value for each variable in a segmented time series """
        return np.min(X)

    def maximum(self, X):
        """ maximum value for each variable in a segmented time series """
        return np.max(X)

    def skew(self, X):
        """ skewness for each variable in a segmented time series """
        return stats.skew(X)

    def kurt(self, X):
        """ kurtosis for each variable in a segmented time series """
        return stats.kurtosis(X)

    def mean_diff(self, X):
        """ mean temporal derivative """
        return np.mean(np.diff(X))

    def means_abs_diff(self, X):
        """ mean absolute temporal derivative """
        return np.mean(np.abs(np.diff(X)))

    def mse(self, X):
        """ computes mean spectral energy for each variable in a segmented time series """
        return np.mean(np.square(np.abs(np.fft.fft(X))))

    def mean_crossings(self, X):
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

    def mean_abs(self, X):
        """ statistical mean of the absolute values for each variable in a segmented time series """
        return np.mean(np.abs(X))

    def zero_crossing(self, X, threshold=0):
        """ number of zero crossings among two consecutive samples above a certain threshold for each
        variable in the segmented time series"""

        sign = np.heaviside(-1 * X[:, :-1] * X[:, 1:], 0)
        abs_diff = np.abs(np.diff(X))
        return np.sum(sign * abs_diff >= threshold, dtype=X.dtype)

    def slope_sign_changes(self, X, threshold=0):
        """ number of changes between positive and negative slope among three consecutive samples
        above a certain threshold for each variable in the segmented time series"""

        change = (X[:, 1:-1] - X[:, :-2]) * (X[:, 1:-1] - X[:, 2:])
        return np.sum(change >= threshold, dtype=X.dtype)

    def waveform_length(self, X):
        """ cumulative length of the waveform over a segment for each variable in the segmented time
        series """
        return np.sum(np.abs(np.diff(X)))

    def root_mean_square(self, X):
        """ root mean square for each variable in the segmented time series """
        segment_width = X.shape[1]
        return np.sqrt(np.sum(X * X) / segment_width)

    def emg_var(self, X):
        """ variance (assuming a mean of zero) for each variable in the segmented time series
        (equals abs_energy divided by (seg_size - 1)) """
        segment_width = X.shape[1]
        return np.sum(X * X) / (segment_width - 1)

    def willison_amplitude(self, X, threshold=0):
        """ the Willison amplitude for each variable in the segmented time series """
        return np.sum(np.abs(np.diff(X)) >= threshold)

    def shannon_entropy(self, X, b=2):
        return pyinform.shannon.entropy(X, b=b)

    def cid_ce(self, X, normalize=True):
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

    def large_standard_deviation(self, x, r=0.5):
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

    def percentage_of_reoccurring_datapoints_to_all_datapoints(self, x):
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

    def symmetry_looking(self, x, r=0.5):
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

    def variance_larger_than_standard_deviation(self, x):
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
    def __init__(self, n_neighbors=30, min_dist=0, n_components=2, metric="euclidean",):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)

    def train(self, data):
        return self.reducer.fit_transform(data)

    def embed(self, data):
        return self.reducer.transform(data)

    def plot(self, data=None, ax=None, labels=None, size=0.1, use_napari=True):

        if use_napari:

            if data is None:
                raise ValueError("please provide the data attribute or set 'use_napari' to False")

            viewer = napari.Viewer()

            points = data

            if labels is None:
                viewer.add_points(points, size=size)
            else:
                labels_ = labels/np.max(labels)
                viewer.add_points(points,
                                  properties={'labels':labels_},
                                  face_color='labels', face_colormap='viridis',
                                  size=size)

            return viewer

        else:

            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            if data is None:
                umap.plot.points(self.reducer, labels=labels, ax=ax)

            else:

                if labels is not None:

                    palette = sns.color_palette("husl", len(np.unique(labels)))
                    ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=size,
                               color=[palette[v] for v in labels])

                else:
                    ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=size)

                return ax

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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss - self.min_validation_loss <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

class CustomUpsample(nn.Module):
    def __init__(self, target_length):
        super(CustomUpsample, self).__init__()
        self.target_length = target_length

    def forward(self, x):
        current_length = x.shape[-1]
        diff = self.target_length - current_length
        zeros = torch.zeros((x.shape[0], x.shape[1], diff)).to(x.device)
        return torch.cat([x, zeros], dim=-1)

class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.15, l1_reg=0.0001):
        super(RNN_Autoencoder, self).__init__()

        self.l1_reg = l1_reg
        # self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=dropout)
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.GRU(hidden_dim, input_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, lengths):
        # Encoder
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.encoder(packed_x)

        # Prepare the initial input and hidden state for decoder
        unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        decoder_input = unpacked_out
        decoder_hidden = h_n[-1].unsqueeze(0)  # Using the last layer's hidden state

        # Decoder
        decoded_out, _ = self.decoder(decoder_input, decoder_hidden)

        # Calculate L1 loss on the latent representation
        latent_representation = h_n.view(h_n.size(1), -1)
        l1_loss = self.l1_reg * torch.norm(latent_representation, 1) if self.l1_reg else 0

        return decoded_out, l1_loss, latent_representation


    def split_dataset(self, data, val_split=0.1, train_split=0.8, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        total_size = len(data)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        return random_split(data, [train_size, val_size, test_size])

    @staticmethod
    def collate_fn(batch):
        # Sort sequences by length in descending order
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        sequences, labels = zip(*batch)

        # Get sequence lengths
        lengths = [len(seq) for seq in sequences]

        # Pad sequences
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

        # Add the feature dimension
        sequences_padded = sequences_padded.unsqueeze(-1)  # Add a singleton dimension for features

        return sequences_padded, lengths

    def train_autoencoder(self, X_train, X_val, X_test, batch_size=32, epochs=100, learning_rate=0.001, patience=5, min_delta=0.0005):

        # Apply the weight initialization to your model
        self.apply(self.init_weights)

        # Create DataLoader
        train_data = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(train_data, train_data)  # autoencoders use same data for input and output
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        val_data = torch.FloatTensor(X_val)
        val_dataset = TensorDataset(val_data, val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        test_data = torch.FloatTensor(X_test)
        test_dataset = TensorDataset(test_data, test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        pbar = tqdm(total=epochs)

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch_data, lengths in train_loader:
                optimizer.zero_grad()
                outputs, l1_loss, _ = self(batch_data, lengths)
                loss = criterion(outputs, batch_data) + l1_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data, lengths in val_loader:
                    outputs, l1_loss, _ = self(batch_data, lengths)
                    loss = criterion(outputs, batch_data) + l1_loss
                    val_loss += loss.item()

            if early_stopper.early_stop(val_loss):
                print("Early stopping!")
                break

            pbar.set_description(f"train_Loss:{train_loss/len(train_loader):.4f}, val_loss:{val_loss/len(val_loader):.4f} (ES:{early_stopper.counter})")
            pbar.update(1)
        pbar.close()

    @staticmethod
    def reshape_to_squareish_matrix(vector):
        length = len(vector)
        sqrt_length = int(math.sqrt(length))
        rows = sqrt_length
        while length % rows != 0:
            rows -= 1
        cols = length // rows
        reshaped_matrix = np.reshape(vector, (rows, cols))
        return reshaped_matrix

    @staticmethod
    def init_weights(m):
        """
        Initialize the weights of the model.

        Parameters:
            m (torch.nn.Module): A PyTorch layer or model
        """

        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == torch.nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)

    def plot_examples_pytorch(self, X_test, show_diff=False, num_samples=9, figsize=(10, 6)):

        model = self
        model.eval()

        test_data = torch.FloatTensor(X_test)
        test_dataset = TensorDataset(test_data, test_data)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=self.collate_fn)

        Y_test = []
        latent_outputs = []

        with torch.no_grad():
            for batch_data, lengths in test_loader:
                # print(batch_data)
                outputs, l1_loss, latent_representation = model(batch_data, lengths)
                # print(outputs)

                outputs = np.array(outputs).squeeze()

                Y_test.append(outputs)
                latent_outputs.append(latent_representation)

        # print(Y_test)
        #
        # print(len(Y_test), len(Y_test[0]), len(X_test))
        Y_test = Y_test[0]
        # print(len(Y_test))
        # print(Y_test[0])

        # Y_test = torch.cat(Y_test, dim=0).numpy()
        # latent_outputs = np.array(latent_outputs)
        # print(latent_outputs.shape, latent_outputs[0].shape)

        num_rounds = 1
        if type(num_samples) != int:
            num_rounds, num_samples = num_samples

        for nr in range(num_rounds):
            fig, axx = plt.subplots(3, num_samples, figsize=figsize, sharey=False)

            for i, idx in enumerate(np.random.randint(0, len(X_test) - 1, size=num_samples)):

                inp = X_test[idx]
                out = Y_test[idx]
                # print(out)

                inp = np.trim_zeros(inp, trim="b")
                out = out[0:len(inp)]

                axx[0, i].plot(inp, alpha=0.75, color="black")
                axx[0, i].plot(out, alpha=0.75, linestyle="--", color="darkgreen")
                axx[1, i].plot(inp - out)

                axx[0, i].sharey(axx[1, i])

                # latent_output = latent_outputs[idx]
                # latent_output = self.reshape_to_squareish_matrix(latent_output)
                #
                # cmap = 'binary' if not show_diff else 'bwr'
                # vmin = 0 if not show_diff else -1
                # axx[2, i].imshow(latent_output, cmap=cmap, interpolation='nearest', aspect='auto',
                #                  vmin=vmin, vmax=1)
                # axx[2, i].get_xaxis().set_visible(False)
                # axx[2, i].get_yaxis().set_visible(False)


                axx[0, i].get_xaxis().set_visible(False)
                axx[1, i].get_xaxis().set_visible(False)

                if i != 0:
                    axx[0, i].get_yaxis().set_visible(False)
                    axx[1, i].get_yaxis().set_visible(False)

            axx[0, 0].set_ylabel("IN/OUT", fontweight=600)
            axx[1, 0].set_ylabel("error", fontweight=600)

class AutoEncoder(nn.Module):
    def __init__(self, target_length, dropout=0.15, l1_reg=0.0001, latent_size=64*6, add_noise=None):
        super(AutoEncoder, self).__init__()

        self.l1_reg = l1_reg

        self.encoder, self.decoder = self.define_layers(dropout=dropout, add_noise=add_noise, target_length=target_length)

        # Manually defining a linear layer to serve as the dense layer for the encoder output
        self.latent_size = latent_size
        self.dense_layer = None
        self.dense_layer_out = None
        self.add_noise = add_noise

    def define_layers(self, dropout=None, add_noise=None, target_length=18):

        encoder_layers = []
        if dropout is not None:
            encoder_layers += [nn.Dropout(dropout)]
        if add_noise is not None:
            encoder_layers += [GaussianNoise(std=add_noise)]
        encoder_layers += [nn.Conv1d(1, 128, 3, padding=1),]
        encoder_layers += [nn.ReLU(),]
        encoder_layers += [nn.MaxPool1d(2),]
        encoder_layers += [nn.Conv1d(128, 64, 3, padding=1),]
        encoder_layers += [nn.ReLU(),]
        encoder_layers += [nn.MaxPool1d(2),]
        encoder = nn.Sequential(*encoder_layers)

        decoder = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            CustomUpsample(target_length=target_length),
            nn.Conv1d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )

        return encoder, decoder

    def forward(self, x):
        x = self.encoder(x)

        shape_before_flatten = x.shape[1:]
        flattened_dim = torch.prod(torch.tensor(shape_before_flatten))

        if self.dense_layer is None:
            self.dense_layer = nn.Linear(flattened_dim.item(), 64*6).to(x.device)
            self.dense_layer_out = nn.Linear(64*6, flattened_dim.item()).to(x.device)

        x = x.view(-1, flattened_dim)

        # Apply the dense layer
        x = self.dense_layer(x)

        # Make the output binary
        x = torch.sigmoid(x)
        x = torch.round(x)

        # Save the encoder output for later use
        encoder_output = x

        # go back to initial size
        x = self.dense_layer_out(x)

        # Add L1 regularization to the encoder output
        l1_loss = self.l1_reg * torch.norm(x, 1)

        x = x.view(-1, *shape_before_flatten)
        x = self.decoder(x)

        return x, l1_loss, encoder_output

    def split_dataset(self, data, val_split=0.1, train_split=0.8, seed=None):

        from torch.utils.data import random_split

        # Assuming you have a PyTorch Dataset object `full_dataset`
        # This could be an instance of a custom dataset class, or one of the built-in classes like `torchvision.datasets.MNIST`

        if seed is not None:
            torch.manual_seed(seed)

        # Define the proportions
        total_size = len(data)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset

    def train_autoencoder(self, X_train, X_val, X_test, patience=5, min_delta=0.0005, epochs=100, learning_rate=0.001, batch_size=32):

        # Create DataLoader
        train_data = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(train_data, train_data)  # autoencoders use same data for input and output
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_data = torch.FloatTensor(X_val)
        val_dataset = TensorDataset(val_data, val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_data = torch.FloatTensor(X_test)
        test_dataset = TensorDataset(test_data, test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = self
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop

        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_data, _ in train_loader:  # autoencoders don't use labels
                batch_data = batch_data.unsqueeze(1)  # add channel dimension

                # Forward pass
                outputs, l1_loss, encoded = model(batch_data)

                # Compute loss
                reconstruction_loss = criterion(outputs, batch_data)
                loss = reconstruction_loss + l1_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    batch_data = batch_data.unsqueeze(1)  # add channel dimension
                    outputs, l1_loss, encoded = model(batch_data)

                    reconstruction_loss = criterion(outputs, batch_data)
                    loss = reconstruction_loss + l1_loss
                    val_loss += loss.item()

            # Early stopping logic
            if early_stopper.early_stop(val_loss):
                print("Early stopping!")
                break

            pbar.set_description(f"train_Loss:{train_loss/len(train_loader):.4f}, val_loss:{val_loss/len(val_loader):.4f}")
            pbar.update(1)
        pbar.close()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_data, _ in test_loader:
                batch_data = batch_data.unsqueeze(1)  # add channel dimension
                outputs, l1_loss, encoded = model(batch_data)

                reconstruction_loss = criterion(outputs, batch_data)
                loss = reconstruction_loss + l1_loss
                test_loss += loss.item()

            print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    @staticmethod
    def reshape_to_squareish_matrix(vector):
        """
        Reshapes a 1D vector to a square-ish 2D matrix.
        """

        import math

        length = len(vector)
        # Find the closest integer square root of the length
        sqrt_length = int(math.sqrt(length))

        # Find the dimensions of the reshaped matrix
        rows = sqrt_length
        while length % rows != 0:
            rows -= 1
        cols = length // rows

        # Reshape the vector
        reshaped_matrix = np.reshape(vector, (rows, cols))

        return reshaped_matrix

    def plot_examples_pytorch(self, X_test, Y_test=None,
                              show_diff=False, num_samples=9, figsize=(10, 6)):

        model = self

        # Convert PyTorch tensor to numpy
        # X_test = torch.from_numpy(X_test).float()
        X_test = np.array(list(X_test))
        print(f"X test shape: {X_test.shape}")

        # Make predictions if Y_test is not provided
        if Y_test is None:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # add channel dimension
                output_tensor, l1_loss, encoder_output = model(input_tensor)
            Y_test = output_tensor.numpy().squeeze()
            encoder_output = encoder_output.numpy().squeeze()

            if show_diff:
                consensus = np.mean(encoder_output, axis=0)
                print(f"consesus shape: {consensus.shape}")
                encoder_output = encoder_output - consensus

            print(f"latent shape: {encoder_output.shape}")

        else:
            Y_test = Y_test.numpy().squeeze()

        num_rounds = 1
        if type(num_samples) != int:
            num_rounds, num_samples = num_samples

        for nr in range(num_rounds):
            fig, axx = plt.subplots(3, num_samples, figsize=figsize, sharey=False)

            for i, idx in enumerate(np.random.randint(0, len(X_test) - 1, size=num_samples)):
                inp = X_test[idx, :]
                out = Y_test[idx, :]

                inp = np.trim_zeros(inp, trim="b")
                out = out[0:len(inp)]

                axx[0, i].plot(inp, alpha=0.75, color="black")
                axx[0, i].plot(out, alpha=0.75, linestyle="--", color="darkgreen")
                axx[1, i].plot(inp - out)

                axx[0, i].sharey(axx[1, i])

                latent_output = encoder_output[idx, :]
                latent_output = self.reshape_to_squareish_matrix(latent_output)

                cmap = 'binary' if not show_diff else 'bwr'
                vmin = 0 if not show_diff else -1
                axx[2, i].imshow(latent_output, cmap=cmap, interpolation='nearest', aspect='auto',
                                 vmin=vmin, vmax=1)
                axx[2, i].get_xaxis().set_visible(False)
                axx[2, i].get_yaxis().set_visible(False)


                axx[0, i].get_xaxis().set_visible(False)
                axx[1, i].get_xaxis().set_visible(False)

                if i != 0:
                    axx[0, i].get_yaxis().set_visible(False)
                    axx[1, i].get_yaxis().set_visible(False)

            axx[0, 0].set_ylabel("IN/OUT", fontweight=600)
            axx[1, 0].set_ylabel("error", fontweight=600)

class CNN:

    """ embeds data in a latent space of defined size
    """

    def __init__(self, encoder=None, autoencoder=None):

        self.encoder = encoder
        self.autoencoder = autoencoder

        self.history = None
        self.X_test = None
        self.Y_test = None
        self.MSE = None

    def train(self, data, train_split=0.9, validation_split=0.1,
              loss='mse', dropout=None, regularize_latent=None,
              epochs=50, batch_size=64, patience=5, min_delta=0.0005, monitor="val_loss"):

        assert isinstance(data, np.ndarray), f"please provide data in 'np.ndarray' format instead of {type(data)}"
        assert isinstance(data.dtype, object), f"please provide data in format other than 'object' type "
        assert not np.isnan(data).any(), "data contains NaN values. Please exclude data points or fill NaN values (eg. np.nan_to_num)"

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

        autoencoder.summary(line_length=100)

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
        logging.info(f"Quality of encoding > MSE: {np.mean(MSE)}") # :.4f

        self.X_test = X_test
        self.Y_test = Y_test
        self.MSE = MSE

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

        return fig

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

            return fig

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


class ClusterTree():

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
