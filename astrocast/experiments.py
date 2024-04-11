import logging
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Dataset

from astrocast.analysis import Plotting
from astrocast.autoencoders import CNN_Autoencoder, PaddedDataLoader, TimeSeriesRnnAE
from astrocast.clustering import CoincidenceDetection, Discriminator, Linkage
from astrocast.helper import DummyGenerator, SignalGenerator
from astrocast.reduction import FeatureExtraction


class Experiments:
    
    def __init__(self, dummy_parameters: Union[dict, List[dict]], replicates: int = 1):
        self.dummy_parameters = dummy_parameters
        
        self.experiments = self._generate(replicates=replicates)
    
    def __getitem__(self, item):
        return self.experiments[item]
    
    def __len__(self):
        return len(self.experiments)
    
    def plot_traces(self, axx=None):
        
        unique_objects = [eObj for eObj in self.experiments if eObj.replicate == 0]
        
        N = len(unique_objects)
        if axx is None:
            fig, axx = plt.subplots(1, N, figsize=(3 * N, 3))
        
        for i, eObj in enumerate(unique_objects):
            _ = eObj.plot.plot_traces(num_samples=4, by="group", alpha=.9, linestyle="--",
                                      title=f"{eObj.name}", ax=axx[i])
    
    def _generate(self, replicates=1):
        
        experiments = []
        i = 0
        for param in self.dummy_parameters:
            
            # extract name
            if "name" in param:
                name = param["name"]
                del param["name"]
            else:
                name = i
            
            # extract timings
            if "timings" in param:
                timings = param["timings"]
            else:
                timings = None
            
            # extract generators
            if "generators" in param and not isinstance(param["generators"], SignalGenerator):
                param["generators"] = [SignalGenerator(**gen_param) for gen_param in param["generators"]]
            
            for replicate in range(replicates):
                
                dg = DummyGenerator(**param)
                eObj = dg.get_events()
                
                eObj.experiment_id = i
                eObj.name = name
                eObj.plot = Plotting(eObj)
                eObj.n_groups = len(eObj.events.group.unique())
                eObj.timings = timings
                eObj.replicate = replicate
                
                eObj.embeddings = {}
                eObj.results = []
                
                experiments.append(eObj)
                
                i += 1
        
        return experiments
    
    def create_embedding(self, embeddings: dict):
        
        for k, kwargs in embeddings.items():
            for eObj in self.experiments:
                
                if k == "FExt":
                    
                    fe = FeatureExtraction(eObj)
                    
                    embedding = fe.all_features(dropna=True)
                    embedding = embedding.values.astype(float)
                
                elif k == "CNN":
                    
                    if eObj._is_ragged():
                        logging.warning(f"Skipping object, since events are ragged.")
                        continue
                    
                    # create CNNAutoEncoder
                    target_length = len(eObj.events.iloc[0].trace)
                    cnn = CNN_Autoencoder(target_length=target_length, use_cuda=True)
                    
                    # prepare data
                    data = np.array(eObj.events.trace.tolist())
                    X_train, X_val, X_test = cnn.split_dataset(data=data)
                    
                    # train
                    cnn.train_autoencoder(X_train=X_train, X_val=X_val, epochs=25)
                    
                    # embedding
                    embedding = cnn.embed(data)
                    embedding = embedding.astype(float)
                
                elif k == "RNN":
                    
                    # create data loader
                    pdl = PaddedDataLoader(data=eObj.events.trace)
                    X_train, X_val, X_test = pdl.get_datasets(batch_size=16,
                                                              val_size=0.1,
                                                              test_size=0.05)
                    # train RecurrentAutoEncoder
                    tRAE = TimeSeriesRnnAE(use_cuda=True)
                    tRAE.train_epochs(dataloader_train=X_train,
                                      dataloader_val=X_val,
                                      num_epochs=10,
                                      patience=10,
                                      safe_after_epoch=None,
                                      show_mode='progress'
                                      )
                    
                    # embedding
                    X = pdl.get_dataloader(data=eObj.events.trace, batch_size=16, shuffle=False)
                    _, _, embedding, _ = tRAE.embedd(X)
                    
                    embedding = np.array(embedding).astype(float)
                
                else:
                    raise ValueError(f"unknown embedding type: {k}")
                
                eObj.embeddings[k] = embedding
    
    @staticmethod
    def _conditional_contrasts_classifier(eObj, embedding, embedding_name):
        
        score_train = dict(evaluation_type="conditional_contrasts", cluster_type="classifier",
                           type="RandomForestClassifier", metric="accuracy", embedding=embedding_name)
        score_test = score_train.copy()
        
        score_train["data_split"] = "train"
        score_test["data_split"] = "test"
        
        discr = Discriminator(eObj)
        
        _ = discr.train_classifier(embedding=embedding, category_vector=eObj.events.group.tolist(),
                                   classifier=score_train["type"])
        scores = discr.evaluate(show_plot=False)
        
        score_train["score"] = scores["train"][score_train["metric"]]
        score_train["cm"] = scores["train"]["cm"]
        eObj.results.append(score_train)
        
        score_test["score"] = scores["test"][score_test["metric"]]
        score_test["cm"] = scores["test"]["cm"]
        eObj.results.append(score_test)
    
    @staticmethod
    def _conditional_contrasts_hierarchical(eObj):
        
        score = dict(evaluation_type="conditional_contrasts", cluster_type="hierarchy",
                     metric="rand_score", data_split="test")
        
        link = Linkage()
        
        num_groups = eObj.n_groups
        for correlation_type in ['pearson', 'dtw']:
            
            score_ = score.copy()
            
            _, cluster_lookup_table = link.get_barycenters(eObj.events,
                                                           cutoff=num_groups, criterion='maxclust',
                                                           distance_type=correlation_type
                                                           )
            
            true_labels = eObj.events.group.tolist()
            predicted_labels = [cluster_lookup_table[n] - 1 for n in range(len(true_labels))]
            
            scores = Discriminator.compute_scores(true_labels, predicted_labels, scoring="clustering")
            
            score_["type"] = "distance"
            score_["embedding"] = correlation_type
            score_["cm"] = None
            score_["score"] = scores[score_["metric"]]
            
            eObj.results.append(score_)
    
    def conditional_contrasts(self):
        
        for eObj in self.experiments:
            for embedding_name, embedding in eObj.embeddings.items():
                
                # Classifier - predict condition
                self._conditional_contrasts_classifier(eObj, embedding, embedding_name)
                
                # Hierarchical - predict condition
                self._conditional_contrasts_hierarchical(eObj)
    
    @staticmethod
    def _coincidence_detection_classifier(eObj, embedding, timing, embedding_name):
        
        score_train = dict(evaluation_type="coincidence_detection", cluster_type="classifier",
                           metric="accuracy", type="RandomForestClassifier",
                           embedding=f"{embedding_name}_c")
        
        score_test = score_train.copy()
        
        score_train["data_split"] = "train"
        score_test["data_split"] = "test"
        
        cDetect = CoincidenceDetection(events=eObj, incidences=timing, embedding=embedding)
        _, scores = cDetect.predict_coincidence(binary_classification=True)
        
        score_train["score"] = scores["train"][score_train["metric"]]
        score_train["cm"] = scores["train"]["cm"]
        eObj.results.append(score_train)
        
        score_test["score"] = scores["test"][score_test["metric"]]
        score_test["cm"] = scores["test"]["cm"]
        eObj.results.append(score_test)
    
    @staticmethod
    def _coincidence_detection_regression(eObj, timing, embedding, embedding_name):
        
        score = dict(evaluation_type="coincidence_detection", cluster_type="regression",
                     metric="regression_error", embedding=f"{embedding_name}_r",
                     type="RandomForestRegressor", cm=None)
        
        score_train = score.copy()
        score_train["data_split"] = "train"
        score_test = score.copy()
        score_test["data_split"] = "test"
        
        cDetect = CoincidenceDetection(events=eObj, incidences=timing, embedding=embedding)
        _, scores = cDetect.predict_incidence_location()
        
        score_train["score"] = scores["train"]["score"]
        eObj.results.append(score_train)
        
        score_test["score"] = scores["test"]["score"]
        eObj.results.append(score_test)
    
    def coincidence_detection(self):
        
        for eObj in self.experiments:
            
            # extract timing
            timings = eObj.timings
            if timings is None:
                raise ValueError(f"No timings present in experiment {eObj}")
            
            timings = [timing for timing in timings if timing is not None]
            if len(timings) < 1:
                raise ValueError(f"No timings present in experiment {timings}")
            
            elif len(timings) > 1:
                timings = list(set(timings))
                if len(timings) != 1:
                    raise ValueError(f"Too many timings present in experiment {timings}")
            
            timing = timings[0]
            
            # detect coincidence
            for embedding_name, embedding in eObj.embeddings.items():
                
                self._coincidence_detection_classifier(eObj, embedding, timing, embedding_name)
                self._coincidence_detection_regression(eObj, timing, embedding, embedding_name)
    
    def get_results(self):
        
        dataframes = []
        for eObj in self.experiments:
            
            df = pd.DataFrame(eObj.results)
            df["name"] = eObj.name
            df["n_groups"] = eObj.n_groups
            df["e_id"] = eObj.experiment_id
            
            dataframes.append(df)
        
        return pd.concat(dataframes, axis=0)
    
    @staticmethod
    def plot_heatmap(df: pd.DataFrame, evaluation_type: str, index: str, columns: str, group_by: str,
                     padding: int = 23, show_legend: bool = True, title: str = None,
                     heatmap_colors: Tuple[int, int] = (14, 145), row_colors=None, ax=None):
        """
        Generates and displays a heatmap based on the specified DataFrame and parameters.

        This function filters the DataFrame based on the evaluation_type, pivots the data for heatmap generation, and
        uses seaborn to plot the heatmap. It allows for customization of the heatmap's appearance, including color scheme,
        legend display, and axis labels.

        Args:
            df: The source DataFrame containing the data.
            evaluation_type: The evaluation type to filter the DataFrame on.
            index: The name of the column to be used as the index of the pivot table.
            columns: The name of the column to be used as the columns of the pivot table.
            heatmap_colors: A tuple representing the start and end colors of the heatmap's diverging palette.
            padding: The padding for the y-axis labels.
            show_legend: A boolean indicating whether to display the legend.
            title: The title of the heatmap. If None, no title is set.

        Returns:
            None. Displays a heatmap based on the specified parameters.
        """
        # Filter DataFrame based on evaluation type
        selected = df[df.evaluation_type == evaluation_type]
        selected = selected.sort_values(by=group_by, ascending=True)
        
        # Generate pivot table
        pivot = selected.pivot_table(index=index, columns=columns, values='score', sort=False,
                                     aggfunc='min')[["train", "test"]]
        
        #
        
        # Prepare color dictionary for cluster types
        unique_types = selected[group_by].unique()
        if row_colors is None:
            row_colors = sns.color_palette('husl', len(unique_types))
        color_dict = {unique_types[i]: row_colors[i] for i in range(len(unique_types))}
        
        # Map embeddings to colors
        temp = selected[[group_by, index]].drop_duplicates()
        temp["color"] = temp[group_by].apply(lambda x: tuple(color_dict[x]))
        embedding_to_color = temp.set_index(index).to_dict()["color"]
        
        # Plotting
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        cmap = sns.diverging_palette(*heatmap_colors, s=60, as_cmap=True)
        cmap.set_bad(".75")
        
        sns.heatmap(data=pivot, cbar=False, annot=True,
                    cmap=cmap,
                    vmax=1, vmin=0.5 if pivot.min().min() > 0 else -1,
                    annot_kws={'color': "black", "size": 12}, ax=ax)
        
        # Add title if provided
        if title is not None:
            ax.set_title(title)
        
        # Add grouping colors to the y-axis
        ax.tick_params(axis='y', which='major', pad=padding, length=0)
        for i, emb in enumerate(pivot.index):
            color = embedding_to_color[emb]
            ax.add_patch(plt.Rectangle(xy=(-0.06, i), width=0.05, height=1, color=color, lw=0,
                                       transform=ax.get_yaxis_transform(), clip_on=False))
        
        # Add legend for cluster types
        if show_legend:
            custom_lines = [Line2D([0], [0], color=color_dict[ct], lw=4) for ct in color_dict]
            ax.legend(custom_lines, color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))


#######################
# Event timing analysis
class EventTimingAnalysis:
    """Encapsulates the workflow for event timing analysis using CNNs.

    This class handles data preprocessing, model training, evaluation, and prediction.

    Attributes:
        obj: An object that provides access to the data.
        cluster_column: The column to use for clustering.
        timings: A list of timing events.
        split: The ratio to split the data into training and testing sets.
        window: The window size for event timing analysis.
        window_slide: The step size for sliding the window.
        exclude_empty_window: Flag to exclude windows with no events.
        skip_first_n_frames: Number of initial frames to skip in each window, if any.
    """
    
    def __init__(self, obj, cluster_column: str, timings: List[int], split: float = 0.8,
                 window: int = 20, window_slide: int = 1,
                 n_layers=3, n_kernels=64,
                 batch_size: int = 32, dropout: float = None,
                 exclude_empty_window: bool = True, skip_first_n_frames: int = None):
        self.obj = obj
        self.cluster_column = cluster_column
        self.timings = timings
        self.split = split
        self.batch_size = batch_size
        self.window = window
        self.window_slide = window_slide
        self.exclude_empty_window = exclude_empty_window
        self.skip_first_n_frames = skip_first_n_frames
        self.model = None
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = self._preprocess_data()
        self.train_loader, self.val_loader = self._create_datasets_and_loaders()
        
        self.model = TimingCNN(n_layers=n_layers, n_kernels=n_kernels, input_channels=1, window_size=window,
                               clusters=self.X_train.shape[-1],
                               dropout=dropout)
    
    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses the data for training and testing.

        Returns:
            A tuple containing numpy arrays for embeddings and binary timings.
        """
        
        # and the count of cluster events per frame (frame x cluster)
        time_map_group = self.obj.get_time_map_by_cluster(cluster_column=self.cluster_column)
        
        # Filter embeddings and calculate timing floats only for windows with timing events
        embeddings = []
        binary_timings = []
        
        for i, start_frame in enumerate(range(0, time_map_group.shape[0] - self.window + 1, self.window_slide)):
            window_timings = [timing for timing in self.timings if start_frame <= timing < start_frame + self.window]
            binary_array = np.zeros(self.window, dtype=int)  # Initialize a binary array for the window
            
            if window_timings:  # If there are events in the current window
                # Set the corresponding positions of the timings to 1
                for timing in window_timings:
                    relative_frame = timing - start_frame
                    binary_array[relative_frame] = 1
            
            # Only append windows that contain at least one timing event
            if binary_array.sum() <= 0 and self.exclude_empty_window:
                continue
            
            if self.skip_first_n_frames is not None and np.sum(binary_array[:self.skip_first_n_frames]) > 0:
                continue
            
            binary_timings.append(binary_array)
            embeddings.append(time_map_group[start_frame:start_frame + self.window, :])
        
        # Calculate index split
        idx_split = int(self.split * len(embeddings))
        
        # Convert to numpy arrays for further processing
        embeddings = np.array(embeddings)
        binary_timings = np.array(binary_timings)
        
        # Split the filtered data into training and testing sets
        X_train, X_test = embeddings[:idx_split], embeddings[idx_split:]
        Y_train, Y_test = binary_timings[:idx_split], binary_timings[idx_split:]
        
        # expand channel dimension
        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)
        
        return X_train, X_test, Y_train, Y_test
    
    def _create_datasets_and_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates PyTorch datasets and data loaders for training and validation.

        Args:
            X_train, Y_train, X_test, Y_test: Arrays containing training and testing data and labels.

        Returns:
            A tuple of DataLoaders for training and validation.
        """
        # Example dataset creation
        train_dataset = TimingDataset(self.X_train, self.Y_train)
        val_dataset = TimingDataset(self.X_test, self.Y_test)
        
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, epochs: int = 10, learning_rate: float = 0.0001, log: Literal['plot', 'verbose'] = "verbose"):
        """Trains the CNN model.

        Args:
            epochs: Number of epochs to train.
            learning_rate: Learning rate for training
            log: Logging mode. 'verbose' for print statements, 'plot' for plotting.
        """
        
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        
        criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
        
        running_losses = []
        validation_losses = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                
                try:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Output shape: {outputs.shape}; Label shape: {labels.shape}")
                    raise  # Reraise the exception to see the full traceback and stop execution
                
                running_loss += loss.item()
            
            running_losses.append(running_loss)
            if log == "verbose":
                print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
            
            # Validation step
            model.eval()
            val_loss = 0.0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
            if log == "verbose":
                print(f'Validation Loss: {val_loss / len(val_loader)}')
            
            validation_losses.append(val_loss)
        
        if log == "plot":
            fig, ax = plt.subplots()
            ax.plot(running_losses, color="blue", label="Training Loss")
            ax.plot(validation_losses, color="green", label="Validation Loss")
            ax.legend()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
        
        return running_losses, validation_losses
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predicts binary timings for given embeddings using the trained model.

        Args:
            embeddings: A numpy array of embeddings.

        Returns:
            A numpy array of predicted binary timings.
        """
        
        model = self.model
        
        # Ensure the model is in evaluation mode
        model.eval()
        
        # Check if the input is a single sample; if so, add the batch dimension
        if embeddings.ndim == 2:
            # Add a new axis at index 0 to add the batch dimension
            embeddings = embeddings[np.newaxis, ...]
        
        # Convert embeddings to a PyTorch tensor and add a channel dimension
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)
        
        # Move the tensor to the same device as the model
        embeddings_tensor = embeddings_tensor.to(next(model.parameters()).device)
        
        # Perform the prediction
        with torch.no_grad():
            outputs = model(embeddings_tensor)
        
        # Apply a threshold to get binary outputs, adjust the threshold if needed
        predicted_binary_timings = torch.round(outputs).cpu().numpy()
        
        # If there was only one input embedding, remove the batch dimension
        if predicted_binary_timings.shape[0] == 1:
            predicted_binary_timings = predicted_binary_timings.squeeze(0)
        
        return predicted_binary_timings
    
    @staticmethod
    def diff_to_percentage(arr: np.ndarray, as_string: bool = True):
        """
        Converts differences in predictions to percentages of false negatives, false positives, and overall falses.

        This function takes an ndarray representing the differences between predicted and actual values,
        where 1 indicates a false negative, -1 indicates a false positive, and 0 indicates a correct prediction.
        It calculates the percentage of false negatives, false positives, and a combined metric of both.

        Args:
          arr: A 2D numpy array of differences.
          as_string: A boolean flag to determine the format of the output. If True, the output is a formatted string;
                     otherwise, it returns a list of percentages.

        Returns:
          A formatted string or a list of percentages (false, false positive, false negative) based on the `as_string` flag.

        """
        false_negative = np.sum(arr == 1, axis=1)
        false_positive = np.sum(arr == -1, axis=1)
        false_ = np.bitwise_or(false_negative, false_positive)
        
        perc_false_negative = np.sum(false_negative) / len(false_negative) * 100
        perc_false_positive = np.sum(false_positive) / len(false_positive) * 100
        perc_false_ = np.sum(false_) / len(false_) * 100
        
        if as_string:
            return f"F: {perc_false_:.1f}%; fp: {perc_false_positive:.1f}%; fn: {perc_false_negative:.1f}%"
        else:
            return [perc_false_, perc_false_positive, perc_false_negative]
    
    def plot_heatmap(self, x_arr: Union[np.ndarray, str], y_arr: Union[np.ndarray, str], cmap="vlag",
                     label: str = None, figsize: Tuple[int, int] = (5, 5), ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(x_arr, str):
            
            mapping = {
                "train": self.X_train,
                "test":  self.X_test
                }
            
            if x_arr not in mapping:
                raise KeyError(f"Cannot find {x_arr}, choose one of {mapping.keys()}")
            else:
                x_arr = mapping[x_arr]
        
        if isinstance(y_arr, str):
            
            mapping = {
                "train": self.Y_train,
                "test":  self.Y_test
                }
            
            if y_arr not in mapping:
                raise KeyError(f"Cannot find {y_arr}, choose one of {mapping.keys()}")
            else:
                y_arr = mapping[y_arr]
        
        pred = self.predict(x_arr[:, 0, :, :])
        
        if np.sum(pred) <= 0:
            logging.warning(f"No predictions were made. Please check the training input.")
        
        diff = y_arr - pred
        
        cmap = sns.color_palette(cmap, as_cmap=True)
        sns.heatmap(diff, vmin=-1, vmax=1, center=False, robust=False, cbar=True, cmap=cmap, ax=ax)
        
        if label is not None:
            _ = ax.set_title(f"{label} ({self.diff_to_percentage(diff, as_string=True)})")
    
    @staticmethod
    def plot_event_matches(y_true: np.ndarray, y_pred: np.ndarray, use_true_as_reference: bool, normalize: bool,
                           plot_unmatched: bool = False,
                           ax: plt.Axes = None, figsize=(5, 4), **kwargs):
        """
        Plots the true vs. predicted frames of events in a regression-like manner.

        Args:
            y_true: 2D numpy array of true binary sequences indicating event occurrences.
            y_pred: 2D numpy array of predicted binary sequences, same shape as y_true.
            use_true_as_reference: If True, true event frames are used as reference; otherwise, predicted frames are used.
            normalize: If True, event frame numbers are normalized to the range [0, 1].
            ax: Optional matplotlib axes object for plotting. If None, a new figure is created.
            figsize: Figure size if a new figure is created.
            plot_unmatched: If true, the unmatched timings are plotted at y=-1
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        reference, comparison = (y_true, y_pred) if use_true_as_reference else (y_pred, y_true)
        n_observations, window_length = reference.shape
        unmatched_count = 0
        
        comp_events = np.zeros(1)
        for i in range(n_observations):
            ref_events = np.where(reference[i] == 1)[0]
            comp_events = np.where(comparison[i] == 1)[0]
            
            for event in ref_events:
                if comp_events.size > 0:
                    closest_event = comp_events[np.argmin(np.abs(comp_events - event))]
                    comp_events = np.delete(comp_events, np.argmin(np.abs(comp_events - event)))
                    
                    if normalize:
                        event /= window_length
                        closest_event /= window_length
                    
                    ax.plot(event, closest_event, **kwargs)
                
                else:
                    
                    if normalize:
                        event /= window_length
                    
                    if plot_unmatched:
                        kwargs_ = kwargs.copy()
                        kwargs_["color"] = "gray"
                        ax.plot(event, -1, **kwargs_)
                    
                    unmatched_count += 1
        
        # Additional unmatched events in comparison
        unmatched_count += len(comp_events)
        
        if normalize:
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.25)  # Center line for perfect match
        else:
            ax.plot([0, window_length], [0, window_length], 'k--', alpha=0.25)
        
        ax.set_xlabel('True Frame' if use_true_as_reference else 'Predicted Frame')
        ax.set_ylabel('Predicted Frame' if use_true_as_reference else 'True Frame')
        
        # Log warning for unmatched events
        if unmatched_count > 0:
            logging.warning(
                    f"Number of unmatched events: {unmatched_count} ({unmatched_count / len(y_true) * 100:.1f}%)")
        
        return unmatched_count
    
    def plot_multi_event_matches(self, x_arrays: List[np.ndarray], y_arrays: List[np.ndarray],
                                 labels: List[str] = None, colors=None, markers: Union[str, List[str]] = 'x',
                                 alpha: float = 0.25, legend_loc: str = "best", plot_unmatched: bool = False,
                                 use_true_as_reference: bool = True, normalize: bool = False,
                                 ax: plt.Axes = None, figsize=(5, 4), **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if colors is None:
            colors = sns.color_palette("husl", len(x_arrays))
        
        if not isinstance(markers, list):
            markers = [markers] * len(x_arrays)
        
        x_arrays = [arr if len(arr) < 5 else arr[:, 0, :, :] for arr in x_arrays]
        
        unmatched_count = []
        for i, x_arr in enumerate(x_arrays):
            
            y_arr = y_arrays[i]
            pred = self.predict(x_arr)
            
            unmatched = self.plot_event_matches(y_arr, pred, use_true_as_reference=use_true_as_reference,
                                                plot_unmatched=plot_unmatched,
                                                normalize=normalize, ax=ax, color=colors[i], marker=markers[i],
                                                alpha=alpha, **kwargs)
            unmatched_count.append(unmatched)
        
        if labels is not None:
            
            legend_elements = []
            for i in range(len(x_arrays)):
                lbl = f"{labels[i]} ({100 - unmatched_count[i] / len(y_arrays[i]) * 100:.1f}%)"
                line = Line2D([0], [0], color=colors[i], marker=markers[i], label=lbl)
                legend_elements.append(line)
            
            ax.legend(handles=legend_elements, loc=legend_loc)


class TimingCNN(nn.Module):
    """A Convolutional Neural Network for analyzing timing events."""
    
    def __init__(self, n_layers: int, n_kernels: int, input_channels: int, window_size: int, clusters: int,
                 dropout=0.1):
        super(TimingCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=input_channels if i == 0 else n_kernels,
                                              out_channels=n_kernels,
                                              kernel_size=(3, 3),
                                              padding='same') for i in range(n_layers)])
        
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        
        # Assuming pooling is applied in each layer and halves both dimensions each time
        final_dim = window_size // 2 ** n_layers
        final_clusters = clusters // 2 ** n_layers
        # Calculate the correct number of flattened features
        flattened_size = n_kernels * final_dim * final_clusters
        
        self.fc = nn.Linear(flattened_size, window_size)  # Output layer now has 'window_size' units
    
    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            
            if self.dropout is not None:
                x = self.dropout(x)
            
            x = F.max_pool2d(x, (2, 2))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # Apply sigmoid to output to get values between 0 and 1
        x = torch.sigmoid(x)
        return x


class TimingDataset(Dataset):
    """Custom Dataset for loading embeddings and binary timing arrays."""
    
    def __init__(self, embeddings: np.ndarray, binary_timings: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.binary_timings = torch.tensor(binary_timings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.binary_timings[idx]
