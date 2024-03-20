import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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
