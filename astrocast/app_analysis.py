import logging
import traceback
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import shiny.experimental.ui as xui
import sklearn.decomposition
import yaml
from matplotlib import pyplot as plt, gridspec
from matplotlib.lines import Line2D
from numba import NumbaDeprecationWarning
from scipy.cluster.hierarchy import dendrogram
from shiny import App, ui, render, reactive, req
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

from astrocast.analysis import Events, Video
from astrocast.autoencoders import TimeSeriesRnnAE, PaddedDataLoader, CNN_Autoencoder
from astrocast.clustering import CoincidenceDetection, Discriminator, Linkage
from astrocast.helper import is_ragged
from astrocast.preparation import IO
from astrocast.reduction import FeatureExtraction

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)
    try:
        import umap
    except ImportError:
        logging.error(f"Unable to import umap package. Some functionality will not work as expected. "
                      f"If you are using astroCAST on MacOS this is expected.")


class Analysis:

    def __init__(self, input_path=None, video_path=None, loc=None, default_settings=None):

        self.path = input_path
        self.video_path = video_path
        self.loc = loc

        settings = {"Events": {"frames": "", "in_switch_dummy_groups": False, "in_textarea_filter": ""},
                    "Extension": {"in_switch_extend": False, "video_path": "", "loc": "", "in_numeric_extend_left": 0,
                                  "in_numeric_extend_right": 0, "in_numeric_enforce_min": None,
                                  "in_numeric_enforce_max": None, "in_switch_use_padding": False,
                                  "in_switch_use_footprint": True, },
                    "Normalization": {"in_switch_norm_default": False, "in_select_norm_mode": "min_max",
                                      "in_select_subtract_order": "", "in_select_subtract_mode": "min",
                                      "in_switch_subtract_pop": False, "in_select_subtract_rows": True,
                                      "in_select_divide_order": "", "in_select_divide_mode": "max_abs",
                                      "in_switch_divide_pop": False, "in_select_divide_rows": True,
                                      "in_select_impute_order": "", "in_switch_impute_fixed": False,
                                      "in_numeric_impute_val": 0, "in_select_gradient_order": ""},
                    "Encoding": {"in_select_encoding_options": "Feature Extraction",
                                 "FE": {"in_switch_encoder_use_fe": False,
                                        "in_switch_encoder_fe_add_def_columns": False, },
                                 "CNN": {"in_switch_cnn_use_cnn": False, "in_slider_cnn_split": 0.9,
                                         "in_numeric_cnn_learning_rate": 0.001, "in_numeric_cnn_latent_size": 128,
                                         "in_numeric_cnn_dropout": None, "in_numeric_cnn_regularize": None,
                                         "in_numeric_cnn_epochs": 10, "in_numeric_cnn_batch_size": 32,
                                         "in_numeric_cnn_patience": 5, "in_numeric_cnn_min_delta": 0.01, },
                                 "RNN": {"in_switch_encoding_use_rnn": False, "in_numeric_rnn_batch_size": 16,
                                         "in_numeric_rnn_val_size": 0.05, "in_numeric_rnn_test_size": 0.05,
                                         "in_numeric_rnn_num_epochs": 10, "in_numeric_rnn_patience": 5,
                                         "in_numeric_rnn_min_delta": 0.001, "in_switch_rnn_use_cuda": False,
                                         "in_numeric_rnn_encode_lr": 0.001, "in_numeric_rnn_decode_lr": 0.001,
                                         "in_numeric_rnn_diminish_lr": 0.99, "in_select_rnn_type": "LSTM",
                                         "in_numeric_rnn_hidden": 32, "in_numeric_rnn_num_layers": 2,
                                         "in_numeric_rnn_dropout": 0, "in_numeric_rnn_clip": 0.5,
                                         "in_switch_rnn_initialize_repeat": True, },
                                 "reducer": {"in_select_dimensionality_reduction_options": "",
                                             "in_switch_encoding_pca_auto_dim": True,
                                             "in_numeric_encoding_pca_n_components": 2,
                                             "in_numeric_encoding_tsne_n_components": 2,
                                             "in_numeric_encoding_tsne_early_exaggeration": 12,
                                             "in_numeric_encoding_tsne_perplexity": 30,
                                             "in_numeric_encoding_tsne_n_iter": 1000,
                                             "in_numeric_encoding_umap_neighbors": 30,
                                             "in_numeric_encoding_umap_min_dist": 0,
                                             "in_numeric_encoding_umap_n_components": 2, }},
                    "Experiment": {"in_textarea_inc_kwargs": "", }}

        if default_settings is not None:

            if not isinstance(default_settings, dict):
                raise ValueError(f"Please provide default_settings as dictionary, not {type(default_settings)}")

            settings = self.update_nested_dict(settings, default_settings)

        self.settings = settings

        self.app_ui = self.create_ui()
        self.app = App(self.app_ui, self.server)

    def update_nested_dict(self, base_dict, new_dict):
        """
        Update a nested dictionary with values from another nested dictionary.

        Parameters:
        - base_dict (dict): The dictionary to be updated.
        - new_dict (dict): The dictionary containing new values.

        Returns:
        - None: The base_dict is updated in-place.
        """
        for key, value in new_dict.items():
            if isinstance(value, dict):
                base_dict[key] = self.update_nested_dict(base_dict.get(key, {}), value)
            else:
                base_dict[key] = value
        return base_dict

    def create_ui(self):

        events_nav = ui.nav(
            "Events", ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.h3(""), ui.h3("Settings"), xui.tooltip(
                        ui.input_text("path", "Event directory", value=self.path),
                        "/absolute/path, comma-separated-list or /glob/path/*.roi", id="tooltip_path", placement="top"
                    ), ui.input_text("frames", "frames", value=self.settings["Events"]["frames"]), ui.input_switch(
                        "in_switch_dummy_groups", "dummy groups",
                        value=self.settings["Events"]["in_switch_dummy_groups"]
                    ), ui.h3("Information"), ui.output_text("out_txt_shape"), ui.output_text("out_txt_number_events"),
                    ui.br(), ui.h3("Filters"), ui.input_select("in_select_columns", label="", choices=[]),
                    xui.input_text_area(
                        "in_textarea_filter", "active filters:", self.settings["Events"]["in_textarea_filter"], rows=5,
                        autoresize=True
                    ), width=3
                ), ui.panel_main(
                    ui.panel_conditional(
                        "input.path !== ''", ui.output_data_frame("out_table_events"), ui.br(),
                        ui.output_plot("out_plot_event_map"), )
                )
            )
        )

        summary_nav = ui.nav(
            "Summary statistics", ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.row(
                        ui.column(3, ui.input_switch("in_switch_correlation", "correlation")),
                        ui.column(3, ui.input_switch("in_switch_logscale", "log scale")), ),
                    ui.input_select("in_select_hue", "category", choices=[]), ui.panel_conditional(
                        "input.in_switch_correlation != true", ui.input_select(
                            "in_select_type", "", choices=("boxplot", 'stripplot', "swarmplot", "violinplot")
                        ), ui.panel_conditional(
                            "input.in_select_type == 'boxplot'",
                            ui.column(3, ui.input_switch("in_switch_fliers", "show outliers", value=True)), ),
                        ui.panel_conditional(
                            "input.in_select_type == 'violinplot'",
                            ui.column(3, ui.input_switch("in_switch_cut", "cut")), ), ),  # ui.br(),
                    ui.h5("Statistics"), ui.output_data_frame("out_table_summary"), ), ui.panel_main(
                    ui.output_plot("out_plot_boxplot")
                )
            )
        )

        outliers = ui.nav(
            "Outliers", ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_switch("in_switch_calc_outliers", "calculate outliers"),
                    ui.h5("NaN settings"),
                    ui.input_select("in_select_nan_settings", "", choices=["column", "rows", "fill"]), ui.h5(""),
                    ui.h5("UMAP settings"), ui.row(
                        ui.column(6, ui.input_numeric("in_numeric_neighbors", "n_neighbors", value=30), ),
                        ui.column(6, ui.input_numeric("in_numeric_min_dist", "min_dist", value=0), ), ),
                    ui.input_select(
                        "in_text_metric", "metric",
                        choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis',
                                 'haversine', 'mahalanobis', 'wminkowski', 'seuclidean',
                                 'Angular and correlation metrics', 'cosine', 'correlation', ]
                    ), ui.h5(""), ui.h5("Plotting settings"), ui.row(
                        ui.column(4, ui.input_numeric("in_numeric_size", "size", value=8)),
                        ui.column(4, ui.input_numeric("in_numeric_alpha", "alpha", value=1, min=0.001, max=1)),
                        ui.column(4, ui.input_numeric("in_numeric_plot_max_cluster", "max traces", value=5)), ), ui.row(
                        ui.column(4, ui.input_switch("in_switch_separate_plots", "Separate")),
                        ui.column(4, ui.input_switch("in_switch_share_y", "share y")), ui.column(4, ), ),
                    ui.br(), ui.h5("Clusters"), ui.output_data_frame("out_data_frame_clusters"), width=3
                ), ui.panel_main(
                    ui.output_plot("out_plot_umap"), ui.output_plot("out_plot_traces", height="600px", width="600px"), )
            )
        )

        nav_extension = ui.nav(
            "Extension", ui.layout_sidebar(
                ui.panel_sidebar(
                    xui.card(
                        xui.card_header("Settings"), ui.input_switch(
                            "in_switch_extend", "use extension", value=self.settings["Extension"]["in_switch_extend"]
                        ), ui.input_text("video_path", "Video", value=self.settings["Extension"]["video_path"]),
                        ui.input_text("loc", "h5 location", value=self.settings["Extension"]["loc"]), ui.row(
                            ui.column(
                                6, ui.input_numeric(
                                    "in_numeric_extend_left", "fixed extension left",
                                    value=self.settings["Extension"]["in_numeric_extend_left"]
                                )
                            ), ui.column(
                                6, ui.input_numeric(
                                    "in_numeric_extend_right", "fixed extension right",
                                    value=self.settings["Extension"]["in_numeric_extend_right"]
                                )
                            ), ), ui.row(
                            ui.column(
                                4, ui.input_numeric(
                                    "in_numeric_enforce_min", "enforce min length",
                                    value=self.settings["Extension"]["in_numeric_enforce_min"]
                                )
                            ), ui.column(
                                4, ui.input_numeric(
                                    "in_numeric_enforce_max", "enforce max length",
                                    value=self.settings["Extension"]["in_numeric_enforce_max"]
                                )
                            ), ui.column(
                                4, ui.input_switch(
                                    "in_switch_use_padding", "use padding",
                                    value=self.settings["Extension"]["in_switch_use_padding"]
                                ), )
                        ), ui.input_switch(
                            "in_switch_use_footprint", "use footprint",
                            value=self.settings["Extension"]["in_switch_use_footprint"]
                        )
                    ), xui.card(
                        xui.card_header("Plotting"),
                        ui.input_switch("in_switch_ext_random", "random selection", value=True),
                        ui.input_switch("in_switch_ext_original", "show original", value=True), ui.panel_conditional(
                            "input.in_switch_ext_random",
                            ui.input_numeric("in_numeric_ext_num", "num_traces", value=3), ),
                        ui.input_text("in_text_ext_ids", "events ids"),
                        ui.input_numeric("in_numeric_panel_height_ext", "panel height", value=100),
                        ui.input_numeric("in_numeric_plot_columns_ext", "plot columns", value=1), ui.row(
                            ui.column(6, ui.input_switch("in_switch_ext_sharex", "share x")),
                            ui.column(6, ui.input_switch("in_switch_ext_sharey", "share y")), )
                    )
                ), ui.panel_main(
                    ui.output_ui("out_ext_dyn_plot")
                )
            )
        )

        nav_normalization = ui.nav(
            "Normalization", ui.layout_sidebar(
                ui.panel_sidebar(
                    xui.card(
                        xui.card_header("Normalization settings"), ui.input_switch(
                            "in_switch_norm_default", "default",
                            value=self.settings["Normalization"]["in_switch_norm_default"]
                        ), ui.panel_conditional(
                            "input.in_switch_norm_default", ui.input_select(
                                "in_select_norm_mode", "mode", choices=["", "min_max", "mean_std"],
                                selected=self.settings["Normalization"]["in_select_norm_mode"]
                            ), ), ui.panel_conditional(
                            "input.in_switch_norm_default == 0", xui.accordion(
                                xui.accordion_panel(
                                    "Subtract", [ui.input_select(
                                        "in_select_subtract_order", "order", choices=[""] + list(range(4)),
                                        selected=self.settings["Normalization"]["in_select_subtract_order"]
                                    ), ui.input_select(
                                        "in_select_subtract_mode", "mode",
                                        choices=["first", "mean", "min", "min_abs", "max", "max_abs", "std"],
                                        selected=self.settings["Normalization"]["in_select_subtract_mode"]
                                    ), ui.input_switch(
                                        "in_switch_subtract_pop", "population_wide",
                                        value=self.settings["Normalization"]["in_switch_subtract_pop"]
                                    ), ui.input_switch(
                                        "in_select_subtract_rows", "rows",
                                        value=self.settings["Normalization"]["in_select_subtract_rows"]
                                    )]
                                ), xui.accordion_panel(
                                    "Divide", [ui.input_select(
                                        "in_select_divide_order", "order", choices=[""] + list(range(4)),
                                        selected=self.settings["Normalization"]["in_select_divide_order"]
                                    ), ui.input_select(
                                        "in_select_divide_mode", "mode",
                                        choices=["first", "mean", "min", "min_abs", "max", "max_abs", "std"],
                                        selected=self.settings["Normalization"]["in_select_divide_mode"]
                                    ), ui.input_switch(
                                        "in_switch_divide_pop", "population_wide",
                                        value=self.settings["Normalization"]["in_switch_divide_pop"]
                                    ), ui.input_switch(
                                        "in_select_divide_rows", "rows",
                                        value=self.settings["Normalization"]["in_select_divide_rows"]
                                    )]
                                ), xui.accordion_panel(
                                    "Impute NaN", [ui.input_select(
                                        "in_select_impute_order", "order", choices=[""] + list(range(4)),
                                        selected=self.settings["Normalization"]["in_select_impute_order"]
                                    ), ui.row(
                                        ui.column(
                                            6, ui.input_switch(
                                                "in_switch_impute_fixed", "fixed value",
                                                value=self.settings["Normalization"]["in_switch_impute_fixed"]
                                            )
                                        ), ui.column(
                                            6, ui.panel_conditional(
                                                "input.in_switch_impute_fixed", ui.input_numeric(
                                                    "in_numeric_impute_val", "value",
                                                    value=self.settings["Normalization"]["in_numeric_impute_val"]
                                                )
                                            )
                                        )
                                    ), ]
                                ), xui.accordion_panel(
                                    "Gradient", ui.input_select(
                                        "in_select_gradient_order", "order", choices=[""] + list(range(4)),
                                        selected=self.settings["Normalization"]["in_select_gradient_order"]
                                    ), ), open=False
                            )
                        ), ), xui.card(
                        xui.card_header("Plotting"),
                        ui.input_switch("in_switch_norm_random", "random selection", value=True),
                        ui.input_switch("in_switch_norm_original", "show original", value=True), ui.panel_conditional(
                            "input.in_switch_norm_random",
                            ui.input_numeric("in_numeric_norm_num", "num_traces", value=3), ),
                        ui.input_text("in_text_norm_ids", "events ids"),
                        ui.input_numeric("in_numeric_panel_height", "panel height", value=100),
                        ui.input_numeric("in_numeric_plot_columns", "plot columns", value=1), ui.row(
                            ui.column(6, ui.input_switch("in_switch_norm_sharex", "share x")),
                            ui.column(6, ui.input_switch("in_switch_norm_sharey", "share y")), )
                    )
                ), ui.panel_main(
                    ui.output_ui("out_dyn_plot")
                )
            )
        )

        nav_encoding = ui.nav(
            "Encoding", ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_select(
                        "in_select_encoding_options", "Encoders",
                        choices=["", "Feature Extraction", "Convolutional Neural Network (CNN)",
                                 "Recurrent Neural Network (RNN)"],
                        selected=self.settings["Encoding"]["in_select_encoding_options"]
                    ), ui.panel_conditional(
                        "input.in_select_encoding_options == 'Feature Extraction'", xui.card(
                            xui.card_header("FE"), ui.input_switch(
                                "in_switch_encoder_use_fe", "use FE encoding",
                                value=self.settings["Encoding"]["FE"]["in_switch_encoder_use_fe"]
                            ), ui.input_switch(
                                "in_switch_encoder_fe_add_def_columns", "add default columns",
                                value=self.settings["Encoding"]["FE"]["in_switch_encoder_fe_add_def_columns"]
                            ), ui.input_select(
                                "in_select_encoder_fe_nan_settings", "NaN behavior",
                                choices=["", "column", "rows", "fill"]
                            ), ui.panel_conditional(
                                "input.in_select_encoder_fe_nan_settings == 'fill'", ui.input_numeric(
                                    "in_numeric_encoder_fe_nan_fixed", "fixed value", value=-10 ** 6
                                ), ), ), ), ui.panel_conditional(
                        "input.in_select_encoding_options == 'Convolutional Neural Network (CNN)'", xui.card(
                            xui.card_header("CNN"), ui.input_switch(
                                "in_switch_cnn_use_cnn", "use cnn encoding",
                                value=self.settings["Encoding"]["CNN"]["in_switch_cnn_use_cnn"]
                            ), ui.output_text("out_text_cnn_warning_ragged"), ui.br(), xui.card(
                                xui.card_header("Settings"), ui.input_file("in_file_cnn_encoder", "encoder path"),
                                ui.row(
                                    ui.column(
                                        4, ui.input_slider(
                                            "in_slider_cnn_split", "Train split", min=0.01, max=0.99,
                                            value=self.settings["Encoding"]["CNN"]["in_slider_cnn_split"]
                                        ), ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_cnn_latent_size", "latent_size",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_latent_size"]
                                        ), ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_cnn_learning_rate", "learning rate",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_learning_rate"]
                                        ), ), ), ui.row(
                                    ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_dropout", "dropout",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_dropout"], max=0.99
                                        )
                                    ), ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_regularize", "regularize latent",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_regularize"],
                                            max=0.99
                                        )
                                    ), ), ui.row(
                                    ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_epochs", "epochs",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_epochs"], min=1
                                        ), ), ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_batch_size", "batch size",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_batch_size"], min=1
                                        ), ), ),  # ui.h6("Early stopping"),
                                ui.row(
                                    ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_patience", "patience",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_patience"], min=1
                                        ), ), ui.column(
                                        6, ui.input_numeric(
                                            "in_numeric_cnn_min_delta", "min_delta",
                                            value=self.settings["Encoding"]["CNN"]["in_numeric_cnn_min_delta"]
                                        ), ), ), ), xui.card(
                                xui.card_header("Plotting"), ui.input_numeric(
                                    "in_numeric_cnn_num_samples", "num_samples", value=9
                                ), ui.input_switch("in_switch_cnn_show_diff", "show latent diff")
                            )
                        ), ), ui.panel_conditional(
                        "input.in_select_encoding_options == 'Recurrent Neural Network (RNN)'", xui.card(
                            xui.card_header("RNN"), ui.input_switch(
                                "in_switch_encoding_use_rnn", "use rnn encoding",
                                value=self.settings["Encoding"]["RNN"]["in_switch_encoding_use_rnn"]
                            ), xui.card(
                                xui.card_header("Settings"), ui.input_switch(
                                    "in_switch_rnn_use_cuda", "use cuda",
                                    value=self.settings["Encoding"]["RNN"]["in_switch_rnn_use_cuda"]
                                ), ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_batch_size", "batch size",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_batch_size"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_val_size", "validation split",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_val_size"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_test_size", "test split",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_test_size"]
                                        )
                                    ), ), ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_num_epochs", "num epochs",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_num_epochs"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_patience", "patience",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_patience"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_min_delta", "min delta",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_min_delta"]
                                        )
                                    ), ), ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_encode_lr", "encoder learning rate",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_encode_lr"], max=1
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_decode_lr", "decoder learning rate",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_decode_lr"], max=1
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_diminish_lr", "learning rate decay",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_diminish_lr"], max=1
                                        )
                                    ), ), ui.row(
                                    ui.column(
                                        4, ui.input_select(
                                            "in_select_rnn_type", "RNN type", choices=["LSTM"],
                                            selected=self.settings["Encoding"]["RNN"]["in_select_rnn_type"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_hidden", "hidden dimension",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_hidden"]
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_num_layers", "num layers",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_num_layers"]
                                        )
                                    ), ), ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_dropout", "dropout",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_dropout"]
                                        ), selected=self.settings["Encoding"]["RNN"]["in_select_rnn_type"]
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_clip", "gradient clipping",
                                            value=self.settings["Encoding"]["RNN"]["in_numeric_rnn_clip"]
                                        )
                                    ), ui.column(
                                        4, ui.input_switch(
                                            "in_switch_rnn_initialize_repeat", "initialize repetition",
                                            value=self.settings["Encoding"]["RNN"]["in_switch_rnn_initialize_repeat"]
                                        )
                                    ), ), ), xui.card(
                                xui.card_header("Plotting"), ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_num_samples", "num samples", value=4
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_plot_height", "plot height", value=1200
                                        )
                                    ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_rnn_plot_width", "plot width", value=600
                                        )
                                    ), ), ), ), ), ui.panel_conditional(
                        "input.in_select_encoding_options != ''", xui.card(
                            xui.card_header("Dimensionality reduction"), ui.input_select(
                                "in_select_dimensionality_reduction_options", "Options",
                                choices=["", "PCA", "tSNE", "UMAP"], selected=self.settings["Encoding"]["reducer"][
                                    "in_select_dimensionality_reduction_options"]
                            ), ui.panel_conditional(
                                "input.in_select_dimensionality_reduction_options == 'PCA'", ui.row(
                                    ui.column(
                                        3, ui.input_switch(
                                            "in_switch_encoding_pca_auto_dim", "automatic choice"
                                        ), value=self.settings["Encoding"]["reducer"]["in_switch_encoding_pca_auto_dim"]
                                    ), ui.column(
                                        9, ui.input_numeric(
                                            "in_numeric_encoding_pca_n_components", "n_components",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_pca_n_components"]
                                        )
                                    ), ), ui.output_text("out_text_reduction_pca_quality"), ), ui.panel_conditional(
                                "input.in_select_dimensionality_reduction_options == 'tSNE'", ui.row(
                                    ui.column(
                                        3, ui.input_numeric(
                                            "in_numeric_encoding_tsne_n_components", "n_components",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_tsne_n_components"]
                                        )
                                    ), ui.column(
                                        3, ui.input_numeric(
                                            "in_numeric_encoding_tsne_early_exaggeration", "early_exaggeration",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_tsne_early_exaggeration"]
                                        )
                                    ), ui.column(
                                        3, ui.input_numeric(
                                            "in_numeric_encoding_tsne_perplexity", "perplexity",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_tsne_perplexity"]
                                        )
                                    ), ui.column(
                                        3, ui.input_numeric(
                                            "in_numeric_encoding_tsne_n_iter", "n_iter",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_tsne_n_iter"], min=250
                                        )
                                    ), ), ), ui.panel_conditional(
                                "input.in_select_dimensionality_reduction_options == 'UMAP'", ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_encoding_umap_neighbors", "n_neighbors",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_umap_neighbors"]
                                        ), ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_encoding_umap_min_dist", "min_dist",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_umap_min_dist"]
                                        ), ), ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_encoding_umap_n_components", "n_components",
                                            value=self.settings["Encoding"]["reducer"][
                                                "in_numeric_encoding_umap_n_components"]
                                        ), ), ), ui.input_select(
                                    "in_text_encoding_umap_metric", "metric",
                                    choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra',
                                             'braycurtis', 'haversine', 'mahalanobis', 'wminkowski', 'seuclidean',
                                             'Angular and correlation metrics', 'cosine', 'correlation', ],
                                    selected='euclidean'
                                ), ), )
                    ), ), ui.panel_main(
                    ui.panel_conditional(
                        "input.in_select_encoding_options != ''", xui.card(
                            xui.card_header("Encoders"), ui.panel_conditional(
                                "input.in_switch_cnn_use_cnn & input.in_select_encoding_options == 'Convolutional Neural Network (CNN)'",
                                ui.output_plot("out_plot_cnn_history"), ui.output_plot("out_plot_cnn_examples"), ),
                            ui.panel_conditional(
                                "input.in_switch_encoding_use_rnn & input.in_select_encoding_options == 'Recurrent Neural Network (RNN)'",
                                ui.output_plot("out_plot_rnn_history"), ui.output_ui("out_plot_rnn_ui"), ), ), xui.card(
                            xui.card_header("Encoding"), ui.output_data_frame("out_data_frame_encoding_"), ),

                        ui.panel_conditional(
                            "input.in_select_dimensionality_reduction_options != ''", xui.card(
                                xui.card_header("Reduction"), ui.row(
                                    ui.column(3, ui.output_data_frame("out_data_frame_reduction_")), ui.column(
                                        9, ui.row(
                                            ui.column(
                                                6, ui.input_select(
                                                    "out_select_reduction_dim1", "X", choices=["dim1"]
                                                )
                                            ), ui.column(
                                                6, ui.input_select("out_select_reduction_dim2", "Y", choices=["dim2"])
                                            ), ), ui.output_plot("out_plot_reduction_scatter")
                                    ), )
                            ), ), ),

                )
            )
        )

        nav_experiment = ui.nav(
            "Experiment", ui.navset_tab_card(
                ui.nav(
                    "Conditional Contrasts", ui.layout_sidebar(
                        ui.panel_sidebar(
                            xui.card(
                                xui.card_header("Contrast groups"),
                                ui.input_select("in_select_contrast_groups", "category", choices=[]),
                                ui.panel_conditional(
                                    "input.in_select_contrast_groups == 'dummy'",
                                    ui.input_numeric("in_numeric_contrast_num_groups", "num contrasts", value=2),
                                    ui.input_select("in_select_contrast_bias", "bias column", choices=[]), ), ),
                            xui.card(
                                xui.card_header("Settings"),
                                ui.input_switch("in_switch_contrast_use_classifier", "use classifier", value=True),
                                ui.panel_conditional(
                                    "input.in_switch_contrast_use_classifier == true", ui.h5("Classifiers"), ui.row(
                                        ui.column(
                                            4, ui.input_numeric(
                                                "in_numeric_contrast_train_split", "train split", value=0.9, min=0.01,
                                                max=0.99
                                            )
                                        ), ui.column(
                                            4, ui.input_switch(
                                                "in_switch_contrast_balance_training", "balance training set",
                                                value=True
                                            )
                                        ), ui.column(
                                            4, ui.input_switch("in_switch_contrast_balance_test", "balance test set")
                                        ), ), ui.input_select(
                                        "in_select_contrast_classifier", "Classifier",
                                        choices=[m for m in Discriminator.get_available_models() if "Classifier" in m],
                                        selected="RandomForestClassifier"
                                    ), ui.input_select(
                                        "in_select_contrast_norm_confusion_matrix", "normalize confusion matrix",
                                        choices=["", "pred", "all", "true"]
                                    ), ), ui.panel_conditional(
                                    "input.in_switch_contrast_use_classifier == false",
                                    ui.h5("Hierarchical Clustering"), ui.row(
                                        ui.column(
                                            6, ui.input_switch("in_switch_contrast_use_embedding", "use_embedding"), ),
                                        ui.column(
                                            6, ui.column(
                                                4, ui.input_switch("in_select_contrasts_cl_transpose", "transpose")
                                            ), ), ), ui.row(
                                        ui.column(
                                            6, ui.input_numeric(
                                                "in_numeric_contrasts_bary_cutoff", "cutoff", value=5
                                            )
                                        ), ui.column(
                                            6, ui.input_select(
                                                "in_select_contrasts_bary_crit", "criterion",
                                                choices=['distance', 'maxclust'], selected="maxclust"
                                            )
                                        ), ), ui.row(
                                        ui.column(
                                            6, ui.input_select(
                                                "in_select_contrasts_cl_method", "method",
                                                choices=['average', 'single', 'complete', 'weighted', 'centroid',
                                                         'median', 'ward']
                                            )
                                        ), ui.column(
                                            6, ui.input_select(
                                                "in_select_contrasts_cl_metric", "metric",
                                                choices=['euclidean', 'minkowski', 'cityblock', 'seuclidean',
                                                         'sqeuclidean', 'cosine', 'correlation', 'hamming', 'jaccard',
                                                         'jensenshannon', 'chebyshev', 'canberra', 'braycurtis',
                                                         'mahalanobis', 'yule', 'matching', 'dice', 'kulczynski1',
                                                         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
                                            )
                                        ), ), ui.row(
                                        ui.column(
                                            6, ui.input_select(
                                                "in_select_contrasts_cl_z_score", "z_score",
                                                choices=["", "rows", "columns"]
                                            )
                                        ), ui.column(
                                            6, ui.input_select(
                                                "in_select_contrasts_cl_center", "center",
                                                choices=["", "rows", "columns"]
                                            )
                                        ), ), ui.row(
                                        ui.column(
                                            4, ui.input_select(
                                                "in_select_contrasts_cl_cmap", "colormap", choices=["vlag"]
                                            )
                                        ), ui.column(
                                            4, ui.input_select(
                                                "in_select_contrasts_cl_cmap_group", "group colormap",
                                                choices=["viridis"]
                                            )
                                        ), ui.column(
                                            4, ui.input_text("in_text_contrasts_cl_group_color_col",
                                                             "group color column")
                                        )
                                    ), )
                            )
                        ), ui.panel_main(
                            ui.panel_conditional(
                                "input.in_switch_contrast_use_classifier == true",
                                ui.output_plot("out_plot_contrast_classifier")
                            ), ui.panel_conditional(
                                "input.in_switch_contrast_use_classifier == false", ui.row(
                                    ui.column(6, ui.output_plot("out_plot_contrast_hierarchical_left", height="800px")),
                                    ui.column(
                                        6, ui.output_plot("out_plot_contrast_hierarchical_right"), height="800px"
                                    ), ),

                            ), )
                    )
                ), ui.nav(
                    "Coincidence Detection", ui.layout_sidebar(
                        ui.panel_sidebar(
                            xui.card(
                                xui.card_header("Incidences"),
                                ui.input_switch("in_switch_inc_create_dummy", "create dummy incidences"),
                                ui.panel_conditional(
                                    "input.in_switch_inc_create_dummy != true",
                                    ui.input_file("in_file_inc_file", label="incidence file path"), ),
                                ui.panel_conditional(
                                    "input.in_switch_inc_create_dummy", ui.row(
                                        ui.column(
                                            6, ui.input_numeric(
                                                "in_numeric_inc_dummy_distance", "average distance", value=25
                                            )
                                        ), ui.column(
                                            6, ui.input_numeric(
                                                "in_numeric_inc_dummy_gap", "minimum gap", value=10
                                            )
                                        ),  # ui.column(1, ui.input_numeric("in_numeric_inc_dummy_", "", value=0)),
                                    )
                                ), ui.output_text("out_text_inc_num_incidence"), ), xui.card(
                                xui.card_header("Settings"),
                                ui.input_switch("in_switch_inc_predict_coincidence", "predict class", value=True),
                                ui.row(
                                    ui.column(
                                        4, ui.input_numeric(
                                            "in_numeric_inc_train_split", "train split", value=0.9, min=0.01, max=0.99
                                        )
                                    ), ui.column(
                                        4, ui.input_switch(
                                            "in_switch_inc_balance_training", "balance training set", value=True
                                        )
                                    ), ui.column(
                                        4, ui.input_switch("in_switch_inc_balance_test", "balance test set")
                                    ), ), ui.panel_conditional(
                                    "input.in_switch_inc_predict_coincidence", ui.h5("Coincidence Classification"),
                                    ui.input_select(
                                        "in_select_inc_classifier", "Classifier",
                                        choices=[m for m in Discriminator.get_available_models() if "Classifier" in m],
                                        selected="RandomForestClassifier"
                                    ), ui.row(
                                        ui.column(
                                            6, ui.input_switch(
                                                "in_switch_inc_binary_class", "binary classification", value=True
                                            )
                                        ), ui.column(
                                            6, ui.input_select(
                                                "in_select_inc_norm_confusion_matrix", "normalize confusion matrix",
                                                choices=["", "pred", "all", "true"], selected=""
                                            )
                                        ), )
                                ), ui.panel_conditional(
                                    "input.in_switch_inc_predict_coincidence == false", ui.h5("Coincidence Regression"),
                                    ui.input_select(
                                        "in_select_inc_classifier_loc", "Classifier",
                                        choices=[m for m in Discriminator.get_available_models() if "Regressor" in m],
                                        selected="RandomForestRegressor"
                                    ), ), ui.input_text_area(
                                    "in_textarea_inc_kwargs", "additional model parameters (yaml format)",
                                    value=self.settings["Experiment"]["in_textarea_inc_kwargs"], height="200px"
                                ), )
                        ), ui.panel_main(
                            ui.output_plot("out_plot_inc")
                        )
                    )
                ), )
        )

        # NAME_nav = ui.nav(
        #             "Title",
        #             ui.layout_sidebar(
        #                 ui.panel_sidebar(),
        #                 ui.panel_main()
        #             )
        # )

        return ui.page_fluid(
            ui.panel_title("Analysis"), ui.navset_tab(
                events_nav, summary_nav, outliers, nav_extension, nav_normalization, nav_encoding, nav_experiment
            )
        )

    def server(self, input, output, session):

        ###############
        # Event Objects

        @reactive.Calc
        def get_events_obj():

            path = input.path()

            split_path = path.split(",")
            if len(split_path) > 1:
                path = path.replace(" ", "").split(",")

                for p in path:
                    if not Path(p).exists():
                        ui.notification_show(f"file missing: {p}", type="error")
                        return None

            elif "*" in path:

                path = Path(path)
                parent = path.parent
                name = path.name
                path = list(parent.glob(name))

            else:
                path = Path(path)
                if not path.exists():
                    ui.notification_show(f"file missing: {path}", type="error")
                    return None

            print(path)

            events = Events(path)

            if input.in_switch_dummy_groups():
                lut_group = {idx: np.random.randint(1, 3) for idx in events.events.index.tolist()}
                events.add_clustering(lut_group, column_name="group")

                lut_subject_id = {idx: np.random.randint(1, 4) for idx in events.events.index.tolist()}
                events.add_clustering(lut_subject_id, column_name="subject_id")

                events.events.group = events.events.group.astype("category")
                events.events.subject_id = events.events.subject_id.astype("category")

            return events

        @reactive.Calc
        def get_events_obj_filtered():

            events = get_events_obj().copy()
            df = events.events
            filter_string = input.in_textarea_filter()

            if events is not None and len(df) > 0 and filter_string is not None and filter_string != "":

                filters = {}
                for i, filter_ in enumerate(filter_string.split("\n")):

                    try:

                        key, value = filter_.split(":")

                        # define dtype
                        typ = df[key].dtype
                        if typ == "object":
                            typ = type(df[key].dropna().iloc[0])

                        if "(" in value:

                            value = value.replace("(", "").replace(")", "")
                            min_, max_ = value.split(",")
                            value = (float(min_), float(max_))

                        elif "[" in value:

                            value = value.replace("[", "").replace("]", "").replace("'", "")
                            value = list(value.split(","))

                            if typ in [str]:
                                value = [str(v) for v in value]

                            elif typ in [int, np.int64]:
                                value = [int(v) for v in value]

                            elif typ in [float, np.float64]:
                                value = [float(v) for v in value]

                            else:
                                ui.notification_show(f"unknown datatype ({i}) type: {typ}", type="error")

                        else:
                            ui.notification_show(f"unknown filter ({i}) type: {type(value)}", type="error")

                        filters[key] = value

                    except Exception as e:
                        ui.notification_show(f"exception ({i}): {e}")

                events.filter(filters=filters, inplace=True)

            return events

        @reactive.Calc
        def get_events_obj_extended():

            events = get_events_obj_filtered().copy()
            use_extend = input.in_switch_extend()
            use_footprint = input.in_switch_use_footprint()
            video = input.video_path()
            loc = input.loc()

            fix_ext_left, fix_ext_right = input.in_numeric_extend_left(), input.in_numeric_extend_right()
            extend = (fix_ext_left, fix_ext_right)
            enforce_min, enforce_max = input.in_numeric_enforce_min(), input.in_numeric_enforce_max()
            use_padding = input.in_switch_use_padding()

            if events is not None and len(events) > 0 and use_extend and video != "" and (
                    fix_ext_left != 0 or fix_ext_right != 0 or enforce_min is not None or enforce_max is not None):

                video = Video(data=video, loc=loc, lazy=True)
                events.get_extended_events(
                    video=video, in_place=True, use_footprint=use_footprint, extend=extend, ensure_min=enforce_min,
                    ensure_max=enforce_max, pad_borders=use_padding
                )

                if not use_padding:

                    df = events.events

                    if enforce_min is not None:
                        df = df[df.dz >= enforce_min]

                    if enforce_max is not None:
                        df = df[df.dz <= enforce_max]

                    if (enforce_max is not None or enforce_min is not None) and len(events.events) != len(df):
                        logging.warning("Events with incorrect size were removed.")
                        ui.notification_show("Events with incorrect size were removed.")

                    events.events = df

            elif use_extend:
                ui.notification_show("cannot extend videos", type="error")

            return events

        @reactive.Calc
        def get_events_obj_normalized():

            events = get_events_obj_extended().copy()

            if events is not None and len(events) > 0:

                instructions = {}

                if input.in_switch_norm_default():

                    mode = input.in_select_norm_mode()
                    if mode is not None:

                        try:
                            instructions["default"] = mode

                        except Exception as err:
                            ui.notification_show(f"No valid mode: {mode}", type="error")
                            logging.warning(f"Error: {err}")

                else:

                    order = [("subtract", input.in_select_subtract_order()), ("divide", input.in_select_divide_order()),
                             ("impute_nan", input.in_select_impute_order()), ("diff", input.in_select_gradient_order())]

                    # Filter out tuples where the second item is an empty string
                    filtered_temp = [t for t in order if t[1] != ""]

                    # Check for duplicates in the second items
                    second_items = [t[1] for t in filtered_temp]
                    if len(second_items) != len(set(second_items)):
                        raise ValueError("Error: Duplicate numbers found.")

                    # Sort the list by the second item in each tuple
                    sorted_temp = sorted(filtered_temp, key=lambda x: x[1])

                    # Extract the first items from the sorted list
                    sorted_keys = [t[0] for t in sorted_temp]

                    instructions = {}
                    for i, key in enumerate(sorted_keys):

                        if key == "subtract":
                            instructions[i] = [key, {"mode": input.in_select_subtract_mode(),
                                                     "population_wide": input.in_switch_subtract_pop(),
                                                     "rows": input.in_select_subtract_rows()}]

                        elif key == "divide":
                            instructions[i] = [key, {"mode": input.in_select_divide_mode(),
                                                     "population_wide": input.in_switch_divide_pop(),
                                                     "rows": input.in_select_divide_rows()}]

                        elif key == "impute_nan":
                            instructions[i] = [key, {
                                "fixed_value": input.in_numeric_impute_val() if input.in_switch_impute_fixed() else None}]

                        elif key == "diff":
                            instructions[i] = [key, {}]

                events.normalize(instructions, inplace=True)

            return events

        @reactive.Calc
        def get_summary_table():
            events = get_events_obj_filtered().events

            if events is not None:
                events = self.get_table_excl(
                    events, excl_columns=(
                        'contours', 'trace', 'noise_mask_trace', 'mask', 'footprint', 'group', 'file_name', 'z0', 'z1',
                        'x0', 'x1', 'y0', 'y1', 'subject_id')
                )

                mean = events.mean(axis=0, skipna=True)
                std = events.std(axis=0, skipna=True)
                summary = pd.DataFrame({"mean": mean, "std": std})

                return summary
            return None

        @reactive.Calc
        def get_event_map():

            try:

                path = Path(input.path()).joinpath("event_map.tdb")

                if path.exists():
                    io = IO()
                    data = io.load(path, lazy=True)
                    return data

                else:
                    raise FileNotFoundError(f"Path doesn't exist: {path}")

            except Exception as e:
                ui.notification_show(f"Error: {e}: {traceback.print_exc()}", type="warning")
                return None

        ###################
        # Event information

        @reactive.Calc
        def get_data_dimension():

            event_map = get_event_map()

            if event_map is not None:
                return event_map.shape
            else:
                return None

        @reactive.Calc
        def get_frames():
            frame_str = input.frames()
            frames = []

            for f in frame_str.split(","):
                if len(f) > 0:
                    frames.append(int(f))

            return frames

        ########
        # EVENTS

        @output
        @render.data_frame()
        def out_table_events():
            events = get_events_obj_filtered().events

            if events is not None:

                events = events.reset_index()

                # events = self.get_table_excl(events)
                events = self.get_table_rounded(events)

                events = render.DataTable(events, row_selection_mode="single")

                return events

            else:
                return None

        @output
        @render.text
        def out_txt_shape():
            return f"shape: {get_data_dimension()}"

        @output
        @render.text
        def out_txt_number_events():

            events = get_events_obj().events
            events_filtered = get_events_obj_filtered().events

            item = None
            if events is not None and events_filtered is not None:
                N = len(events)
                Nf = len(events_filtered)

                if N != Nf:
                    item = f"#events: {Nf}/{N} ({Nf / N * 100:.1f}%)"
                else:
                    item = f"#events: {N}"

            return item

        @reactive.Effect
        def update_columns():
            events = get_events_obj().events
            if events is not None and len(events) > 0:

                columns = events.columns.tolist()
                for i, col in enumerate(columns):
                    if isinstance(events[col].head(1).values[0], np.ndarray):
                        del columns[i]

                ui.update_select("in_select_columns", choices=[""] + columns)

        @reactive.Effect
        def update_filter_template():
            column = input.in_select_columns()
            events = get_events_obj().events

            if column != "" and column is not None and events is not None and len(events) > 0:

                col = events[column]
                col = col.dropna()
                v0 = col.iloc[0]

                if col.dtype == "category" or isinstance(v0, str):
                    options = col.unique().tolist()
                    filter_string = f"{column}:{options}"

                elif isinstance(v0, (int, float, np.int64)):

                    min_, max_ = np.round(col.min(), 2), np.round(col.max(), 2)
                    filter_string = f"{column}:({min_}, {max_})"

                else:
                    filter_string = f"{column}: {type(v0)}"

                prev_filters = input.in_textarea_filter()
                if prev_filters.startswith("\n"):
                    prev_filters = prev_filters[1:]

                if prev_filters == "":
                    new_filter = filter_string
                else:
                    new_filter = f"{prev_filters}\n{filter_string}"

                ui.update_select("in_select_columns", selected="")
                ui.update_text_area("in_textarea_filter", value=new_filter)
                ui.update_text_area("")

        #########
        # SUMMARY
        @output
        @render.data_frame
        def out_table_summary():

            summary = get_summary_table()

            if summary is not None:
                summary = self.get_table_rounded(summary)
                summary = summary.reset_index()

                summary = render.DataTable(summary, height=None, summary=False, row_selection_mode="multiple")

                return summary
            return None

        @reactive.Effect
        def update_in_select_hue():

            events = get_events_obj_filtered().events

            if events is not None:
                cols = [col for col in events.columns if
                        events[col].dtype == "category" and len(np.unique(events[col].dropna())) > 1]

                ui.update_select("in_select_hue", choices=[""] + cols)

        @output
        @render.plot()
        def out_plot_boxplot():

            events = get_events_obj_filtered().copy().events
            summary = get_summary_table()

            indices = list(req(input.out_table_summary_selected_rows()))
            col = [summary.iloc[ind].name for ind in indices]

            if events is not None and col is not None:

                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)

                    # select hue
                    hue = input.in_select_hue() if input.in_select_hue() != "" else None

                    # plot
                    if input.in_switch_correlation() and len(col) > 1:

                        if len(col) == 2:

                            events[col[0]] = events[col[0]].astype(float)
                            events[col[1]] = events[col[1]].astype(float)

                            fig = seaborn.jointplot(
                                data=events, x=col[0], y=col[1], hue=hue, kind="reg" if hue is None else 'scatter'
                            )

                        else:

                            g = sns.PairGrid(
                                events, x_vars=col, y_vars=col, hue=hue, diag_sharey=False
                            )

                            g.map_lower(sns.scatterplot, s=15)
                            g.map_diag(sns.kdeplot, lw=2)
                            g.map_upper(sns.kdeplot, fill=True)

                            fig = g

                    else:

                        id_vars = ["index", hue] if hue is not None else "index"

                        events = events.reset_index()
                        events = events.melt(id_vars=id_vars, value_vars=col)
                        events.value = events.value.astype(float)

                        typ = input.in_select_type()
                        if typ == "boxplot":
                            fig = seaborn.boxplot(
                                data=events, x="variable", y="value", hue=hue, showfliers=input.in_switch_fliers()
                            )

                        elif typ == "stripplot":
                            fig = seaborn.stripplot(data=events, x="variable", y="value", hue=hue)

                        elif typ == "swarmplot":
                            fig = seaborn.swarmplot(data=events, x="variable", y="value", hue=hue)

                        elif typ == "violinplot":
                            fig = seaborn.violinplot(
                                data=events, x="variable", y="value", hue=hue,
                                split=True if len(id_vars) == 2 and len(np.unique(events[hue])) == 2 else False,
                                cut=0 if input.in_switch_cut() else 2
                            )

                    if input.in_switch_logscale():
                        plt.yscale('log')

                    return fig

            else:
                fig, ax = plt.subplots(figsize=(10, 10))
                return fig

        @output
        @render.plot
        def out_plot_event_map():

            data = get_event_map()
            frames = get_frames()

            if len(frames) > 0:
                return self.plot_images([data], frames, lbls=["event map"])
            else:
                return None

        ##########
        # OUTLIERS

        @reactive.Calc
        def get_table_no_nan():

            events_obj = get_events_obj_filtered().copy()

            if events_obj is not None:

                events = events_obj.events
                events = events[[col for col in events.columns if col.startswith("v")]]

                if events.isnull().values.any():
                    ui.notification_show("None values present. Trying to fix automatically.", type="error")

                    ui.input_select("in_select_nan_settings", "", choices=["column", "rows", "fill"]),

                    if input.in_select_nan_settings() == "fill":
                        max_negative_float32 = -np.finfo(np.float32).max
                        events = events.fillna(max_negative_float32)

                    elif input.in_select_nan_settings() == "column":
                        events = events.dropna(axis=1, how='any')

                    elif input.in_select_nan_settings() == "rows":
                        events = events.dropna(axis=0, how='any')

                events_obj.events = events

            return events_obj

        @reactive.Calc
        def get_embedding():

            events = get_table_no_nan()

            if events is not None and input.in_switch_calc_outliers():
                events = events.events

                # embed
                reducer = umap.UMAP(n_neighbors=input.in_numeric_neighbors(), min_dist=input.in_numeric_min_dist(),
                                    metric=input.in_text_metric())

                embedding = reducer.fit_transform(events.values)

                # find outliers
                hdb = hdbscan.HDBSCAN(
                    min_samples=1, min_cluster_size=2, allow_single_cluster=True, core_dist_n_jobs=-1,
                    prediction_data=True
                )

                hdb_labels = hdb.fit_predict(embedding)

                return embedding, hdb_labels
            return None, None

        @output
        @render.plot
        async def out_plot_umap():

            embedding, hdb_labels = get_embedding()

            if embedding is not None:
                fig, ax = plt.subplots()

                palette = sns.color_palette("husl", len(np.unique(hdb_labels)))

                ax.scatter(
                    embedding[:, 0], embedding[:, 1], alpha=input.in_numeric_alpha(), s=input.in_numeric_size(),
                    color=[palette[v] for v in hdb_labels]
                )

                return fig
            return None

        @output
        @render.data_frame
        def out_data_frame_clusters():

            embedding, hdb_labels = get_embedding()

            if embedding is not None:
                df = pd.DataFrame({"labels": hdb_labels})

                counts = df.value_counts().to_frame().reset_index()

                return render.DataTable(counts, height=None, summary=False, row_selection_mode="single")
            return None

        @output
        @render.plot
        def out_plot_traces():

            embedding, hdb_labels = get_embedding()
            df = pd.DataFrame({"labels": hdb_labels})
            counts = df.value_counts().to_frame().reset_index()

            idx_selected = list(req(input.out_data_frame_clusters_selected_rows()))[0]
            lbl_selected = counts.iloc[idx_selected].labels

            hdb_labels = np.array(hdb_labels)
            indices = np.where(hdb_labels == int(lbl_selected))[0]

            events = get_events_obj_filtered().events
            events = events.iloc[indices]

            if len(events) > input.in_numeric_plot_max_cluster():
                events = events.sample(input.in_numeric_plot_max_cluster())

            if input.in_switch_separate_plots():
                fig, axx = plt.subplots(len(events), figsize=(20, 5 * len(events)), sharey=input.in_switch_share_y())
            else:
                fig, axx = plt.subplots(figsize=(20, 5))
                axx = [axx for _ in range(len(events))]

            for i, (idx, row) in enumerate(events.iterrows()):

                if input.in_switch_separate_plots():
                    axx[i].plot(range(row.z0.astype(int), row.z1.astype(int)), row.trace)
                    axx[i].set_ylabel(f"idx: {idx}")
                else:
                    axx[i].plot(range(row.dz.astype(int)), row.trace)
                    axx[i].set_ylabel(f"idx: {events.index.tolist()}")

        ###########
        # EXTENSION

        @reactive.Effect
        def update_ext_ids():

            events = get_events_obj_filtered().events

            if events is not None and len(
                    events
            ) > 0 and input.in_switch_ext_random() and input.in_numeric_ext_num() is not None:

                traces = events.trace

                num_traces = min(len(traces), input.in_numeric_ext_num())
                traces = traces.sample(num_traces)

                ids = ""
                for idx in traces.index.tolist():
                    ids += f"{idx},"

                ids = ids[:-1]

                ui.update_text("in_text_ext_ids", value=ids)

        @output
        @render.ui
        def out_ext_dyn_plot():

            dyn_height = f"{200 + input.in_numeric_panel_height_ext() * input.in_numeric_norm_num()}px"
            return xui.output_plot("out_plot_ext", height=dyn_height)

        @output
        @render.plot()
        def out_plot_ext():

            events = get_events_obj_filtered().events
            ext = get_events_obj_extended().events

            if events is not None:

                # get ids
                ids = input.in_text_ext_ids()
                if len(ids) < 1:
                    ids = []

                else:
                    ids = ids.replace(" ", "")
                    ids = [int(idx) for idx in ids.split(",")]

                if len(ids) < 1:
                    return None

                # create figure
                fig, axx = plt.subplots(
                    len(ids), input.in_numeric_plot_columns_ext(), sharex=input.in_switch_ext_sharex(),
                    sharey=input.in_switch_ext_sharey()
                )

                axx = list(axx.flatten()) if len(ids) > 1 else [axx]

                for i_ in range(len(ids) + 1, len(axx)):
                    fig.delaxes(axx[i_])

                # plot original
                if input.in_switch_ext_original():

                    for i, idx in enumerate(ids):
                        row = events.loc[idx]

                        X = range(row.z0, row.z1)
                        Y = np.array(row.trace)

                        axx[i].plot(X, Y, color="black", alpha=0.5, label="data")
                        axx[i].set_ylabel(f"idx: {idx}")

                # plot extended
                if ext is not None and input.in_switch_extend():

                    for i, idx in enumerate(ids):
                        row = ext.loc[idx]

                        X = range(row.z0, row.z1)
                        Y = np.array(row.trace)

                        axx[i].plot(X, Y, linestyle="--", color="red", alpha=0.5, label="extended")
                        axx[i].set_ylabel(f"idx: {idx}")

                handles, labels = axx[-1].get_legend_handles_labels()
                fig.legend(
                    handles, labels, loc='upper right', framealpha=1, facecolor="white", borderaxespad=2
                )

                return fig

            return None

        ###############
        # NORMALIZATION

        @reactive.Effect
        def update_norm_ids():

            events = get_events_obj_extended().events

            if events is not None and len(
                    events
            ) > 0 and input.in_switch_norm_random() and input.in_numeric_norm_num() is not None:

                traces = events.trace

                num_traces = min(len(traces), input.in_numeric_norm_num())
                traces = traces.sample(num_traces)

                ids = ""
                for idx in traces.index.tolist():
                    ids += f"{idx},"

                ids = ids[:-1]

                ui.update_text("in_text_norm_ids", value=ids)

        @output
        @render.ui
        def out_dyn_plot():

            dyn_height = f"{200 + input.in_numeric_panel_height() * input.in_numeric_norm_num()}px"
            return xui.output_plot("out_plot_norm", height=dyn_height)

        @output
        @render.plot()
        def out_plot_norm():

            events = get_events_obj_extended().events
            norm = get_events_obj_normalized().events

            if events is not None and norm is not None and len(events) > 0:

                # traces = events.trace
                # norm_traces = norm.trace
                #
                # # convert to nupy array
                # norm_traces = [np.array(x) for x in norm_traces.tolist()]
                # norm_traces = pd.Series(data=norm_traces, index=traces.index)

                # get ids
                ids = input.in_text_norm_ids()
                if len(ids) < 1:
                    ids = []

                else:
                    ids = ids.replace(" ", "")
                    ids = [int(idx) for idx in ids.split(",")]

                if len(ids) < 1:
                    return None

                # create figure
                fig, axx = plt.subplots(
                    len(ids), input.in_numeric_plot_columns(), sharex=input.in_switch_norm_sharex(),
                    sharey=input.in_switch_norm_sharey()
                )

                axx = list(axx.flatten()) if len(ids) > 1 else [axx]

                for i_ in range(len(ids) + 1, len(axx)):
                    fig.delaxes(axx[i_])

                # plot original
                if input.in_switch_norm_original():

                    axx_twin = [ax.twinx() for ax in axx]

                    if input.in_switch_norm_sharey():
                        for ax in axx_twin[1:]:
                            ax.sharey(axx_twin[0])

                    for i, idx in enumerate(ids):
                        row = events.loc[idx]

                        X = range(row.z0, row.z1)
                        Y = np.array(row.trace)

                        axx_twin[i].plot(X, Y, color="black", alpha=0.5, label="data")
                        axx[i].set_ylabel(f"idx: {idx}")

                if (
                        input.in_switch_norm_default() and input.in_select_norm_mode() != "") or not input.in_switch_norm_default():

                    for i, idx in enumerate(ids):
                        row = norm.loc[idx]

                        X = range(row.z0, row.z1)
                        Y = np.array(row.trace)

                        axx[i].plot(X, Y, linestyle="--", color="red", alpha=0.5, label="normalized")
                        axx[i].set_ylabel(f"idx: {idx}")

                handles, labels = axx[-1].get_legend_handles_labels()
                fig.legend(
                    handles, labels, loc='upper right', framealpha=1, facecolor="white", borderaxespad=2
                )

                return fig

            return None

        ##########
        # ENCODERS
        ##########

        @reactive.Calc
        def get_embedding():

            event_obj = get_events_obj_normalized()
            if event_obj is not None:

                if input.in_switch_encoder_use_fe() and input.in_select_encoding_options() == "Feature Extraction":
                    fe = FeatureExtraction(event_obj)

                    # extracted features
                    features = fe.all_features()

                    if input.in_switch_encoder_fe_add_def_columns():
                        events = get_events_obj_filtered().events.copy()
                        v_columns = [col for col in events.columns if col.startswith("v_")]
                        def_features = events[v_columns]

                        # combine
                        features = pd.concat([def_features, features], axis=1)

                    if features.isnull().values.any():

                        nan_settings = input.in_select_encoder_fe_nan_settings()
                        if nan_settings == "" or nan_settings is None:
                            ui.notification_show("NaN values present. Please choose a NaN behavior.", type="error")
                            return None

                        elif nan_settings == "fill":

                            fixed_value = input.in_numeric_encoder_fe_nan_fixed()
                            if fixed_value is None:
                                fixed_value = -np.finfo(np.float32).max

                            features = features.fillna(fixed_value)

                        elif nan_settings == "column":
                            features = features.dropna(axis=1, how='any')

                        elif nan_settings == "rows":
                            features = features.dropna(axis=0, how='any')

                    return features

                elif input.in_switch_cnn_use_cnn() and input.in_select_encoding_options() == "Convolutional Neural Network (CNN)":
                    embedding = embedd_cnn()
                    return embedding

                elif input.in_switch_encoding_use_rnn() and input.in_select_encoding_options() == 'Recurrent Neural Network (RNN)':
                    embedding = embedd_rnn()
                    return embedding

            return None

        pca_explained_variance_ration = reactive.Value([])

        @output
        @render.text
        def out_text_reduction_pca_quality():

            ratios = pca_explained_variance_ration.get()
            if len(ratios) > 0:
                out_str = "variance_ratios:\n"
                for i, r in enumerate(ratios):
                    out_str += f"dim{i}:{np.round(r, 3)}\n"
                return out_str
            return None

        @reactive.Calc
        def get_reduction():

            # get embeddings
            embedding = get_embedding()

            if embedding is not None:

                if isinstance(embedding, list):
                    embedding_data = np.array(embedding)
                elif isinstance(embedding, (pd.Series, pd.DataFrame)):
                    embedding_data = embedding.values
                else:
                    embedding_data = embedding

                if input.in_select_dimensionality_reduction_options() == "PCA":

                    if input.in_switch_encoding_pca_auto_dim():
                        n_components = "mle"
                    else:
                        n_components = input.in_numeric_encoding_pca_n_components()

                    pca = sklearn.decomposition.PCA(n_components=n_components)
                    X_embedded = pca.fit_transform(embedding_data)

                    pca_explained_variance_ration.set(pca.explained_variance_ratio_)

                    return pd.DataFrame(X_embedded)

                elif input.in_select_dimensionality_reduction_options() == "tSNE":
                    X_embedded = TSNE(n_components=input.in_numeric_encoding_tsne_n_components(), init='random',
                                      perplexity=input.in_numeric_encoding_tsne_early_exaggeration(),
                                      n_iter=input.in_numeric_encoding_tsne_n_iter(),
                                      early_exaggeration=input.in_numeric_encoding_tsne_early_exaggeration()).fit_transform(
                        embedding_data
                    )
                    return pd.DataFrame(X_embedded)

                elif input.in_select_dimensionality_reduction_options() == "UMAP":
                    umap_ = umap.UMAP(
                        n_neighbors=input.in_numeric_encoding_umap_neighbors(),
                        min_dist=input.in_numeric_encoding_umap_min_dist(),
                        n_components=input.in_numeric_encoding_umap_n_components(),
                        metric=input.in_text_encoding_umap_metric()
                    )
                    X_embedded = umap_.fit_transform(embedding_data)
                    return pd.DataFrame(X_embedded)

                else:
                    return embedding

            return None

        @output
        @render.data_frame
        def out_data_frame_encoding_():

            embedding = get_embedding()
            if embedding is not None:

                if isinstance(embedding, list):
                    embedding = pd.DataFrame({"trace": embedding})
                elif isinstance(embedding, np.ndarray):
                    embedding = pd.DataFrame({"trace": embedding.tolist()})

                return render.DataTable(embedding)
            return None

        @reactive.Effect
        def update_reduction_selectors():

            reduction = get_reduction()
            if reduction is not None and isinstance(reduction, pd.DataFrame):

                columns = [f"dim{i}" for i in reduction.columns]
                if len(columns) > 1:
                    ui.update_select("out_select_reduction_dim1", choices=columns, selected=columns[0])
                    ui.update_select("out_select_reduction_dim2", choices=columns, selected=columns[1])

        @output
        @render.data_frame
        def out_data_frame_reduction_():

            reduction = get_reduction()
            if reduction is not None:
                reduction.columns = [f"dim{i}" for i in range(len(reduction.columns))]
                reduction = reduction.round(3)

                return render.DataTable(reduction)
            return None

        ####
        # FE

        @output
        @render.plot
        def out_plot_reduction_scatter():
            reduction = get_reduction()
            if reduction is not None:
                dim1 = input.out_select_reduction_dim1()
                dim2 = input.out_select_reduction_dim2()

                fig, ax = plt.subplots()
                ax.scatter(reduction[dim1], reduction[dim2])
                ax.set_xlabel(dim1)
                ax.set_ylabel(dim2)

                return fig
            return None

        #####
        # CNN
        @output
        @render.text
        def out_text_cnn_warning_ragged():

            event_obj = get_events_obj_normalized().copy()

            if event_obj is not None:

                traces = event_obj.events.trace

                if is_ragged(traces):
                    return f"Warning! Traces are of unequal length and cannot be used with CNN encoding."
            return None

        @reactive.Calc
        def get_CNN():

            if input.in_switch_cnn_use_cnn():

                event_obj = get_events_obj_normalized()
                if event_obj is not None:
                    data = event_obj.events.trace.tolist()
                    target_length = len(data[0])

                    cnn = CNN_Autoencoder(target_length, dropout=input.in_numeric_cnn_dropout(),
                                          l1_reg=input.in_numeric_cnn_regularize(),
                                          latent_size=input.in_numeric_cnn_latent_size())

                    with reactive.isolate():
                        train_split = input.in_slider_cnn_split()
                        val_split = 1 - input.in_slider_cnn_split()
                        patience = input.in_numeric_cnn_patience()
                        min_delta = input.in_numeric_cnn_min_delta()
                        batch_size = input.in_numeric_cnn_batch_size()

                    X_train, X_val, X_test = cnn.split_dataset(
                        data, train_split=train_split, val_split=val_split
                    )

                    cnn.train_autoencoder(
                        X_train, X_val, X_test, patience=patience, min_delta=min_delta,
                        epochs=input.in_numeric_cnn_epochs(), learning_rate=input.in_numeric_cnn_learning_rate(),
                        batch_size=batch_size
                    )

                    return cnn, (X_train, X_val, X_test)
            return None

        @reactive.Calc
        def embedd_cnn():

            event_obj = get_events_obj_normalized()
            if input.in_switch_cnn_use_cnn() and event_obj is not None:
                cnn, _ = get_CNN()

                if event_obj is not None:
                    data = np.array(event_obj.events.trace.tolist())
                    latent = cnn.embed(data)
                    return latent
            return None

        @output
        @render.plot
        def out_plot_cnn_history():

            if input.in_switch_cnn_use_cnn():
                cnn, _ = get_CNN()

                if cnn is not None:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    losses = np.array(cnn.losses)

                    ax.plot(losses[:, 0], color="black", label="training")
                    ax.plot(losses[:, 1], color="green", label="validation")

                    ax.set_title(f"losses")
                    ax.set_yscale("log")
                    ax.legend()

            return None

        @output
        @render.plot
        def out_plot_cnn_examples():

            if input.in_switch_cnn_use_cnn():
                cnn, (X_train, X_test, X_val) = get_CNN()

                fig = cnn.plot_examples_pytorch(
                    X_test, show_diff=input.in_switch_cnn_show_diff(), num_samples=input.in_numeric_cnn_num_samples(),
                    figsize=(10, 6)
                )
                return fig
            return None

        # RNN
        @reactive.Calc
        def get_RNN():

            event_obj = get_events_obj_normalized()
            if input.in_switch_encoding_use_rnn() and event_obj is not None:
                events = event_obj.events

                # TODO normalization???

                pdl = PaddedDataLoader(events.trace.tolist())
                X_train, X_val, X_test = pdl.get_datasets(
                    batch_size=input.in_numeric_rnn_batch_size(), val_size=input.in_numeric_rnn_val_size(),
                    test_size=input.in_numeric_rnn_test_size()
                )

                params = {'num_traces': len(events), 'num_features': 1, 'encoder_lr': input.in_numeric_rnn_encode_lr(),
                          'decoder_lr': input.in_numeric_rnn_decode_lr(),
                          'rnn_hidden_dim': input.in_numeric_rnn_hidden(),
                          'num_layers': input.in_numeric_rnn_num_layers(), 'dropout': input.in_numeric_rnn_dropout(),
                          'initialize_repeat': input.in_switch_rnn_initialize_repeat(),
                          'clip': input.in_numeric_rnn_clip(), }

                rnnAE = TimeSeriesRnnAE(**params)

                train_params = {'dataloader_train': X_train, 'dataloader_val': X_val,
                                'num_epochs': input.in_numeric_rnn_num_epochs(),
                                'diminish_learning_rate': input.in_numeric_rnn_diminish_lr(),
                                'patience': input.in_numeric_rnn_patience(),
                                'min_delta': input.in_numeric_rnn_min_delta(), 'show_mode': "progress"}

                rnnAE.train_epochs(dataloader_train=X_train, dataloader_val=X_val, **train_params)

                return rnnAE, (X_train, X_val, X_test)
            return None, (None, None, None)

        @reactive.Calc
        def embedd_rnn():

            if input.in_switch_encoding_use_rnn():

                rnnAE, _ = get_RNN()
                event_obj = get_events_obj_normalized()

                if rnnAE is not None and event_obj is not None:
                    data = event_obj.events.trace.tolist()

                    pdl = PaddedDataLoader(data)
                    loader = pdl.get_dataloader(data, batch_size=32, shuffle=False)

                    _, _, latent, _ = rnnAE.embedd(loader)

                    return latent
            return None

        @output
        @render.plot
        def out_plot_rnn_history():

            rnnAE, (X_train, X_val, X_test) = get_RNN()
            if rnnAE is not None:

                fig, axx = plt.subplots(1, 2, figsize=(9, 4))

                axx[0].plot(rnnAE.train_losses, color="black", label="training")
                if X_val is not None:
                    axx[0].plot(np.array(rnnAE.val_losses).flatten(), color="green", label="validation")

                axx[0].set_title(f"losses")
                axx[0].set_yscale("log")
                axx[0].legend()

                lrates = np.array(rnnAE.learning_rates)
                axx[1].plot(lrates[:, 0], color="green", label="encoder", linestyle="--")
                axx[1].plot(lrates[:, 1], color="red", label="decoder", linestyle="--")
                axx[1].set_title(f"learning rates")
                axx[1].legend()

                fig.suptitle(f"Epoch {len(rnnAE.train_losses)}/{input.in_numeric_rnn_num_epochs()}")
                return fig
            return None

        @output
        @render.ui
        def out_plot_rnn_ui():
            return ui.output_plot(
                "out_plot_rnn_examples", width=f"{input.in_numeric_rnn_plot_height()}px",
                height=f"{input.in_numeric_rnn_plot_width()}px"
            )

        @output
        @render.plot
        def out_plot_rnn_examples():

            rnnAE, (X_train, X_val, X_test) = get_RNN()
            if rnnAE is not None:
                fig, _, _, _, _ = rnnAE.plot_traces(
                    X_test, n_samples=input.in_numeric_rnn_num_samples()
                )
                return fig
            return None

        ###############
        # Experiments #
        ###############

        @reactive.Calc
        def read_kwargs():

            textbox_input = input.in_textarea_inc_kwargs()

            if textbox_input == "":
                return {}

            else:

                try:
                    parsed_dict = yaml.safe_load(textbox_input)
                    print(parsed_dict)
                    if not isinstance(parsed_dict, dict):
                        raise ValueError("The input does not represent a dictionary.")
                    return parsed_dict
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML: {e}")
                    return None

        #########################
        # Conditional Contrasts #
        #########################

        @reactive.Effect
        def update_contrast_groups():

            events = get_events_obj_filtered().events

            if events is not None:
                cols = [col for col in events.columns if
                        events[col].dtype == "category" and len(np.unique(events[col].dropna())) > 1]

                ui.update_select("in_select_contrast_groups", choices=["", "dummy"] + cols)

        @reactive.Effect
        def update_contrast_bias():

            events = get_events_obj_filtered().events

            if events is not None:
                cols = [col for col in events.columns if events[col].dtype != "category" and col.startswith("v_")]

                ui.update_select("in_select_contrast_bias", choices=[""] + cols)

        @reactive.Calc
        def get_contrast_classes():

            events = get_events_obj_filtered()
            if events is not None and input.in_select_contrast_groups() != "":

                num_contrasts = input.in_numeric_contrast_num_groups()
                group_column = input.in_select_contrast_groups()
                if group_column == "dummy":

                    bias = input.in_select_contrast_bias()
                    if bias != "":

                        bias_column = events.events[bias].astype(float)

                        contrasts = pd.qcut(bias_column, num_contrasts, labels=(0, num_contrasts))
                        contrasts = np.array(contrasts.tolist())
                    else:
                        contrasts = np.random.randint(0, num_contrasts, size=len(events))

                    mapping_dict = {v: v for v in np.unique(contrasts)}
                    rev_mapping_dict = {v: k for k, v in mapping_dict.items()}

                else:
                    values = events.events[group_column]
                    unique_values = values.unique()

                    mapping_dict = {v: i for i, v in enumerate(unique_values)}
                    rev_mapping_dict = {v: k for k, v in mapping_dict.items()}
                    contrasts = np.array(values.map(mapping_dict).tolist())

                return contrasts, rev_mapping_dict
            return None, None

        @reactive.Calc
        def get_contrast_classifier():

            events = get_events_obj_filtered()
            contrasts, contrast_map = get_contrast_classes()
            embedding = get_reduction()
            if events is not None and embedding is not None and contrasts is not None:

                if isinstance(embedding, pd.DataFrame):
                    embedding = embedding.values

                train_split = input.in_numeric_contrast_train_split()
                balance_training_set = input.in_switch_contrast_balance_training()
                balance_test_set = input.in_switch_contrast_balance_test()

                discr = Discriminator(events=events)
                discr.split_dataset(
                    embedding, contrasts, split=train_split, balance_training_set=balance_training_set,
                    balance_test_set=balance_test_set
                )

                extra_arguments = read_kwargs()
                clf = discr.train_classifier(classifier=input.in_select_contrast_classifier(), **extra_arguments)

                norm_conf_matrix = input.in_select_contrast_norm_confusion_matrix()
                evaluation = discr.evaluate(cutoff=None, normalize=norm_conf_matrix if norm_conf_matrix != "" else None)

                predictions = clf.predict(embedding)

                return clf, evaluation, predictions
            return None, None, None

        @output
        @render.plot
        def out_plot_contrast_classifier():

            embedding = get_reduction()
            contrasts, contrast_map = get_contrast_classes()
            clf, evaluation, predictions = get_contrast_classifier()
            if clf is not None:

                # Create the figure
                fig = plt.figure(figsize=(20, 20))

                # Create a GridSpec layout with 2 rows and 2 columns
                gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

                # Create subplots
                ax1 = plt.subplot(gs[0, 0])  # A
                ax2 = plt.subplot(gs[1, 0])  # B
                ax3 = plt.subplot(gs[:, 1])  # C

                # plot quality control
                cmd_1 = ConfusionMatrixDisplay(evaluation[0])
                cmd_1.plot(ax=ax1)
                ax1.set_title("training confusion matrix")

                cmd_2 = ConfusionMatrixDisplay(evaluation[1])
                cmd_2.plot(ax=ax2)
                ax2.set_title("test confusion matrix")

                if isinstance(embedding, pd.DataFrame):
                    embedding = embedding.values

                # plot UMAP
                if embedding.shape[1] > 2:
                    umap_ = umap.UMAP()
                    two_dim_embedding = umap_.fit_transform(embedding)
                else:
                    two_dim_embedding = embedding

                available_markers = list(Line2D.markers.keys())
                filtered_markers = [m for m in available_markers if m not in ['None', ' ', '', 'None|', None]]

                labeled = {c: False for c in np.unique(contrasts)}
                for i in range(len(contrasts)):
                    true_, pred = contrasts[i], predictions[i]

                    ax3.scatter(
                        two_dim_embedding[i, 0], two_dim_embedding[i, 1], c="green" if true_ == pred else "red",
                        marker=filtered_markers[true_ + 2], label=contrast_map[true_] if not labeled[true_] else None
                    )

                    labeled[true_] = True

                ax3.legend()

                return fig
            return None

        @reactive.Calc
        def get_barycenters():

            events = get_events_obj_normalized()
            if events is not None:

                events = events.copy()
                linkage = Linkage()

                if input.in_switch_contrast_use_embedding():

                    embedding = get_embedding()
                    if embedding is None:
                        ui.notification_show(f"cannot find embedding; please calculate with 'Encoding' tab.")
                        return None

                    if isinstance(embedding, pd.DataFrame):
                        embedding = np.array(embedding.values).tolist()
                    elif isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    events.events.trace = embedding

                cutoff = input.in_numeric_contrasts_bary_cutoff()
                criterion = input.in_select_contrasts_bary_crit()
                distance_type = "pearson" if input.in_switch_contrast_use_embedding() else "dtw"
                barycenters, lut, Z = linkage.get_barycenters(
                    events, cutoff=cutoff, criterion=criterion, distance_type=distance_type, return_linkage_matrix=True
                )

                return barycenters, lut, Z
            return None, None

        @output
        @render.plot
        def out_plot_contrast_hierarchical_left():

            barycenters, lut, Z = get_barycenters()
            contrasts, mapping_dict = get_contrast_classes()
            events = get_events_obj_filtered()
            if events is not None and barycenters is not None and contrasts is not None:

                # Create the figure
                fig = plt.figure(figsize=(10, 10))

                # Create a GridSpec layout with 2 rows and 2 columns
                gs = gridspec.GridSpec(2, 1)

                # Create subplots
                ax1 = plt.subplot(gs[0, 0])  # A
                ax2 = plt.subplot(gs[1, 0])  # B

                # Dendogram
                p = input.in_numeric_contrasts_bary_cutoff()
                if input.in_select_contrasts_bary_crit() == "maxclust":
                    truncate_mode = "lastp"
                else:
                    truncate_mode = "level"

                _ = dendrogram(
                    Z, p=p, truncate_mode=truncate_mode, color_threshold=7, orientation="left", leaf_font_size=12,
                    above_threshold_color="black", ax=ax1,
                )

                # Barplot
                events = events.copy()
                cluster_column = "bary_cluster"
                events.add_clustering(cluster_lookup_table=lut, column_name=cluster_column)
                events.add_clustering(
                    cluster_lookup_table={idx: v for idx, v in list(zip(events.events.index.tolist(), contrasts))},
                    column_name="contrast"
                )

                sel = pd.DataFrame(
                    {"dz": events.events.dz, cluster_column: events.events.bary_cluster, "contrasts": contrasts,
                     "contrast_name": [mapping_dict[c] for c in contrasts]}
                )
                sel = sel.groupby([cluster_column, "contrasts"]).count().reset_index()
                sel[cluster_column] = sel[cluster_column].astype(str)
                sns.barplot(data=sel, x="contrasts", y="dz", hue=cluster_column, ax=ax2)

                return fig
            return None

        @output
        @render.plot
        def out_plot_contrast_hierarchical_right():

            barycenters, lut, Z = get_barycenters()
            contrasts, mapping_dict = get_contrast_classes()
            events = get_events_obj_filtered()
            if events is not None and barycenters is not None and contrasts is not None:
                events = events.copy()
                cluster_column = "bary_cluster"
                events.add_clustering(cluster_lookup_table=lut, column_name=cluster_column)
                events.add_clustering(
                    cluster_lookup_table={idx: v for idx, v in list(zip(events.events.index.tolist(), contrasts))},
                    column_name="contrast"
                )

                # Clustermap
                counts = events.get_counts_per_cluster(cluster_column, group_col="contrast")

                # group_color_col = input.in_text_contrasts_cl_group_color_col()
                # if group_color_col != "":
                #     color_palette_ = sns.color_palette(input.in_select_contrasts_cl_cmap(), len(group_color_col))
                #

                in_dict = {"": None, "rows": 0, "columns": 1}
                clustermap, pred_scores = events.plot_cluster_counts(
                    counts, normalize_instructions=None, method=input.in_select_contrasts_cl_method(),
                    metric=input.in_select_contrasts_cl_metric(),
                    z_score=in_dict[input.in_select_contrasts_cl_z_score()],
                    center=in_dict[input.in_select_contrasts_cl_center()],
                    color_palette=input.in_select_contrasts_cl_cmap_group(), group_cmap=None,
                    cmap=input.in_select_contrasts_cl_cmap(), transpose=input.in_select_contrasts_cl_transpose()
                )

                print("clustermap scores: ", pred_scores)

                fig = clustermap.ax_heatmap.get_figure()
                return fig
            return None

        ##########################
        # Coincidence Prediction #
        ##########################

        @reactive.Calc
        def get_incidences():

            events = get_events_obj_filtered()
            if events is not None:
                if input.in_switch_inc_create_dummy():

                    events = events.events
                    z0, z1 = events.z0.min(), events.z1.max()

                    avg_length = input.in_numeric_inc_dummy_distance()
                    min_gap = input.in_numeric_inc_dummy_gap()

                    incidences = []
                    z = 0
                    while z < z1:
                        loc = np.random.randint(z, z + avg_length, size=1)[0]
                        incidences.append(loc)

                        z = loc + min_gap

                    incidences = np.array(incidences)

                else:

                    if input.in_file_inc_file() is not None:

                        path = Path(input.in_file_inc_file())

                        if not path.exists():
                            ui.notification_show(f"cannot find file: {path}", type="error")
                            return None

                        io = IO()
                        incidences = io.load(path)
                    else:
                        return None

                return incidences
            return None

        @reactive.Calc
        def get_coincidence_prediction():

            events = get_events_obj_filtered()
            embedding = get_reduction()
            if embedding is not None:

                incidences = get_incidences()

                train_split = input.in_numeric_inc_train_split()
                balance_training_set = input.in_switch_inc_balance_training()
                balance_test_set = input.in_switch_inc_balance_test()

                cd = CoincidenceDetection(
                    events, incidences, embedding, train_split=train_split, balance_training_set=balance_training_set,
                    balance_test_set=balance_test_set
                )

                extra_arguments = read_kwargs()

                if input.in_switch_inc_predict_coincidence():

                    norm_confusion = input.in_select_inc_norm_confusion_matrix()

                    clf, evaluation = cd.predict_coincidence(
                        binary_classification=input.in_switch_inc_binary_class(),
                        classifier=input.in_select_inc_classifier(),
                        normalize_confusion_matrix=norm_confusion if norm_confusion != "" else None, **extra_arguments
                    )

                else:

                    clf, evaluation = cd.predict_incidence_location(
                        classifier=input.in_select_inc_classifier_loc(), **extra_arguments
                    )

                return clf, evaluation

            return None

        @output
        @render.plot
        def out_plot_inc():

            inc_pred = get_coincidence_prediction()
            if inc_pred is not None:

                clf, evaluation = inc_pred
                labels = ["train", "test"]

                if isinstance(evaluation[0], float):
                    fig, ax = plt.subplots()
                    for i in range(len(evaluation)):
                        ax.scatter(i, evaluation[i], label=labels[i])

                    ax.set_title(f"Score: {evaluation[1]:.4f}")
                    ax.legend()

                else:

                    fig, axx = plt.subplots(1, 2, figsize=(15, 15))

                    for i, cm in enumerate(evaluation):
                        cmd = ConfusionMatrixDisplay(cm)
                        cmd.plot(ax=axx[i])
                        axx[i].set_title(labels[i])

                return fig
            return None

    def plot_images(self, arr, frames, lbls=None, figsize=(10, 5), vmin=None, vmax=None):

        if not isinstance(arr, list):
            arr = [arr]

        if lbls is not None and not isinstance(lbls, list):
            lbls = [lbls]

        N, M = len(arr), len(frames)

        spec = gridspec.GridSpec(nrows=N, ncols=M, wspace=0.2, hspace=0.05)

        figsize = (N * figsize[0], M * figsize[1])
        fig = plt.figure(figsize=figsize)

        for x, img in enumerate(arr):
            for y, f in enumerate(frames):

                if f > img.shape[0]:
                    ui.notification_show(f"frame {f} outide of bounds (0, {img.shape[0]}.", type="error")
                    continue

                ax = fig.add_subplot(spec[x, y])

                ax.imshow(img[f, :, :], vmin=vmin, vmax=vmax)

                ax.set_xticklabels([])
                ax.set_yticklabels([])

                ax.set_xlabel(f"#{f}")

                if lbls is not None:
                    ax.set_ylabel(f"{lbls[x]}")

        return fig

    def get_table_excl(self, df, excl_columns=('contours', 'trace', 'mask', 'footprint')):
        cols = [col for col in df.columns if col not in excl_columns]
        return df[cols]

    def get_table_rounded(self, df):

        def custom_formatter(x):

            try:

                if isinstance(x, (float, np.float64, np.float32)):
                    x = f"{x:.2f}"

                elif isinstance(x, (int, np.int32, np.int64)):
                    x = x

                elif isinstance(x, np.ndarray):
                    x = "[...]"

                elif isinstance(x, list):
                    x = "[...]"

                elif isinstance(x, str):
                    x = x

                else:
                    print(f"unknown datatype: {type(x)}")

            except Exception:
                x = "N/A"

            return x

        return df.map(custom_formatter)

    def run(self, port=8000):
        self.app.run(port=port)


# Run the app
if __name__ == '__main__':
    app_instance = Analysis()
    app_instance.run()
