import traceback
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt, gridspec
from numba import NumbaDeprecationWarning
from shiny import App, ui, render, reactive, req
import shiny.experimental as x

from astrocast.analysis import Events
from astrocast.preparation import IO
import seaborn as sns

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)
    import umap

class Analysis:

    def __init__(self, input_path=None):

        self.path = input_path

        self.app_ui = self.create_ui()
        self.app = App(self.app_ui, self.server)

    def create_ui(self):

        events_nav = ui.nav(
                        "Events",
                        ui.layout_sidebar(
                            ui.panel_sidebar(
                                ui.h3(""),
                                ui.h3("Settings"),
                                ui.input_text("path", "Event directory", value=self.path),
                                ui.input_text("frames", "frames", value=""),
                                ui.input_switch("switch_show_filters", "filters", value=False),
                                ui.h3("Information"),
                                ui.output_text("out_txt_shape"),
                                ui.output_text("out_txt_number_events"),
                                width=3),
                            ui.panel_main(
                                ui.panel_conditional(
                                    "input.path !== ''",
                                    ui.output_data_frame("out_table_events"),
                                    ui.br(),
                                    ui.output_plot("out_plot_event_map"),
                    ))))

        summary_nav = ui.nav(
                    "Summary statstics",
                    ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.row(
                                ui.column(3, ui.input_switch("in_switch_correlation", "correlation")),
                                ui.column(3, ui.input_switch("in_switch_logscale", "log scale")),
                            ),
                            ui.input_select("in_select_hue", "category", choices=[]),
                            ui.panel_conditional(
                                "input.in_switch_correlation != true",
                                ui.input_select("in_select_type", "",
                                            choices=("boxplot", 'stripplot', "swarmplot", "violinplot")),
                                ui.panel_conditional(
                                     "input.in_select_type == 'boxplot'",
                                     ui.column(3, ui.input_switch("in_switch_fliers", "show outliers", value=True)),
                                 ),
                                ui.panel_conditional(
                                    "input.in_select_type == 'violinplot'",
                                    ui.column(3, ui.input_switch("in_switch_cut", "cut", value=False)),
                                ),
                            ),
                            # ui.br(),
                            ui.h5("Statistics"),
                            ui.output_data_frame("out_table_summary"),
                        ),
                        ui.panel_main(
                            ui.output_plot("out_plot_boxplot")
                        ))
                )

        outliers = ui.nav(
                    "Outliers",
                    ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5("NaN settings"),
                            ui.input_select("in_select_nan_settings", "", choices=["column", "rows", "fill"]),
                            ui.h5(""),
                            ui.h5("UMAP settings"),
                            ui.row(
                                ui.column(6, ui.input_numeric("in_numeric_neighbors", "n_neighbors", value=30),),
                                ui.column(6, ui.input_numeric("in_numeric_min_dist", "min_dist", value=0),),
                            ),
                            ui.input_select("in_text_metric", "metric", choices=['euclidean',
                                'manhattan', 'chebyshev', 'minkowski', 'canberra',
                                'braycurtis', 'haversine', 'mahalanobis', 'wminkowski',
                                'seuclidean', 'Angular and correlation metrics', 'cosine', 'correlation',]
                                            ),
                            ui.h5(""),
                            ui.h5("Plotting settings"),
                            ui.row(
                                ui.column(4, ui.input_numeric("in_numeric_size", "size", value=8)),
                                ui.column(4, ui.input_numeric("in_numeric_alpha", "alpha", value=1, min=0.001, max=1)),
                                ui.column(4, ui.input_numeric("in_numeric_plot_max_cluster", "max traces", value=5)),
                            ),
                            ui.row(
                                ui.column(4, ui.input_switch("in_switch_separate_plots", "Separate", value=False)),
                                ui.column(4, ui.input_switch("in_switch_share_y", "share y", value=False)),
                                ui.column(4, ),
                            ),
                            ui.br(),
                            ui.h5("Clusters"),
                            ui.output_data_frame("out_data_frame_clusters"),
                        width=3),
                        ui.panel_main(
                            ui.output_plot("out_plot_umap"),
                            ui.output_plot("out_plot_traces", height="600px"),
                        )
                    )
        )

        # NAME_nav = ui.nav(
        #             "Title",
        #             ui.layout_sidebar(
        #                 ui.panel_sidebar(),
        #                 ui.panel_main()
        #             )
        # )

        return ui.page_fluid(
            ui.panel_title("Analysis"),
            ui.navset_tab( events_nav, summary_nav, outliers)
        )

    def server(self, input, output, session):

        @reactive.Calc
        def get_events_obj():
            path = Path(input.path())

            if path.exists():
                events = Events(path)
            else:
                events = None

            return events

        @reactive.Calc
        def get_events_table():

            events_obj = get_events_obj()

            if events_obj is None:
                return None

            df = events_obj.events
            df["group"] = np.random.randint(1, 3, size=(len(df)))
            df["subject_id"] = np.random.randint(1, 4, size=(len(df)))

            df.group = df.group.astype("category")
            df.subject_id = df.subject_id.astype("category")

            return df

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

        def custom_formatter(x):

            if isinstance(x, (float, np.float64)):
                x = np.round(x, decimals=2)

                if (x is None) or np.isnan(x):
                    x = "N/A"

            return x

        def get_table_excl(df, excl_columns=('contours', 'trace', 'mask', 'footprint')):
            cols = [col for col in df.columns if col not in excl_columns]
            return df[cols]

        def get_table_rounded(df):
                return df.map(custom_formatter)

        @output
        @render.data_frame()
        def out_table_events():
            events = get_events_table()

            if events is not None:

                events = events.reset_index()

                if input.switch_show_filters():
                    events.columns = [f"{col:_^20}" for col in events.columns]

                events = get_table_excl(events)
                events = get_table_rounded(events)

                events = render.DataTable(events, height="500px", summary=True,
                                          filters=input.switch_show_filters(), row_selection_mode="single")

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

            events = get_events_obj()
            return f"#events: {len(events)}" if events is not None else None

        @reactive.Calc
        def get_summary_table():
            events = get_events_table()
            if events is not None:

                events = get_table_excl(events, excl_columns=('contours', 'trace', 'mask', 'footprint', 'group',
                                                              'file_name', 'z0', 'z1', 'x0', 'x1', 'y0', 'y1', 'subject_id'))

                mean = events.mean(axis=0, skipna=True)
                std = events.std(axis=0, skipna=True)
                summary = pd.DataFrame({"mean":mean, "std":std})

                return summary
            return None

        @output
        @render.data_frame
        def out_table_summary():

            summary = get_summary_table()

            if summary is not None:
                summary = get_table_rounded(summary)
                summary = summary.reset_index()

                summary = render.DataTable(summary, height=None, summary=False, row_selection_mode="multiple")

                return summary
            return None

        # SUMMARY

        @reactive.Effect
        def update_in_select_hue():

            events = get_events_table()
            cols = [col for col in events.columns if events[col].dtype =="category" and len(np.unique(events[col].dropna())) > 1]

            ui.update_select("in_select_hue", choices=[""] + cols)

        @output
        @render.plot()
        def out_plot_boxplot():

            events = get_events_table()
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
                            events[col[1]] = events[col[0]].astype(float)

                            fig = seaborn.jointplot(data=events, x=col[0], y=col[1], hue=hue,
                                                    kind="reg" if hue is None else 'scatter')

                        else:

                            g = sns.PairGrid(events,
                                             x_vars=col, y_vars=col,
                                             hue=hue, diag_sharey=False)

                            g.map_lower(sns.scatterplot, s=15)
                            g.map_diag(sns.kdeplot, lw=2)
                            g.map_upper(sns.kdeplot, fill=True)

                            fig = g

                    else:

                        id_vars = ["index", hue] if hue != None else "index"

                        events = events.reset_index()
                        events = events.melt(id_vars=id_vars, value_vars=col)
                        events.value = events.value.astype(float)

                        typ = input.in_select_type()
                        if typ == "boxplot":
                            fig = seaborn.boxplot(data=events, x="variable", y="value", hue=hue,
                                                  showfliers=input.in_switch_fliers())

                        elif typ == "stripplot":
                            fig = seaborn.stripplot(data=events, x="variable", y="value", hue=hue)

                        elif typ == "swarmplot":
                            fig = seaborn.swarmplot(data=events, x="variable", y="value", hue=hue)

                        elif typ == "violinplot":
                            fig = seaborn.violinplot(data=events, x="variable", y="value", hue=hue,
                                                     split=True if len(id_vars)==2 and len(np.unique(events[hue]))==2 else False,
                                                     cut=0 if input.in_switch_cut() else 2
                                                     )

                    if input.in_switch_logscale():
                        plt.yscale('log')

                    return fig

            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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

        @reactive.Calc
        def get_table_no_nan():

            events = get_events_table()

            if events is not None:
                events = get_table_excl(events, excl_columns=('contours', 'trace', 'mask', 'footprint', 'group',
                                      'file_name', 'z0', 'z1', 'x0', 'x1', 'y0', 'y1', 'subject_id'))

                if events.isnull().values.any():
                    ui.notification_show("None values present. Trying to fix automatically.",
                                         type="error", duration=5)

                    ui.input_select("in_select_nan_settings", "", choices=["column", "rows", "fill"]),

                    if input.in_select_nan_settings() == "fill":
                        max_negative_float32 = -np.finfo(np.float32).max
                        events = events.fillna(max_negative_float32)

                    elif input.in_select_nan_settings() == "column":
                        events = events.dropna(axis=1, how='any')

                    elif input.in_select_nan_settings() == "rows":
                        events = events.dropna(axis=0, how='any')

                return events
            return None

        @reactive.Calc
        def get_embedding():

            events = get_table_no_nan()

            if events is not None:
                # embed
                reducer = umap.UMAP(
                    n_neighbors=input.in_numeric_neighbors(),
                    min_dist=input.in_numeric_min_dist(),
                    n_components=2,
                    metric=input.in_text_metric()
                )

                embedding = reducer.fit_transform(events.values)

                # find outliers
                hdb = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=2,
                               allow_single_cluster=True, core_dist_n_jobs=-1,
                               prediction_data=True)

                hdb_labels =  hdb.fit_predict(embedding, y=None)

                return embedding, hdb_labels
            return None, None

        @output
        @render.plot
        async def out_plot_umap():

            embedding, hdb_labels = get_embedding()

            if embedding is not None:

                fig, ax = plt.subplots(1, 1)

                palette = sns.color_palette("husl", len(np.unique(hdb_labels)))

                ax.scatter(embedding[:, 0], embedding[:, 1],
                           alpha=input.in_numeric_alpha(), s=input.in_numeric_size(),
                           color=[palette[v] for v in hdb_labels])

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

            events = get_events_table()
            events = events.iloc[indices]

            if len(events) > input.in_numeric_plot_max_cluster():
                events = events.sample(input.in_numeric_plot_max_cluster())

            if input.in_switch_separate_plots():
                fig, axx = plt.subplots(len(events), 1, figsize=(20, 5*len(events)),
                                        sharey=input.in_switch_share_y())
            else:
                fig, axx = plt.subplots(1, 1, figsize=(20, 5))
                axx = [axx for _ in range(len(events))]

            for i, (idx, row) in enumerate(events.iterrows()):

                if input.in_switch_separate_plots():
                    axx[i].plot(range(row.z0.astype(int), row.z1.astype(int)), row.trace)
                    axx[i].set_ylabel(f"idx: {idx}")
                else:
                    axx[i].plot(range(row.dz.astype(int)), row.trace)
                    axx[i].set_ylabel(f"idx: {events.index.tolist()}")





        # @output
        # @render.text
        # @reactive.event(input.btn_save)
        # async def argument_file():
        #
        #     save_path = Path(input.save_path())
        #
        #     arguments = {"detect-events": {
        #         # file params
        #         "h5_loc": input.h5_loc(),
        #         # smoothing
        #         "use_smoothing": input.use_smoothing(),
        #         "smooth_sigma": input.sigma(),
        #         "smooth_radius": input.radius(),
        #         # Spatial
        #         "use_spatial": input.use_spatial(),
        #         "spatial_min_ratio": input.min_ratio(),
        #         "spatial_z_depth": input.z_depth(),
        #         # Temporal
        #         "use_temporal": input.use_temporal(),
        #         "temporal_prominence": input.prominence(),
        #         "temporal_width": input.width(),
        #         "temporal_rel_height": input.rel_height(),
        #         "temporal_wlen": input.wlen(),
        #         # Morphological operations
        #         "fill_holes": input.use_holes(),
        #         "area_threshold": input.area_threshold(),
        #         "holes_connectivity": input.connectivity_holes(),
        #         "holes_depth": input.holes_depth(),
        #         "remove_objects": input.use_objects(),
        #         "min_size": input.min_size(),
        #         "object_connectivity": input.connectivity_objects(),
        #         "objects_depth": input.objects_depth(),
        #         "fill_holes_first": True if  input.comb_options() == "holes > objects" else False,
        #         "comb_type": str(input.comb_type()),
        #         # additional
        #         "output_path": input.output_path(),
        #         "exclude_border": input.exclude_border(),
        #         "split_events": input.split_events(),
        #         "overwrite": input.overwrite(),
        #         "logging_level": input.logging_level(),
        #         "debug": input.debug(),
        #     }}
        #
        #     with open(save_path.as_posix(), 'w') as f:
        #         yaml.dump(arguments, f)
        #
        #     ui.notification_show(f"Saved to: {save_path}")
        #
        #     # load
        #     with open(save_path.as_posix(), 'r') as file:
        #         config = yaml.safe_load(file)
        #
        #     # create string
        #     config_string = ""
        #     for section, section_data in config.items():
        #         print("\n")
        #         formatted_section_data = yaml.dump({section: section_data}, indent=4, default_flow_style=True)
        #         config_string += formatted_section_data
        #
        #     return config_string

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

    def run(self):
        self.app.run()


# Run the app
if __name__ == '__main__':
    app_instance = Analysis()
    app_instance.run()
