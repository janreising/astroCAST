import logging
import traceback
from itertools import combinations, permutations
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
from shiny import App, ui, render, reactive

from astrocast import detection
from astrocast.preparation import IO


class Explorer:

    def __init__(self, input_path=None, h5_loc=None):

        self.path = input_path
        self.h5_loc = h5_loc

        self.app_ui = self.create_ui()
        self.app = App(self.app_ui, self.server)

    def create_ui(self):
        return ui.page_fluid(
            ui.panel_title("Argument Explorer"),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.navset_tab(
                        ui.nav("File",
                               ui.h5(""),
                               ui.input_text("path", "Path", value=self.path),
                               ui.input_text("h5_loc", "dataset", value=self.h5_loc),
                               ui.output_text("data_shape"),
                               ui.h5(""),
                               ui.panel_conditional(
                                   "input.path !== ''",
                                   ui.h5("Select frames"),
                                   ui.input_slider("z_select", "", value=(164,194), min=0, max=100),
                               ),
                               ui.h5("Points of Interest"),
                               ui.input_text("frames", "frames", value="1,10,13"),
                               ui.input_text("pixel", "pixels", value='(41, 185, "blue"), (57, 43, "red"), (104, 24, "green")'),
                            ),
                        ui.nav("Smoothing",
                               ui.h5(""),
                               ui.input_switch("use_smoothing", "activate"),
                               ui.input_numeric("sigma", "Sigma", value=3),
                               ui.input_numeric("radius", "Radius", value=2),
                            ),
                        ui.nav("Spatial Thresholding",
                                ui.h5(""),
                                ui.input_switch("use_spatial", "activate"),
                                ui.input_numeric("min_ratio", "Min Ratio", value=1),
                                ui.input_numeric("z_depth", "Z-Depth", value=1),
                            ),

                        ui.nav("Temporal Thresholding",
                                ui.h5(""),
                                ui.input_switch("use_temporal", "activate"),
                                ui.input_numeric("prominence", "Prominence", value=10),
                                ui.input_numeric("width", "Width", value=3),
                                ui.input_numeric("rel_height", "Relative Height", value=0.9),
                                ui.input_numeric("wlen", "Wlen", value=60),
                            ),
                        ui.nav("Morphology",
                                ui.h5(""),
                                ui.h5("Fill Small Holes"),
                                ui.input_switch("use_holes", "fill"),
                                ui.input_numeric("area_threshold", "Area Threshold", value=10),
                                ui.input_numeric("connectivity_holes", "Connectivity", value=1),
                                ui.h5("Remove Small Objects"),
                                ui.input_switch("use_objects", "remove"),
                                ui.input_numeric("min_size", "Min Size", value=10),
                                ui.input_numeric("connectivity_objects", "Connectivity", value=1),
                                ui.panel_conditional(
                                    "input.use_holes & input.use_objects",
                                    ui.h5("Sequential Operation"),
                                    ui.input_radio_buttons("comb_options", "Combination order",
                                                           ["None", "holes > objects", "objects > holes"]),
                                ),
                                ui.panel_conditional(
                                    "input.use_spatial & input.use_temporal",
                                    ui.h5("Combine Spatial and Temporal"),
                                    ui.input_radio_buttons("comb_type", "Union type", ["None", "&", "|"]),
                                )
                            ),
                    ),
                ),
                ## DATA PANEL
                ui.panel_main(
                    ui.panel_conditional(
                        "input.path !== ''",

                        # Original
                        ui.output_plot("original"),

                        ui.panel_conditional(
                            "input.use_smoothing",
                            ui.output_plot("smooth"),
                        ),

                        ui.panel_conditional(
                            "input.use_spatial",
                            ui.output_plot("spatial"),
                        ),

                        ui.panel_conditional(
                            "input.use_temporal",
                            ui.output_plot("temporal"),
                        ),

                        # Binary Morphology
                        ui.panel_conditional(
                            "input.use_holes | input.use_objects",
                            ui.output_plot("morph"),
                        ),

                        # Combination
                        ui.panel_conditional(
                            "input.comb_type !== 'None'",
                            ui.output_plot("combined_"),
                        ),

                    )
                ),
            )
        )

    def server(self, input, output, session):

        @reactive.Calc
        def load_data():

            path = Path(input.path())

            if path.is_file():
                io = IO()

                try:

                    data = io.load(path, h5_loc=input.h5_loc(), z_slice=input.z_select(), lazy=True)
                except Exception as e:
                    ui.notification_show(f"Error: {e}: {traceback.print_exc()}", type="warning")
                    return None

                return data

            else:
                ui.notification_show("Path doesn't exist", type="warning")
                return None

        @reactive.Calc
        def get_data_dimension():

            path = Path(input.path())

            try:
                io = IO()
                data = io.load(path, h5_loc=input.h5_loc(), z_slice=None, lazy=True)
                return data.shape

            except:
                return None

        @reactive.Calc
        def get_smooth():
            data = load_data()
            smooth = detection.Detector.gaussian_smooth_3d(data, sigma=input.sigma(), radius=input.radius())
            return smooth

        @reactive.Calc
        def get_spatial():


            with ui.Progress(min=0, max=2) as p:
                p.set(0, message="Loading data", detail="")

                if input.use_smoothing:
                    data = get_smooth()
                else:
                    data = load_data()

                p.set(1, message="Spatial thresholding.", detail="This may take a while ...")
                spatial = detection.Detector.spatial_threshold(data,
                                                           min_ratio=input.min_ratio(), threshold_z_depth=input.z_depth())

                p.set(2, message="Done")

            return spatial

        @reactive.Calc
        def get_temporal():

            with ui.Progress(min=0, max=2) as p:
                p.set(0, message="Loading data", detail="")

                if input.use_smoothing:
                    data = get_smooth()
                else:
                    data = load_data()

                p.set(1, message="Temporal thresholding.", detail="This may take a while ...")
                temporal = detection.Detector.temporal_threshold(data,
                                                                prominence=input.prominence(), width=input.width(),
                                                                rel_height=input.rel_height(), wlen=input.wlen())

                p.set(2, message="Done")

                return temporal

        @reactive.Calc
        def get_morph():

            with ui.Progress(min=0, max=3) as p:
                i=0
                p.set(i, message="Loading data", detail="")

                data = []
                lbls = []

                if input.use_spatial():
                    data.append(get_spatial())
                    lbls.append("S")

                if input.use_temporal():
                    data.append(get_temporal())
                    lbls.append("T")

                res = []
                res_lbls = []
                for dat, lbl in list(zip(data, lbls)):

                    if input.comb_options() == "None":

                        if input.use_holes():

                            filled = detection.Detector.remove_holes(dat,
                                                              area_threshold=input.area_threshold(),
                                                              connectivity=input.connectivity_holes())
                            res.append(filled)
                            res_lbls.append(lbl+"_fill")

                        if input.use_objects():

                            rem = detection.Detector.remove_objects(dat,
                                                              min_size=input.min_size(),
                                                              connectivity=input.connectivity_holes())
                            res.append(rem)
                            res_lbls.append(lbl+"_rem")

                    elif input.comb_options() == "holes > objects":

                        filled = detection.Detector.remove_holes(dat,
                                                          area_threshold=input.area_threshold(),
                                                          connectivity=input.connectivity_holes())
                        rem = detection.Detector.remove_objects(filled,
                                                          min_size=input.min_size(),
                                                          connectivity=input.connectivity_holes())

                        res.append(rem)
                        res_lbls.append(lbl+"_fill_rem")

                    elif input.comb_options() == "objects > holes":

                        rem = detection.Detector.remove_objects(dat,
                                                          min_size=input.min_size(),
                                                          connectivity=input.connectivity_holes())

                        filled = detection.Detector.remove_holes(rem,
                                                          area_threshold=input.area_threshold(),
                                                          connectivity=input.connectivity_holes())

                        res.append(filled)
                        res_lbls.append(lbl+"_rem_fill")

                return res, res_lbls

        @reactive.Calc
        def get_union():

            if input.comb_type() == "None" or not(input.use_spatial and input.use_temporal):
                return None

            with ui.Progress(min=0, max=12) as p:
                i=0
                p.set(i, message="Loading data", detail="")

                data, lbls = get_morph()
                res_array = []
                res_lbls = []
                for ((d1, l1), (d2, l2)) in permutations(list(zip(data, lbls)), 2):

                    if input.comb_type() == "&":
                        rarr = np.minimum(d1, d2)
                    else:
                        rarr = np.maximum(d1, d2)

                    rarr = rarr.astype(np.bool_)

                    res_array.append(rarr)
                    res_lbls.append(f"{l1}_{l2}")

                return res_array, res_lbls

        @reactive.Calc
        def get_frames():
            frame_str = input.frames()
            if frame_str == "":
                frames = [0]
            else:

                frames = []
                for f in frame_str.split(","):

                    if len(f) > 0:
                        frames.append(int(f))

            return frames

        @output
        @render.text
        def data_shape():

            dim = get_data_dimension()

            if dim is not None:
                return f"shape: {dim}\n"
            else:
                return f"data not found\n"

        @reactive.Effect()
        def _():

            dim = get_data_dimension()
            if dim is not None:
                ui.update_slider(
                    "z_select", min=0, max=dim[0]
                )

        @output
        @render.plot
        def original():

            data = load_data()
            frames = get_frames()

            return self.plot_images([data], frames, lbls=["INPUT"])

        @output
        @render.plot
        def smooth():
            smooth = get_smooth()
            frames = get_frames()

            return self.plot_images([smooth], frames, lbls=["SMOOTH"])

        @output
        @render.plot
        def spatial():
            if input.use_spatial():

                spatial = get_spatial()
                frames = get_frames()

                return self.plot_images([spatial], frames, lbls=["> SPATIAL"])
            else:
                return None

        @output
        @render.plot
        def temporal():
            if input.use_temporal():
                temporal = get_temporal()
                frames = get_frames()

                return self.plot_images([temporal], frames, lbls=["> TEMPORAL"])
            else:
                return None

        @output
        @render.plot
        def morph():

            if input.use_holes() or input.use_objects():

                arr, lbls = get_morph()
                frames = get_frames()

                return self.plot_images(arr, frames, lbls=lbls)
            else:
                return None

        @output
        @render.plot
        def combined_():

            if input.use_temporal() and input.use_spatial() and input.comb_type != "None":

                frames = get_frames()
                arr, lbls = get_union()

                return self.plot_images(arr, frames, lbls=lbls)

            else:
                return None

    def plot_images(self, arr, frames, lbls=None, figsize=(5, 2), vmin=None, vmax=None):

        N, M = len(arr), len(frames)

        spec = gridspec.GridSpec(nrows=N, ncols=M, wspace=0.2, hspace=0.05)

        figsize = (N*figsize[0], M*figsize[1])
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
    app_instance = Explorer()
    app_instance.run()
