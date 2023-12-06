import traceback
from pathlib import Path

import dask.array as da
import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt, gridspec
from shiny import App, ui, render, reactive

from astrocast import detection
from astrocast.preparation import IO, Delta


class Explorer:

    def __init__(self, input_path=None, loc=None):

        self.path = input_path
        self.loc = loc

        self.app_ui = self.create_ui()
        self.app = App(self.app_ui, self.server, )

    def create_ui(self):
        return ui.page_fluid(
            ui.panel_title("Argument Explorer"), ui.navset_tab(
                ui.nav(
                    "File", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.input_text("path", "Path", value=self.path),
                            ui.input_text("loc", "dataset", value=self.loc),
                            ui.input_switch("lazy", "lazy loading", value=True), ui.output_text("data_shape"),
                            ui.h5(""), ui.panel_conditional(
                                "input.path !== ''", ui.h5("Select frames"),
                                ui.input_slider("z_select", "", value=(0, 1), min=0, max=100), ),
                            ui.h5("Points of Interest"), ui.input_text("frames", "frames"), ui.input_text("pixel",
                                                                                                          "pixels")
                        ), ui.panel_main(
                            ui.panel_conditional(
                                "input.path !== ''", ui.output_plot("original")
                            )
                        )
                    )
                ), ui.nav(
                    "Delta", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.input_switch("use_delta", "calculate"),
                            ui.input_select("delta_method", "method", choices=["background", "dF", "dFF"],
                                            selected="dF"),
                            ui.input_numeric("delta_window", "window", 10),
                            ui.input_switch("delta_overwrite_first_frame", "Overwrite 1st frame"),
                            ui.h5("Plotting options"),
                            ui.input_switch("delta_plot_original", "Plot original", value=True),
                            ui.input_switch("delta_show_separate", "separate panels"),
                            ui.input_switch("delta_twinx", "Twin axis"),
                            ui.input_numeric("delta_alpha", "Original: alpha", value=0.7), ui.row(
                                ui.h6("Figsize"), ui.column(5, ui.input_numeric("delta_figsize_x", "", 20)),
                                ui.column(2, ui.h6(" x ")), ui.column(5, ui.input_numeric("delta_figsize_y", "", 5))
                            )
                        ), ui.panel_main(
                            ui.panel_conditional(
                                "input.use_delta", ui.output_plot("delta")
                            )
                        )
                    )
                ), ui.nav(
                    "Smoothing", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.input_switch("use_smoothing", "activate"),
                            ui.input_numeric("sigma", "Sigma", value=3),
                            ui.input_numeric("radius", "Radius", value=2), ), ui.panel_main(
                            ui.panel_conditional(
                                "input.use_smoothing", ui.output_plot("smooth")
                            )
                        )
                    )
                ), ui.nav(
                    "Thresholding", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.h3("Spatial Thresholding"), ui.input_switch("use_spatial", "activate"),
                            ui.input_numeric("min_ratio", "Min Ratio", value=1),
                            ui.input_numeric("z_depth", "Z-Depth", value=1), ui.h5(""), ui.h3("Temporal Thresholding"),
                            ui.input_numeric("prominence", "Prominence", value=10),
                            ui.input_numeric("width", "Width", value=3),
                            ui.input_numeric("rel_height", "Relative Height", value=0.9),
                            ui.input_numeric("wlen", "Wlen", value=60), ui.row(
                                ui.column(
                                    6, ui.h6("Time Series"), ui.input_switch("temp_show_trace", "activate"),
                                    ui.input_switch("temp_show_separate", "separate panels", value=True), ), ui.column(
                                    6, ui.h6("Frames"), ui.input_switch("use_temporal", "activate"), )
                            ), ui.panel_conditional(
                                "input.use_spatial & input.use_temporal", ui.h5(""),
                                ui.h5("Combine Spatial and Temporal"),
                                ui.input_radio_buttons("comb_type", "Union type", ["&", "|"]), )
                        ), ui.panel_main(
                            ui.panel_conditional(
                                "input.use_spatial", ui.h3("Spatial Thresholding"), ui.output_plot("spatial"), ),
                            ui.panel_conditional(
                                "input.temp_show_trace", ui.h3("Temporal Thresholding (timeseries)"),
                                ui.output_plot("temporal_timeseries"), ), ui.panel_conditional(
                                "input.use_temporal", ui.h3("Temporal Thresholding (images)"),
                                ui.output_plot("temporal"), ), ui.panel_conditional(
                                "input.use_spatial & input.use_temporal", ui.h4("Combined image"),
                                ui.output_plot("combined_"), )
                        )
                    )
                ), ui.nav(
                    "Morphology", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.h5("Fill Small Holes"), ui.input_switch("use_holes", "fill"),
                            ui.input_numeric("area_threshold", "Area Threshold", value=10),
                            ui.input_numeric("connectivity_holes", "Connectivity", value=1),
                            ui.input_numeric("holes_depth", "z_depth", value=1), ui.h5("Remove Small Objects"),
                            ui.input_switch("use_objects", "remove"),
                            ui.input_numeric("min_size", "Min Size", value=10),
                            ui.input_numeric("connectivity_objects", "Connectivity", value=1),
                            ui.input_numeric("objects_depth", "z_depth", value=1), ui.panel_conditional(
                                "input.use_holes & input.use_objects", ui.h5("Sequential Operation"),
                                ui.input_radio_buttons(
                                    "comb_options", "Combination order", ["None", "holes > objects", "objects > holes"]
                                ), )
                        ), ui.panel_main(
                            ui.panel_conditional(
                                "input.use_holes | input.use_objects", ui.h4("Morphed image"), ui.output_plot("morph")
                            ),

                        )
                    )
                ), ui.nav(
                    "Export", ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.h5(""), ui.h5("Additional parameters"),
                            ui.input_text("output_path", "Output path", value="infer"),
                            ui.input_numeric("exclude_border", "Border exclusion", value=0),
                            ui.input_switch("split_events", "Split events"),
                            ui.input_switch("overwrite", "Overwrite Output"),
                            ui.input_numeric("logging_level", "Logging level", value=20),
                            ui.input_switch("debug", "Debugging"), ui.h5(""), ui.h5("Export"),
                            ui.input_text("save_path", "Config path", value="./config.yaml"),
                            ui.input_action_button("btn_save", "Save arguments", class_="btn-primary"), ),
                        ui.panel_main(
                            ui.h4("Config file:"), ui.output_ui("argument_file"), ui.h4("Run detection with:"),
                            ui.output_ui("run_command"), ui.h4("Visualize results with:"),
                            ui.output_ui("visualize_command")
                        )
                    )
                )
            )
        )

    def server(self, input, output, session):

        @reactive.Calc
        def load_data():

            path = Path(input.path())

            if path.is_file():
                io = IO()

                try:

                    data = io.load(path, loc=input.loc(), z_slice=input.z_select(), lazy=input.lazy)

                    if not isinstance(data, da.Array):
                        data = da.from_array(data)

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
                data = io.load(path, loc=input.loc(), lazy=True)
                return data.shape

            except:
                return None

        @reactive.Calc
        def get_pixel():

            color_palette = sns.color_palette("husl", 8)
            color_counter = 0

            px_str = input.pixel().replace(" ", "")
            px_str = px_str.split(";")

            pixels = []
            colors = []
            for px in px_str:

                px = px.split(",")

                if len(px) not in (2, 3):
                    ui.notification_show(f"incorrect pixel dimension: {len(px)}", type="error")
                    continue

                pixels.append((int(px[0]), int(px[1])))

                if len(px) == 3:
                    colors.append(px[2])
                else:

                    if color_counter > 7:
                        color_counter = 0

                    colors.append(color_palette[color_counter])
                    color_counter += 1

            return pixels, colors

        @reactive.Calc
        def get_pixel_traces():

            pixels, _ = get_pixel()
            data = load_data()

            Z, X, Y = data.shape
            small_data = np.zeros((Z, len(pixels), 1), dtype=data.dtype)
            for i, (px, py) in enumerate(pixels):
                small_data[:, i, 0] = data[:, px, py]

            if isinstance(small_data, da.Array):
                small_data = small_data.compute()

            return small_data

        @reactive.Calc
        def get_delta():

            data = load_data()

            deltaObj = Delta(data=data, loc="")
            result = deltaObj.run(
                method=input.delta_method(), window=input.delta_window(), chunks=(-1, 1, 1),
                overwrite_first_frame=input.delta_overwrite_first_frame()
            )

            return result  # .compute()

        @reactive.Calc
        def get_delta_obj():
            small_data = get_pixel_traces()

            deltaObj = Delta(data=small_data, loc="")
            deltaObj.run(
                method=input.delta_method(), window=input.delta_window(),
                overwrite_first_frame=input.delta_overwrite_first_frame(),
                compute=True
            )

            return deltaObj

        @reactive.Calc
        def get_smooth():
            data = load_data()
            smooth_ = detection.Detector._gaussian_smooth_3d(data, sigma=input.sigma(), radius=input.radius())

            if not input.lazy():
                smooth_ = smooth_.compute()
                smooth_ = da.from_array(smooth_, chunks=("auto", "auto", "auto"))

            return smooth_

        @reactive.Calc
        def get_spatial():

            with ui.Progress(max=2) as p:
                p.set(0, message="Loading data", detail="")

                if input.use_smoothing():
                    data = get_smooth()
                else:
                    data = load_data()

                p.set(1, message="Spatial thresholding.", detail="This may take a while ...")
                spatial = detection.Detector._spatial_threshold(
                    data, min_ratio=input.min_ratio(), threshold_z_depth=input.z_depth()
                )

                p.set(2, message="Done")

            return spatial

        @reactive.Calc
        def get_temporal():

            with ui.Progress(max=2) as p:
                p.set(0, message="Loading data", detail="")

                if input.use_smoothing():
                    data = get_smooth()
                else:
                    data = load_data()

                p.set(1, message="Temporal thresholding.", detail="This may take a while ...")
                temporal = detection.Detector._temporal_threshold(
                    data, prominence=input.prominence(), width=input.width(), rel_height=input.rel_height(),
                    wlen=input.wlen()
                )

                p.set(2, message="Done")

                return temporal

        @reactive.Calc
        def get_union():

            if not (input.use_spatial and input.use_temporal):
                return None

            with ui.Progress(max=12) as p:
                i = 0
                p.set(i, message="Loading data", detail="")

                spatial = get_spatial()
                temporal = get_temporal()

                if input.comb_type() == "&":
                    arr = np.minimum(spatial, temporal)
                else:
                    arr = np.maximum(spatial, temporal)

                return arr, f"SPATIAL {input.comb_type()} TEMPORAL"

        @reactive.Calc
        def get_morph():

            with ui.Progress(max=3) as p:
                i = 0
                p.set(i, message="Loading data", detail="")

                data = []
                lbls = []

                if input.use_spatial() and not input.use_temporal():
                    data.append(get_spatial())
                    lbls.append("SPATIAL")

                elif not input.use_spatial() and input.use_temporal():
                    data.append(get_temporal())
                    lbls.append("TEMPORAL")

                else:
                    dat, lbl = get_union()
                    data.append(dat)
                    lbls.append(lbl)

                res = []
                res_lbls = []
                for dat, lbl in list(zip(data, lbls)):

                    if input.comb_options() == "None":

                        if input.use_holes():
                            filled = detection.Detector._fill_holes(
                                dat, area_threshold=input.area_threshold(), connectivity=input.connectivity_holes(),
                                depth=input.holes_depth()
                            )
                            res.append(filled)
                            res_lbls.append(lbl + "_fill")

                        if input.use_objects():
                            rem = detection.Detector._remove_objects(
                                dat, min_size=input.min_size(), connectivity=input.connectivity_holes(),
                                depth=input.objects_depth()
                            )
                            res.append(rem)
                            res_lbls.append(lbl + "_rem")

                    elif input.comb_options() == "holes > objects":

                        filled = detection.Detector._fill_holes(
                            dat, area_threshold=input.area_threshold(), connectivity=input.connectivity_holes(),
                            depth=input.objects_depth()
                        )
                        rem = detection.Detector._remove_objects(
                            filled, min_size=input.min_size(), connectivity=input.connectivity_holes(),
                            depth=input.holes_depth()
                        )

                        res.append(rem)
                        res_lbls.append(lbl + "_fill_rem")

                    elif input.comb_options() == "objects > holes":

                        rem = detection.Detector._remove_objects(
                            dat, min_size=input.min_size(), connectivity=input.connectivity_holes(),
                            depth=input.objects_depth()
                        )

                        filled = detection.Detector._fill_holes(
                            rem, area_threshold=input.area_threshold(), connectivity=input.connectivity_holes(),
                            depth=input.holes_depth()
                        )

                        res.append(filled)
                        res_lbls.append(lbl + "_rem_fill")

                return res, res_lbls

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
            """ Get raw data prior to analysis """

            data = load_data()
            frames = get_frames()

            pixel, _ = get_pixel()
            if pixel == 1:
                pixel = None

            return self.plot_images([data], frames, lbls=["INPUT"], pixels=pixel)

        @output
        @render.plot
        def delta():

            delta_obj = get_delta_obj()
            abs_pixels, colors = get_pixel()

            if isinstance(abs_pixels, tuple):
                abs_pixels = [abs_pixels]

            rel_pixels = [(i, 0) for i in range(len(abs_pixels))]
            labels = [f"pixel {x}x{y}" for x, y in abs_pixels]

            return delta_obj.plot(pixels=rel_pixels, show_original=input.delta_plot_original(), colors=colors,
                                  separate_panels=input.delta_show_separate(), alpha=input.delta_alpha(),
                                  twin_y_axis=input.delta_twinx(), labels=labels,
                                  figsize=(input.delta_figsize_x(), input.delta_figsize_y()))

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
        def temporal_timeseries():

            if input.temp_show_trace():

                pixels, colors = get_pixel()

                if input.use_smoothing():
                    data = get_smooth()
                else:
                    data = load_data()

                traces = np.zeros((data.shape[0], len(pixels), 1))
                for i, (x, y) in enumerate(pixels):
                    traces[:, i, 0] = data[:, x, y]

                mask = detection.Detector._temporal_threshold(
                    traces, prominence=input.prominence(), width=input.width(), rel_height=input.rel_height(),
                    wlen=input.wlen()
                )

                if input.temp_show_separate():
                    fig, axx = plt.subplots(len(pixels), figsize=(5 * len(pixels), 20))
                else:
                    fig, axx = plt.subplots(figsize=(5 * len(pixels), 20))
                    axx = [axx for _ in range(len(pixels))]

                for i in range(len(pixels)):
                    x, y = pixels[i]

                    # Get indices where mask is 0 for background, and 1 for signal
                    indices_background = np.where(mask[:, i, 0] == 0)[0]
                    indices_signal = np.where(mask[:, i, 0] == 1)[0]

                    if isinstance(indices_background, da.Array):
                        indices_background = indices_background.compute()

                    if isinstance(indices_signal, da.Array):
                        indices_signal = indices_signal.compute()

                    # Create the x-axis data range
                    X = np.arange(data.shape[0])  # Use np.arange for better performance with NumPy

                    # Get y-values for signal and background
                    signal_y = data[indices_signal, x, y]
                    background_y = data[indices_background, x, y]

                    # Create contiguous sections for signal
                    contiguous_signal_sections = np.split(signal_y, np.where(np.diff(indices_signal) != 1)[0] + 1)
                    contiguous_signal_indices = np.split(indices_signal,
                                                         np.where(np.diff(indices_signal) != 1)[0] + 1)

                    # Plot each contiguous section for signal
                    lbl = f"pixel {x}x{y}"
                    for sec_indices, sec_y in zip(contiguous_signal_indices, contiguous_signal_sections):
                        sec_x = X[sec_indices]

                        axx[i].plot(sec_x, sec_y, linestyle="--", color=colors[i], label=lbl)
                        lbl = None

                    # Create contiguous sections for background
                    contiguous_background_sections = np.split(background_y,
                                                              np.where(np.diff(indices_background) != 1)[0] + 1)
                    contiguous_background_indices = np.split(indices_background,
                                                             np.where(np.diff(indices_background) != 1)[0] + 1)

                    # Plot each contiguous section for background
                    for bg_indices, bg_y in zip(contiguous_background_indices, contiguous_background_sections):
                        bg_x = X[bg_indices]
                        axx[i].plot(bg_x, bg_y, linestyle="-", color="gray")

                    axx[i].legend()

                return fig

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
        def combined_():

            if input.use_temporal() and input.use_spatial():

                frames = get_frames()
                arr, lbls = get_union()

                return self.plot_images(arr, frames, lbls=lbls)

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
        @render.text
        @reactive.event(input.btn_save)
        async def argument_file():

            save_path = Path(input.save_path())

            arguments = {"detect-events": {  # file params
                "loc": input.loc(),  # smoothing
                "use_smoothing": input.use_smoothing(), "smooth_sigma": input.sigma(), "smooth_radius": input.radius(),
                # Spatial
                "use_spatial": input.use_spatial(), "spatial_min_ratio": input.min_ratio(),
                "spatial_z_depth": input.z_depth(),  # Temporal
                "use_temporal": input.use_temporal(), "temporal_prominence": input.prominence(),
                "temporal_width": input.width(), "temporal_rel_height": input.rel_height(),
                "temporal_wlen": input.wlen(),  # Morphological operations
                "fill_holes": input.use_holes(), "area_threshold": input.area_threshold(),
                "holes_connectivity": input.connectivity_holes(), "holes_depth": input.holes_depth(),
                "remove_objects": input.use_objects(), "min_size": input.min_size(),
                "object_connectivity": input.connectivity_objects(), "objects_depth": input.objects_depth(),
                "fill_holes_first": True if input.comb_options() == "holes > objects" else False,
                "comb_type": str(input.comb_type()),  # additional
                "output_path": input.output_path(), "exclude_border": input.exclude_border(),
                "split_events": input.split_events(), "overwrite": input.overwrite(),
                "logging_level": input.logging_level(), "debug": input.debug(), }}

            with open(save_path.as_posix(), 'w') as f:
                yaml.dump(arguments, f)

            ui.notification_show(f"Saved to: {save_path}")

            # load
            with open(save_path.as_posix()) as file:
                config = yaml.safe_load(file)

            # create string
            config_string = ""
            for section, section_data in config.items():
                print("\n")
                formatted_section_data = yaml.dump({section: section_data}, indent=4, default_flow_style=True)
                config_string += formatted_section_data

            return config_string

        @output
        @render.text
        @reactive.event(input.btn_save)
        async def run_command():
            return f"astrocast --config {input.save_path()} detect-events {input.path()}"

        @output
        @render.text
        @reactive.event(input.btn_save)
        async def visualize_command():
            return f"astrocast view-detection-results " \
                   f"--lazy False --h5-loc {input.loc()} {input.path().replace('.h5', '.roi')}"

    @staticmethod
    def plot_images(arr, frames, pixels=None, lbls=None, figsize=(10, 5), vmin=None, vmax=None):

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

                if pixels is not None:
                    for py, px in pixels:
                        ax.scatter(px, py, color="red", alpha=0.5)

        return fig

    def run(self, port=8000):
        self.app.run(port=port)


# Run the app
if __name__ == '__main__':
    app_instance = Explorer()
    app_instance.run()
