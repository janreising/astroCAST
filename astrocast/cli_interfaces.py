import datetime as dt
import logging
import os
import shutil
import time
from pathlib import Path
import click
import humanize
import yaml
from functools import partial
import numpy as np

from colorama import init as init_colorama
from colorama import Fore
import inspect
from prettytable import PrettyTable

init_colorama(autoreset=True)

click_custom_option = partial(click.option, show_default=True)


def check_output(output_path, input_path, h5_loc_save, overwrite):
    if output_path is None:
        logging.warning(f"No output_path provided. Assuming input_path: {input_path}")
        output_path = input_path

    output_path = Path(output_path)
    input_path = Path(input_path)

    if output_path.name.startswith("."):
        output_path = input_path.with_suffix(output_path.name)
        logging.warning(f"Output path inferred as: {output_path}")

    if output_path.exists():

        if output_path.suffix in (".hdf5", ".h5"):

            import h5py

            with h5py.File(output_path.as_posix(), "a") as f:
                if h5_loc_save in f and not overwrite:
                    logging.error(f"{h5_loc_save} already exists in {output_path}. "
                                  f"Please choose a different output location or use '--overwrite True'"
                                  )
                    return 0

                elif h5_loc_save in f:
                    logging.warning(f"{h5_loc_save} already exists in {output_path}. Deleting previous output.")
                    del f[h5_loc_save]

        else:

            if overwrite:
                logging.warning(f"{h5_loc_save} already exists in {output_path}. Deleting previous output.")
                output_path.unlink()

            else:
                logging.error(f"file already exists {output_path}. Please choose a different output location "
                              f"or use '--overwrite True'."
                              )
                return 0

    return output_path


def parse_chunks(chunks):
    if chunks is None or chunks == "infer":
        return chunks

    else:
        chunks = tuple(int(c) for c in chunks.split(","))
        if len(chunks) != 3:
            raise ValueError(
                f"please provide 'chunks' parameter as None, infer or comma-separated list of 3 int values."
                )
        return chunks


class UserFeedback:

    def __init__(
            self, params=None, logging_level=logging.WARNING, max_value_len=25,
            box_color=Fore.BLUE, msg_color=Fore.GREEN, table_color=Fore.CYAN
            ):

        logging.basicConfig(level=logging_level)

        self.t0 = None
        self.params = params
        self.max_value_len = max_value_len
        self.box_color = box_color
        self.msg_color = msg_color
        self.table_color = table_color

    def _collect_parameters(self):
        if self.params:

            params = self.params.copy()
            if 'feedback' in params:
                del params['feedback']

            table = PrettyTable()
            table.field_names = ["Parameter", "Value"]
            v_len = self.max_value_len  # Maximum length of the value
            overridden = False

            # Find the frame where ctx is defined
            for frame_info in inspect.stack():
                frame = frame_info.frame
                if '_Context__self' in frame.f_locals:
                    ctx = frame.f_locals['_Context__self']
                    break
            else:
                ctx = None

            default_map = ctx.default_map if ctx else {}

            for key, value in params.items():
                str_value = str(value)
                if key in default_map and default_map[key] != value:
                    overridden = True
                    str_value += " *"

                if len(str_value) > v_len:
                    str_value = "..." + str_value[-v_len:]

                table.add_row([key, str_value])

            table_str = f"\n{self.table_color}{table.get_string()}"
            print(table_str)

            if overridden:
                print(self.table_color + "  * config value was replaced by user input\n")

    def start(self, level=1):
        module_name = inspect.stack()[level].function
        module_name = module_name.replace("_", " ")

        print(self.box_color + "┌─" + "─" * len(module_name) + "─┐")
        print(self.box_color + "│ " + module_name + " │")
        print(self.box_color + "└─" + "─" * len(module_name) + "─┘")
        print(self.msg_color + "Starting module: " + module_name)

        self._collect_parameters()
        self.t0 = time.time()

    def end(self, level=1):
        module_name = inspect.stack()[level].function
        module_name = module_name.replace("_", " ")

        delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - self.t0))
        print(f"{self.msg_color}Completed module: {module_name} (runtime: {delta})")
        print()

    def __enter__(self):
        self.start(level=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end(level=2)


@click.group(context_settings={'auto_envvar_prefix': 'CLI'}, chain=True)
@click.option('--config', default=None, type=click.Path())  # this allows us to change config path
@click.pass_context
def cli(ctx, config):
    """Command Line Interface for the astroCAST package."""

    if config is not None:
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

        ctx.default_map = config


@cli.command()
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--output-path', type=click.Path(),
                     help='Path to save the processed data. If None, the processed data is returned.'
                     )
@click_custom_option('--sep', default="_", help='Separator used for sorting file names.')
@click_custom_option('--channels', default=1, help='Number of channels or dictionary specifying channel names.')
@click_custom_option('--z-slice', default=None, help='Z slice index.')
@click_custom_option('--lazy', is_flag=True, help='Lazy loading flag.')
@click_custom_option('--subtract-background', default=None, help='Background subtraction parameter.')
@click_custom_option('--subtract-func', default="mean", help='Function to use for background subtraction.')
@click_custom_option('--rescale', default=None, help='Rescale parameter.')
@click_custom_option('--dtype', default=np.uint, help='Data type to convert the processed data.')
@click_custom_option('--in-memory', is_flag=True, help='If True, the processed data is loaded into memory.')
@click_custom_option('--h5-loc', default="data", help='Prefix to use when saving the processed data.')
@click_custom_option('--chunks', type=click.STRING, default="infer",
                     help='Chunk size to use when saving to HDF5 or TileDB.'
                     )
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option('--overwrite', type=click.BOOL, default=False,
                     help='Flag for overwriting previous result in output location'
                     )
def convert_input(
        input_path, logging_level, output_path, sep, channels, z_slice, lazy, subtract_background,
        subtract_func, rescale, dtype, in_memory, h5_loc, chunks, compression, overwrite
        ):
    """
    Convert user files to astroCAST compatible format using the Input class.
    """

    from astrocast.preparation import Input

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, h5_loc, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # check chunks
        chunks = parse_chunks(chunks)

        # convert input
        input_instance = Input(logging_level=logging_level)
        input_instance.run(input_path=input_path, output_path=output_path, sep=sep, channels=channels, z_slice=z_slice,
                           lazy=lazy, subtract_background=subtract_background, subtract_func=subtract_func,
                           rescale=rescale,
                           dtype=dtype, in_memory=in_memory, h5_loc=h5_loc, chunks=chunks, compression=compression
                           )


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--working-directory', type=click.Path(), default=None,
                     help='Working directory for temporary files.'
                     )
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--h5-loc', type=click.STRING, default="",
                     help='Dataset name in case of input being an HDF5 file.'
                     )
@click_custom_option('--max-shifts', type=click.Tuple([int, int]), default=(50, 50),
                     help='Maximum allowed rigid shift.'
                     )
@click_custom_option('--niter-rig', type=click.INT, default=3,
                     help='Maximum number of iterations for rigid motion correction.'
                     )
@click_custom_option('--splits-rig', type=click.INT, default=14,
                     help='Number of splits across time for parallelization during rigid motion correction.'
                     )
@click_custom_option('--num-splits-to-process-rig', type=click.INT, default=None,
                     help='Number of splits to process during rigid motion correction.'
                     )
@click_custom_option('--strides', type=click.Tuple([int, int]), default=(48, 48),
                     help='Intervals at which patches are laid out for motion correction.'
                     )
@click_custom_option('--overlaps', type=click.Tuple([int, int]), default=(24, 24),
                     help='Overlap between patches (size of patch strides+overlaps).'
                     )
@click_custom_option('--pw-rigid', type=click.BOOL, default=False,
                     help='Flag for performing motion correction when calling motion_correct.'
                     )
@click_custom_option('--splits-els', type=click.INT, default=14,
                     help='Number of splits across time for parallelization during elastic motion correction.'
                     )
@click_custom_option('--num-splits-to-process-els', type=click.INT, default=None,
                     help='Number of splits to process during elastic motion correction.'
                     )
@click_custom_option('--upsample-factor-grid', type=click.INT, default=4,
                     help='Upsample factor of shifts per patches to avoid smearing when merging patches.'
                     )
@click_custom_option('--max-deviation-rigid', type=click.INT, default=3,
                     help='Maximum deviation allowed for patch with respect to rigid shift.'
                     )
@click_custom_option('--nonneg-movie', type=click.BOOL, default=True,
                     help='Make the saved movie and template mostly nonnegative by removing min_mov from movie.'
                     )
@click_custom_option('--gsig-filt', type=click.Tuple([int, int]), default=(20, 20),
                     help='Tuple indicating the size of the filter.'
                     )
@click_custom_option('--h5-loc-save', type=click.STRING, default="mc",
                     help='Location within the HDF5 file to save the data.'
                     )
@click_custom_option('--chunks', type=click.STRING, default=None,
                     help='Chunk shape for creating a dask array when saving to an HDF5 file.'
                     )
@click_custom_option('--compression', type=click.STRING, default=None,
                     help='Compression algorithm to use when saving to an HDF5 file.'
                     )
@click_custom_option('--overwrite', type=click.BOOL, default=False,
                     help='Flag for overwriting previous result in output location'
                     )
def motion_correction(
        input_path, working_directory, logging_level, output_path, h5_loc,
        max_shifts, niter_rig, splits_rig, num_splits_to_process_rig, strides,
        overlaps, pw_rigid, splits_els, num_splits_to_process_els, upsample_factor_grid,
        max_deviation_rigid, nonneg_movie, gsig_filt, h5_loc_save, chunks, compression, overwrite
        ):
    """
    Correct motion artifacts of input data using the MotionCorrection class.
    """

    from astrocast.preparation import MotionCorrection

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, h5_loc_save, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # check chunks
        chunks = parse_chunks(chunks)

        # Initialize the MotionCorrection instance
        logging.info("creating motion correction instance ...")
        mc = MotionCorrection(working_directory=working_directory, logging_level=logging_level)

        # Call the run method with the necessary parameters
        logging.info("applying motion correction ...")
        mc.run(input_=input_path, h5_loc=h5_loc, max_shifts=max_shifts, niter_rig=niter_rig,
               splits_rig=splits_rig, num_splits_to_process_rig=num_splits_to_process_rig,
               strides=strides, overlaps=overlaps, pw_rigid=pw_rigid, splits_els=splits_els,
               num_splits_to_process_els=num_splits_to_process_els, upsample_factor_grid=upsample_factor_grid,
               max_deviation_rigid=max_deviation_rigid, nonneg_movie=nonneg_movie, gSig_filt=gsig_filt
               )

        # Save the results to the specified output path
        logging.info("saving result ...")
        mc.save(output_path, h5_loc=h5_loc_save, chunks=chunks, compression=compression)


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--window', type=click.INT, required=True, help='Size of the window for the minimum filter.')
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--h5-loc-in', type=click.STRING, default="",
                     help='Location of the data in the HDF5 file (if applicable).'
                     )
@click_custom_option('--method', type=click.Choice(['background', 'dF', 'dFF']), default='background',
                     help='Method to use for delta calculation.'
                     )
@click_custom_option('--processing-chunks', type=click.STRING, default="infer", help='Chunk size for data processing.')
@click_custom_option('--save-chunks', type=click.STRING, default="infer", help='Chunk size for saving.')
@click_custom_option('--overwrite-first-frame', type=click.BOOL, default=True,
                     help='Whether to overwrite the first frame with the second frame after delta calculation.'
                     )
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Flag for lazy data loading and computation.')
@click_custom_option('--h5-loc-out', type=click.STRING, default="dff",
                     help='Location within the HDF5 file to save the data.'
                     )
@click_custom_option('--compression', type=click.STRING, default=None,
                     help='Compression algorithm to use when saving to an HDF5 file.'
                     )
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--overwrite', type=click.BOOL, default=False,
                     help='Flag for overwriting previous result in output location'
                     )
def subtract_delta(
        input_path, output_path, h5_loc_in, method, window, processing_chunks, save_chunks, overwrite_first_frame, lazy, h5_loc_out,
        compression, logging_level, overwrite
        ):
    """
    Subtract baseline of input using the Delta class.
    """

    from astrocast.preparation import Delta

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, h5_loc_out, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # check chunks
        processing_chunks = parse_chunks(processing_chunks)
        save_chunks = parse_chunks(save_chunks)

        # Initialize the Delta instance
        logging.info("creating delta instance ...")
        delta_instance = Delta(input_=input_path, loc=h5_loc_in)

        # Run the delta calculation
        logging.info("subtracting background ...")
        result = delta_instance.run(method=method, window=window, processing_chunks=processing_chunks, output_path=None,
                                    overwrite_first_frame=overwrite_first_frame, lazy=lazy
                                    )

        # Save the results to the specified output path
        logging.info("saving result ...")
        delta_instance.save(output_path=output_path, h5_loc=h5_loc_out, chunks=save_chunks, compression=compression,
                            overwrite=overwrite
                            )


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--model', type=click.Path(), required=True,
                     help='Path to the trained model file or the model object itself.'
                     )
@click_custom_option('--output-file', type=click.Path(), required=True,
                     help='Path to the output file where the results will be saved. If not provided, the result will be returned instead of being saved to a file.'
                     )
@click_custom_option('--batch-size', type=click.INT, default=16, help='batch size processed in each step.')
@click_custom_option('--input-size', type=(int, int), default=(100, 100), help='size of the denoising window')
@click_custom_option('--pre-post-frame', type=click.INT, default=5,
                     help='Number of frames before and after the central frame in each data chunk.'
                     )
@click_custom_option('--gap-frames', type=click.INT, default=0,
                     help='Number of frames to skip in the middle of each data chunk.'
                     )
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None,
                     help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
                     )
@click_custom_option('--overlap', type=click.FLOAT, default=None, help='Overlap between data chunks.')
@click_custom_option('--padding', type=click.STRING, default=None, help='Padding mode for the data chunks.')
@click_custom_option('--normalize', type=click.STRING, default=None, help='Normalization mode for the data.')
@click_custom_option('--loc', type=click.STRING, default="data/",
                     help='Location in the input file(s) where the data is stored.'
                     )
@click_custom_option('--in-memory', type=click.BOOL, default=False, help='Whether to store data in memory.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--out-loc', type=click.STRING, default=None,
                     help='Location in the output file where the results will be saved.'
                     )
@click_custom_option('--dtype', type=click.STRING, default="same",
                     help='Data type for the output. If "same", the data type of the input will be used.'
                     )
@click_custom_option('--chunk-size', type=(int, int), default=None,
                     help='Chunk size for saving the results in the output file. If not provided, a default chunk size will be used.'
                     )
@click_custom_option('--rescale', type=click.BOOL, default=True, help='Whether to rescale the output values.')
def denoise(
        input_file, batch_size, input_size, pre_post_frame, gap_frames, z_select,
        logging_level, model, output, out_loc, dtype, chunk_size, rescale,
        overlap, padding,
        normalize, loc, in_memory
        ):
    """
    Denoise the input data using the SubFrameGenerator class and infer method.
    """

    from astrocast.denoising import SubFrameGenerator

    with UserFeedback(params=locals(), logging_level=logging_level):
        # Initializing the SubFrameGenerator instance
        sub_frame_generator = SubFrameGenerator(
            paths=input_file,
            batch_size=batch_size,
            input_size=input_size,
            pre_post_frame=pre_post_frame,
            gap_frames=gap_frames,
            z_steps=None,
            z_select=z_select,
            allowed_rotation=[0],
            allowed_flip=[-1],
            random_offset=False,
            add_noise=False,
            drop_frame_probability=None,
            max_per_file=None,
            overlap=overlap,
            padding=padding,
            shuffle=False,
            normalize=normalize,
            loc=loc,
            output_size=None,
            cache_results=False,
            in_memory=in_memory,
            save_global_descriptive=False,
            logging_level=logging_level
        )

        # Running the infer method
        result = sub_frame_generator.infer(
            model=model,
            output=output,
            out_loc=out_loc,
            dtype=dtype,
            chunk_size=chunk_size,
            rescale=rescale
        )


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--h5-loc', type=click.STRING, default=None, help='Specific dataset to run the process on.')
@click_custom_option('--threshold', type=click.FLOAT, default=None,
                     help='Threshold value to discriminate background from events.'
                     )
@click_custom_option('--exclude-border', type=click.INT, default=0,
                     help='Number of pixels at the image border to exclude from the detection.'
                     )
@click_custom_option('--use-smoothing', type=click.BOOL, default=True, help='Whether to use smoothing.')
@click_custom_option('--smooth-radius', type=click.INT, default=2, help='Radius for smoothing kernel.')
@click_custom_option('--smooth-sigma', type=click.INT, default=2, help='Sigma value for smoothing kernel.')
@click_custom_option('--use-spatial', type=click.BOOL, default=True,
                     help='Whether to use spatial considerations in the algorithm.'
                     )
@click_custom_option('--spatial-min-ratio', type=click.INT, default=1, help='Minimum ratio for spatial considerations.')
@click_custom_option('--spatial-z-depth', type=click.INT, default=1, help='Z-depth for spatial considerations.')
@click_custom_option('--use-temporal', type=click.BOOL, default=True,
                     help='Whether to consider temporal elements in the algorithm.'
                     )
@click_custom_option('--temporal-prominence', type=click.INT, default=10,
                     help='Prominence value for temporal considerations.'
                     )
@click_custom_option('--temporal-width', type=click.INT, default=3, help='Width value for temporal considerations.')
@click_custom_option('--temporal-rel-height', type=click.FLOAT, default=0.9,
                     help='Relative height for temporal considerations.'
                     )
@click_custom_option('--temporal-wlen', type=click.INT, default=60, help='Window length for temporal considerations.')
@click_custom_option('--temporal-plateau-size', type=click.INT, default=None,
                     help='Plateau size for temporal considerations.'
                     )
@click_custom_option('--comb-type', type=click.STRING, default='&',
                     help='Combination type for merging spatial and temporal results.'
                     )
@click_custom_option('--fill-holes', type=click.BOOL, default=True,
                     help='Whether to fill holes in the detected events.'
                     )
@click_custom_option('--area-threshold', type=click.INT, default=10, help='Area threshold for hole filling.')
@click_custom_option('--holes-connectivity', type=click.INT, default=1,
                     help='Connectivity consideration for hole filling.'
                     )
@click_custom_option('--holes-depth', type=click.INT, default=1, help='Depth consideration for hole filling.')
@click_custom_option('--remove-objects', type=click.BOOL, default=True,
                     help='Whether to remove objects during processing.'
                     )
@click_custom_option('--min-size', type=click.INT, default=20, help='Minimum size for object removal.')
@click_custom_option('--object-connectivity', type=click.INT, default=1,
                     help='Connectivity consideration for object removal.'
                     )
@click_custom_option('--objects-depth', type=click.INT, default=1, help='Depth consideration for object removal.')
@click_custom_option('--fill-holes-first', type=click.BOOL, default=True,
                     help='Whether to fill holes before other processing steps.'
                     )
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option('--adjust-for-noise', type=click.BOOL, default=False,
                     help='Whether to adjust event detection for background noise.'
                     )
@click_custom_option('--subset', type=click.STRING, default=None, help='Subset of the dataset to process.')
@click_custom_option('--split-events', type=click.BOOL, default=False,
                     help='Whether to split detected events into smaller events if multiple peaks are detected.'
                     )
@click_custom_option('--debug', type=click.BOOL, default=False, help='Enable debug mode.')
@click_custom_option('--parallel', type=click.BOOL, default=True, help='Enable parallel execution.')
@click_custom_option('--output-path', type=click.Path(), default=None, help='Path to the output file.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--overwrite', type=click.BOOL, default=False,
                     help='Flag for overwriting previous result in output location'
                     )
def detect_events(
        input_path, h5_loc, exclude_border, threshold, use_smoothing, smooth_radius, smooth_sigma, use_spatial, spatial_min_ratio,
        spatial_z_depth, use_temporal, temporal_prominence, temporal_width, temporal_rel_height, temporal_wlen,
        temporal_plateau_size, comb_type, fill_holes, area_threshold, holes_connectivity, holes_depth, remove_objects,
        min_size, object_connectivity, objects_depth, fill_holes_first, lazy, adjust_for_noise, subset, split_events,
        debug, parallel, output_path, logging_level, overwrite
):
    """
    Detect events using the Detector class.
    """

    from astrocast.detection import Detector

    with UserFeedback(params=locals(), logging_level=logging_level):

        if output_path == "infer":
            output_path = Path(input_path)
            output_path = output_path.with_suffix(".roi")

        # check output
        if output_path is not None and Path(output_path).exists():

            if overwrite:
                logging.warning(f"overwrite is {overwrite}, deleting previous result")
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Aborting detection because previous calculation exists ({output_path}."
                                      f"Please provide an alternative output path or set '--overwrite True'"
                                      )

        logging.warning(f"input: {input_path}")

        # Initializing the Detector instance
        detector = Detector(input_path=input_path, output=output_path, logging_level=logging_level)

        # Running the detection
        detector.run(h5_loc=h5_loc, exclude_border=exclude_border, threshold=threshold,
                     use_smoothing=use_smoothing, smooth_radius=smooth_radius, smooth_sigma=smooth_sigma,
                     use_spatial=use_spatial, spatial_min_ratio=spatial_min_ratio, spatial_z_depth=spatial_z_depth,
                     use_temporal=use_temporal, temporal_prominence=temporal_prominence, temporal_width=temporal_width,
                     temporal_rel_height=temporal_rel_height, temporal_wlen=temporal_wlen,
                     temporal_plateau_size=temporal_plateau_size,
                     comb_type=comb_type,
                     fill_holes=fill_holes, area_threshold=area_threshold, holes_connectivity=holes_connectivity,
                     holes_depth=holes_depth,
                     remove_objects=remove_objects, min_size=min_size, object_connectivity=object_connectivity,
                     objects_depth=objects_depth,
                     fill_holes_first=fill_holes_first,
                     lazy=lazy, adjust_for_noise=adjust_for_noise,
                     subset=subset, split_events=split_events,
                     debug=debug,
                     parallel=parallel
                     )


def visualize_h5_recursive(loc, indent='', prefix=''):
    """Recursive part of the function to visualize the structure."""

    import h5py

    items = list(loc.items())
    for i, (name, item) in enumerate(items):
        is_last = i == len(items) - 1
        new_prefix = '│  ' if not is_last else '   '

        if isinstance(item, h5py.Group):
            print(f"{indent}{prefix}├─ {name}/")
            visualize_h5_recursive(item, indent + new_prefix, prefix='├─ ')

        elif isinstance(item, h5py.Dataset):
            details = [
                f"shape: {item.shape}",
                f"dtype: {item.dtype}",
            ]
            if item.compression:
                details.append(f"compression: {item.compression}")
            if item.chunks:
                details.append(f"chunks: {item.chunks}")

            details_str = ', '.join(details)
            print(f"{indent}{prefix}├─ {name} ({details_str})")


@cli.command()
@click.argument('input-path', type=click.Path())
def visualize_h5(input_path):
    """
    Visualizes the structure of a .h5 file in a tree format.

    This function uses recursion to traverse through all groups and datasets in the .h5 file and
    prints the structure in a pretty way. It can be used to quickly inspect the contents of a .h5 file.

    Parameters:
    input_path (str): The path to the .h5 file that needs to be visualized.

    Returns:
    None

    Example:
    visualize_h5('path/to/your/file.h5')
    """

    file_size = humanize.naturalsize(os.path.getsize(input_path))
    print(f"\n> {os.path.basename(input_path)} ({file_size})")

    import h5py

    with h5py.File(input_path, 'r') as f:
        visualize_h5_recursive(f['/'])


@cli.command()
@click.argument('input-path', type=click.Path())
@click.argument('h5-loc', nargs=-1)
@click_custom_option('--colormap', type=click.STRING, default="gray", help='Color of the video layer.')
@click_custom_option('--show-trace', type=click.BOOL, default=False, help='Display trace of the video')
@click_custom_option('--window', type=click.INT, default=160, help='window of trace to be shown')
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None,
                     help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
                     )
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
def view_data(input_path, h5_loc, z_select, lazy, show_trace, window, colormap):
    """
    Displays a video from a data file (.h5, .tiff, .tdb).

    This function uses the Video class to create a video object from a dataset  and displays the video using napari.
    The function provides options to select a specific dataset within the h5 file, a range of frames to display,
    and whether to use lazy loading.

    Parameters:
    input_path (str): The path to the h5 file.
    h5_loc (str): The name or identifier of the dataset within the h5 file. Defaults to an empty string, which indicates the root group.
    z_select (tuple of int, optional): A tuple specifying the range of frames to select in the Z dimension. The tuple contains two elements: the start and end frame numbers. Defaults to None, which indicates that all frames should be selected.
    lazy (bool): Whether to implement lazy loading, which can improve performance when working with large datasets by only loading data into memory as it is needed. Defaults to True.

    Returns:
    None

    Examples:
    view_data('path/to/your/file.h5', h5_loc='dataset_name', z_select=(10, 20), lazy=True)
    """

    import napari
    from astrocast.analysis import Video

    vid = Video(data=input_path, z_slice=z_select, h5_loc=h5_loc, lazy=lazy)
    vid.show(show_trace=show_trace, window=window, colormap=colormap)
    napari.run()


@cli.command()
@click.argument('event_dir', type=click.Path())
@click_custom_option('--video-path', type=click.STRING, default="infer", help='Path to the data used for detection.')
@click_custom_option('--h5-loc', type=click.STRING, default="",
                     help='Name or identifier of the dataset used for detection.'
                     )
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None,
                     help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
                     )
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
def view_detection_results(event_dir, video_path, h5_loc, z_select, lazy):
    """
    view the detection results; optionally overlayed on the input video.

    Parameters:
    event_dir (str): The path to the directory where the event data is stored. This path must exist.
    video_path (str, optional): The path to the data used for detection. If "infer", the path will be inferred. Defaults to "infer".
    h5_loc (str, optional): The name or identifier of the dataset used for detection within the HDF5 file. Defaults to an empty string.
    z_select (tuple of int, optional): The range of frames to select in the Z dimension, specified as a tuple of start and end frame indices. Defaults to None, indicating that all frames will be selected.
    lazy (bool, optional): Indicates whether to implement lazy loading, which defers data loading until necessary, potentially saving memory. Defaults to True.

    Returns:
    None: The function initiates a Napari viewer instance to visualize the detection results but does not return any value.

    Usage:
    To use this command, specify the necessary parameters as described above. For example:
    $ astrocast -view-detection-results --lazy False /path/to/event_dir

    """

    import napari
    from astrocast.analysis import Events

    event = Events(event_dir=event_dir, data=video_path, h5_loc=h5_loc, z_slice=z_select, lazy=lazy)
    viewer = event.show_event_map(video=None, h5_loc=None, z_slice=z_select)
    viewer.show()
    napari.run()


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=click.Path(), required=True, help='Path to the output file.')
@click_custom_option('--h5-loc-in', type=click.STRING, default="",
                     help='Name or identifier of the dataset in the h5 file.'
                     )
@click_custom_option('--h5-loc-out', type=click.STRING, default="",
                     help='Name or identifier of the dataset in the h5 file.'
                     )
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None,
                     help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
                     )
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option('--chunk-size', type=(int, int), default=None,
                     help='Chunk size for saving the results in the output file. If not provided, a default chunk size will be used.'
                     )
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option('--overwrite', type=click.BOOL, default=False,
                     help='Flag for overwriting previous result in output location'
                     )
def export_video(input_path, output_path, h5_loc_in, h5_loc_out, z_select, lazy, chunk_size, compression, overwrite):
    """
    Exports a video dataset from the input file to another file with various configurable options.

    This function uses the IO class to load a dataset from an input h5 file and save it to another output file.
    The dataset can be identified using the h5 location in both input and output files.
    The function allows for various configurations including lazy loading, chunk size specification for saving,
    and option to select a specific frame range in the Z dimension. It also allows for data compression and
    overwriting existing data in the output location.

    Parameters:
    input_path (str): The path to the input h5 file containing the video dataset to export.
    output_path (str): The path where the output file will be saved.
    h5_loc_in (str, optional): The name or identifier of the dataset within the input h5 file. Defaults to an empty string, which indicates the root group.
    h5_loc_out (str, optional): The name or identifier of the dataset within the output file. Defaults to an empty string, which indicates the root group.
    z_select (tuple of int, optional): A tuple specifying the range of frames to select in the Z dimension. The tuple contains two elements: the start and end frame numbers. Defaults to None, which indicates that all frames should be selected.
    lazy (bool, optional): Whether to implement lazy loading, which can improve performance when working with large datasets by only loading data into memory as it is needed. Defaults to True.
    chunk_size (tuple of int, optional): A tuple specifying the chunk size for saving the results in the output file. If not provided, a default chunk size will be used. Defaults to None.
    compression (str, optional): The compression method to use when saving data to the output file. If not provided, no compression is applied. Defaults to None.
    overwrite (bool, optional): Whether to overwrite previous results in the output location if they exist. Defaults to False.

    Returns:
    None

    Example:
    export_video('input.h5', 'output.h5', h5_loc_in='dataset1', h5_loc_out='dataset2', z_select=(10, 20), lazy=True, chunk_size=(100, 100), compression='gzip', overwrite=True)
    """

    if Path(output_path).exists():
        logging.error(f"file already exists {output_path}. Please choose a different output location "
                      f"or use '--overwrite True'."
                      )

        return 0

    from astrocast.preparation import IO

    io = IO()
    data = io.load(input_path, h5_loc=h5_loc_in, z_slice=z_select, lazy=lazy)

    io.save(output_path, data=data, h5_loc=h5_loc_out, chunks=chunk_size, compression=compression, overwrite=overwrite)


@cli.command()
@click_custom_option('--input-path', type=click.Path(), default=None, help='Path to input file.')
@click_custom_option('--h5-loc', type=click.STRING, default=None,
                     help='Name or identifier of the dataset in the h5 file.'
                     )
def explorer(input_path, h5_loc):

    from astrocast.app_preparation import Explorer

    app_instance = Explorer(input_path=input_path, h5_loc=h5_loc)
    app_instance.run()

@cli.command()
@click_custom_option('--input-path', type=click.Path(), default=None, help='Path to event detection output (.roi).')
@click_custom_option('--video-path', type=click.Path(), default=None, help='Path to video file.')
@click_custom_option('--h5-loc', type=click.STRING, default="", help='dataset location for .h5 files')
@click_custom_option('--default-settings', type=dict, default={}, help='settings for app.')
def analysis(input_path, video_path, h5_loc, default_settings):

    """Run interactive analysis of events."""

    from astrocast.app_analysis import Analysis

    app_instance = Analysis(input_path=input_path, video_path=video_path,
                            h5_loc=h5_loc, default_settings=default_settings)
    app_instance.run()


if __name__ == '__main__':
    cli()
