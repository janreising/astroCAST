import ast
import datetime as dt
import inspect
import logging
import os
import shutil
import time
import warnings
from functools import partial
from pathlib import Path

import click
import humanize
import numpy as np
import yaml
from colorama import Fore
from colorama import init as init_colorama
from prettytable import PrettyTable

from astrocast.helper import is_docker
from astrocast.preparation import Input

init_colorama(autoreset=True)

click_custom_option = partial(click.option, show_default=True)


def check_output(output_path, input_path, loc_in, overwrite):
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
                if loc_in in f and not overwrite:
                    logging.error(
                        f"{loc_in} already exists in {output_path}. "
                        f"Please choose a different output location or use '--overwrite True'"
                    )
                    return 0

                elif loc_in in f:
                    logging.warning(f"{loc_in} already exists in {output_path}. Deleting previous output.")
                    del f[loc_in]

        else:

            if overwrite:
                logging.warning(f"{loc_in} already exists in {output_path}. Deleting previous output.")
                output_path.unlink()

            else:
                logging.error(
                    f"file already exists {output_path}. Please choose a different output location "
                    f"or use '--overwrite True'."
                )
                return 0

    return output_path


def parse_chunks(infer_chunks, chunks):
    if infer_chunks:

        if chunks is not None:
            logging.warning(f"inferring chunks, although 'chunks' parameter was provided.")

        chunks = "infer"

    elif chunks is None or isinstance(chunks, tuple):
        pass

    elif isinstance(chunks, str):
        chunks = tuple(int(c) for c in chunks.split(","))
        if len(chunks) != 3:
            raise ValueError(
                f"Provide 'chunks' parameter as None, infer or comma-separated list of 3 int values."
                f"Not: {type(chunks)}, {chunks}"
            )

    return chunks


class UserFeedback:

    def __init__(
            self, params=None, logging_level=logging.WARNING, max_value_len=25, box_color=Fore.BLUE,
            msg_color=Fore.GREEN, table_color=Fore.CYAN
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
            if default_map is not None:

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
        with open(config) as file:
            config = yaml.safe_load(file)

        ctx.default_map = config


@cli.command()
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option(
    '--output-path', type=click.Path(), help='Path to save the processed data. If None, the processed data is returned.'
)
@click_custom_option('--sep', default="_", help='Separator used for sorting file names.')
@click_custom_option('--num-channels', default=1, help='Number of channels.')
@click_custom_option('--channel-names', default=None, help='Channel names as comma separated list.')
@click_custom_option('--z-slice', type=(click.INT, click.INT), default=None, help='Z slice index.')
@click_custom_option('--lazy', is_flag=True, help='Lazy loading flag.')
@click_custom_option('--subtract-background', default=None, help='Background subtraction parameter.')
@click_custom_option('--subtract-func', default="mean", help='Function to use for background subtraction.')
@click_custom_option('--rescale', type=click.FLOAT, default=1.0, help='Rescale parameter.')
@click_custom_option('--in-memory', is_flag=True, help='If True, the processed data is loaded into memory.')
@click_custom_option('--loc-in', default=None, help='Prefix to use when loading the processed data.')
@click_custom_option('--loc-out', default=None, help='Prefix to use when saving the processed data.')
@click_custom_option(
    '--chunk-strategy', type=click.Choice(["None", 'balanced', 'XY', 'Z']), default="balanced",
    help='Infer chunks size.'
)
@click_custom_option(
    '--chunks', type=(click.INT, click.INT, click.INT), default=None,
    help='Chunk size to use when saving to HDF5 or TileDB.'
)
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option(
    '--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location'
)
def convert_input(
        input_path, logging_level, output_path, sep, num_channels, channel_names, z_slice, lazy, subtract_background,
        subtract_func, rescale, in_memory, loc_in, loc_out, chunks, compression, overwrite, chunk_strategy
):
    """
    Convert user files to astroCAST compatible format using the Input class.
    """

    from astrocast.preparation import Input

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, loc_out, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # define channel names
        if channel_names is None:

            if loc_out is not None and num_channels == 1:
                channel_names = [loc_out]
            elif loc_out is not None:
                if loc_out[-1] == "/":
                    channel_names = [f"{loc_out}ch{i}" for i in range(num_channels)]
                else:
                    channel_names = [f"{loc_out}/ch{i}" for i in range(num_channels)]
            else:
                channel_names = [f"io/ch{i}" for i in range(num_channels)]

        elif isinstance(channel_names, str):
            channel_names = channel_names.split(",")

        if len(channel_names) != num_channels:
            warnings.warn(
                f"Number of channels {num_channels} does not match channel names {channel_names}. Choosing default."
            )
            channel_names = [f"io/ch{i}" for i in range(num_channels)]

        channels = {i: n for i, n in enumerate(channel_names)}

        if chunk_strategy == "None":
            chunk_strategy = None
        logging.warning(f"chunks > {chunks}, chunk_strategy > {chunk_strategy}")

        # convert input
        input_instance = Input(logging_level=logging_level)
        input_instance.run(
            input_path=input_path, output_path=output_path, sep=sep, channels=channels, z_slice=z_slice, lazy=lazy,
            subtract_background=subtract_background, subtract_func=subtract_func, rescale=rescale, in_memory=in_memory,
            loc_in=loc_in, loc_out=loc_out, chunks=chunks, chunk_strategy=chunk_strategy, compression=compression
        )


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option(
    '--working-directory', type=click.Path(), default=None, help='Working directory for temporary files.'
)
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option(
    '--loc-in', type=click.STRING, default="", help='Dataset name in case of input being an HDF5 file.'
)
@click_custom_option(
    '--loc-out', type=click.STRING, default="mc/ch0", help='Location within the HDF5 file to save the data.'
)
@click_custom_option(
    '--max-shifts', type=(click.INT, click.INT), default=(50, 50), help='Maximum allowed rigid shift.'
)
@click_custom_option(
    '--niter-rig', type=click.INT, default=3, help='Maximum number of iterations for rigid motion correction.'
)
@click_custom_option(
    '--splits-rig', type=click.INT, default=14,
    help='Number of splits across time for parallelization during rigid motion correction.'
)
@click_custom_option(
    '--num-splits-to-process-rig', type=click.INT, default=None,
    help='Number of splits to process during rigid motion correction.'
)
@click_custom_option(
    '--strides', type=click.Tuple([int, int]), default=(48, 48),
    help='Intervals at which patches are laid out for motion correction.'
)
@click_custom_option(
    '--overlaps', type=click.Tuple([int, int]), default=(24, 24),
    help='Overlap between patches (size of patch strides+overlaps).'
)
@click_custom_option(
    '--pw-rigid', type=click.BOOL, default=False,
    help='Flag for performing motion correction when calling motion_correct.'
)
@click_custom_option(
    '--splits-els', type=click.INT, default=14,
    help='Number of splits across time for parallelization during elastic motion correction.'
)
@click_custom_option(
    '--num-splits-to-process-els', type=click.INT, default=None,
    help='Number of splits to process during elastic motion correction.'
)
@click_custom_option(
    '--upsample-factor-grid', type=click.INT, default=4,
    help='Upsample factor of shifts per patches to avoid smearing when merging patches.'
)
@click_custom_option(
    '--max-deviation-rigid', type=click.INT, default=3,
    help='Maximum deviation allowed for patch with respect to rigid shift.'
)
@click_custom_option(
    '--nonneg-movie', type=click.BOOL, default=True,
    help='Make the saved movie and template mostly nonnegative by removing min_mov from movie.'
)
@click_custom_option(
    '--gsig-filt', type=(click.INT, click.INT), default=(20, 20), help='Tuple indicating the size of the filter.'
)
@click_custom_option(
    '--chunk-strategy', type=click.STRING, default="balanced", help='Strategy to infer chunks: ["balanced", "Z", "XY"]'
)
@click_custom_option(
    '--chunks', type=(click.INT, click.INT, click.INT), default=None,
    help='Chunk shape for creating a dask array when saving to an HDF5 file.'
)
@click_custom_option(
    '--compression', type=click.STRING, default=None, help='Compression algorithm to use when saving to an HDF5 file.'
)
@click_custom_option(
    '--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location'
)
def motion_correction(
        input_path, working_directory, logging_level, output_path, loc_in, loc_out, max_shifts, niter_rig, splits_rig,
        num_splits_to_process_rig, strides, overlaps, pw_rigid, splits_els, num_splits_to_process_els,
        upsample_factor_grid, max_deviation_rigid, nonneg_movie, gsig_filt, chunk_strategy, chunks, compression,
        overwrite
):
    """
    Correct motion artifacts of input data using the MotionCorrection class.
    """

    from astrocast.preparation import MotionCorrection

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, loc_out, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # Initialize the MotionCorrection instance
        logging.info("creating motion correction instance ...")
        mc = MotionCorrection(working_directory=working_directory, logging_level=logging_level)

        # Call the run method with the necessary parameters
        logging.info("applying motion correction ...")
        mc.run(
            path=input_path, loc=loc_in, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
            num_splits_to_process_rig=num_splits_to_process_rig, strides=strides, overlaps=overlaps, pw_rigid=pw_rigid,
            splits_els=splits_els, num_splits_to_process_els=num_splits_to_process_els,
            upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
            nonneg_movie=nonneg_movie, gSig_filt=gsig_filt
        )

        # Save the results to the specified output path
        logging.info("saving result ...")
        mc.save(output_path, loc=loc_out, chunk_strategy=chunk_strategy, chunks=chunks, compression=compression)


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--window', type=click.INT, default=None, help='Size of the window for the minimum filter.')
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option(
    '--loc-in', type=click.STRING, default="", help='Location of the data in the HDF5 file (if applicable).'
)
@click_custom_option(
    '--method', type=click.Choice(['background', 'dF', 'dFF']), default='dF',
    help='Method to use for delta calculation.'
)
@click_custom_option(
    '--chunk-strategy', type=click.STRING, default="balanced", help='Strategy to infer chunks: ["balanced", "Z", "XY"]'
)
@click_custom_option(
    '--chunks', type=(click.INT, click.INT, click.INT), default=None, help='Manual definition of chunk size for saving.'
)
@click_custom_option(
    '--overwrite-first-frame', type=click.BOOL, default=True,
    help='Whether to overwrite the first frame with the second frame after delta calculation.'
)
@click_custom_option(
    '--loc-out', type=click.STRING, default="dff", help='Location within the HDF5 file to save the data.'
)
@click_custom_option(
    '--compression', type=click.STRING, default=None, help='Compression algorithm to use when saving to an HDF5 file.'
)
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option(
    '--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location'
)
def subtract_delta(
        input_path, output_path, loc_in, method, window, chunk_strategy, chunks, overwrite_first_frame, loc_out,
        compression, logging_level, overwrite
):
    """
    Subtract baseline of input using the Delta class.
    """

    if window is None:
        raise ValueError(f"please provide window argument")

    from astrocast.preparation import Delta

    with UserFeedback(params=locals(), logging_level=logging_level):
        # check output
        output_path = check_output(output_path, input_path, loc_out, overwrite)
        if output_path == 0:
            logging.warning("skipping this step because output exists.")
            return 0

        # Initialize the Delta instance
        logging.info("creating delta instance ...")
        delta_instance = Delta(data=input_path, loc=loc_in)

        # Run the delta calculation
        logging.info("subtracting background ...")
        result = delta_instance.run(method=method, window=window, overwrite_first_frame=overwrite_first_frame)

        # Save the results to the specified output path
        logging.info("saving result ...")
        delta_instance.save(
            output_path=output_path, loc=loc_out, chunk_strategy=chunk_strategy, chunks=chunks, compression=compression,
            overwrite=overwrite
        )


@cli.command()
@click_custom_option(
    '--training-files', type=click.STRING, required=True, default=None,
    help="Path to training file or a glob search string (eg. './train_*.h5')"
)
@click_custom_option(
    '--validation-files', type=click.STRING, default=None,
    help="Path to validation file or a glob search string (eg. './train_*.h5')"
)
@click_custom_option(
    '--loc', type=click.STRING, default=None, help="Dataset location if .h5 files are provided."
)
@click_custom_option(
    '--batch-size', type=click.INT, default=16, help="Batch size that is trained at once."
)
@click_custom_option(
    '--batch-size-val', type=click.INT, default=4, help="Batch size that is validated at once."
)
@click_custom_option(
    '--epochs', type=click.INT, default=10, help="Maximum number of epochs trained."
)
@click_custom_option(
    '--patience', type=click.INT, default=3, help="Maximum number of epochs without improvement before early stopping."
)
@click_custom_option(
    '--min-delta', type=click.INT, default=0.001, help="Minimum difference considered to be improvement."
)
@click_custom_option(
    '--input-size', type=(click.INT, click.INT), default=(256, 256),
    help="Field of view (FoV) that is denoised at each iteration."
)
@click_custom_option(
    '--train-rotation', type=(click.INT, click.INT, click.INT), default=(1, 2, 3),
    help="Allowed rotations of training data."
)
@click_custom_option(
    '--train-flip', type=(click.INT, click.INT), default=(0, 1), help="Allowed mirroring axis of training data"
)
@click_custom_option(
    '--pre-post-frames', type=click.INT, default=5,
    help="Number of frames before and after reconstruction image that are provided for interpolation."
)
@click_custom_option(
    '--gap-frames', type=click.INT, default=0,
    help="Number of frames skipped between reconstruction image and pre-post-frames."
)
@click_custom_option(
    '--normalize', type=click.STRING, default="global", help="Type of normalization applied to input data."
)
@click_custom_option(
    '--padding', type=click.INT, default=None, help="Padding added to the FoV."
)
@click_custom_option(
    '--max-per-file', type=click.INT, default=8,
    help="Maximum number of sections extracted from training file per epoch."
)
@click_custom_option(
    '--max-per-val-file', type=click.INT, default=3,
    help="Maximum number of sections extracted from validation file per epoch."
)
@click_custom_option(
    '--learning-rate', type=click.FLOAT, default=0.001, help="Learning rate applied to the model"
)
@click_custom_option(
    '--decay-rate', type=click.FLOAT, default=0.99,
    help="Exponential decay of learning rate. Set to None to learn at steady rate."
)
@click_custom_option(
    '--decay-steps', type=click.INT, default=250,
    help="Used with decay_rate to set step interval at which decay occurs."
)
@click_custom_option(
    '--n-stacks', type=click.INT, default=2, help="Number of stacked layers for encoder and decoder architecture."
)
@click_custom_option(
    '--kernel', type=click.INT, default=32, help="Number of kernels in the initial convolutional layer."
)
@click_custom_option(
    '--batch-normalize', type=click.BOOL, default=False, help="Enables batch normalization."
)
@click_custom_option(
    '--loss', type=click.STRING, default="annealed_loss", help="Loss function used to assess reconstruction quality."
)
@click_custom_option(
    '--pretrained-weights', type=click.STRING, default=None,
    help="Loads pretrained weights from saved model (file path) or keras checkpoints (directory)."
)
@click_custom_option(
    '--save-path', type=click.STRING, default=None, help="Path to save model and checkpoints to."
)
@click_custom_option(
    '--use-cpu', type=click.BOOL, default=True, help="Toggles between training on CPU and GPU. "
                                                     "GPU is significantly faster, but might not be available on all systems."
)
@click_custom_option(
    '--in-memory', type=click.BOOL, default=False, help="Toggle if training data is loaded into memory."
)
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
def train_denoiser(
        training_files, validation_files, input_size, learning_rate, decay_rate, decay_steps, n_stacks, kernel,
        batch_normalize, loss, pretrained_weights, use_cpu, train_rotation, pre_post_frames, gap_frames, loc,
        max_per_file, max_per_val_file, batch_size, padding, in_memory, normalize, train_flip, batch_size_val, epochs,
        patience, min_delta, save_path, logging_level
):
    import astrocast.denoising as denoising

    with UserFeedback(params=locals(), logging_level=logging_level):
        # Trainer
        if "*" in training_files:
            train_str = Path(training_files)
            train_paths = list(Path(train_str.parent).glob(train_str.name))
        else:
            train_paths = Path(training_files)

        if isinstance(train_paths, list) and len(train_paths) < 1:
            raise FileNotFoundError(
                f"No training files found in folder "
                f"{Path(train_str.parent)} with search string "
                f"{train_str.name}"
            )

        train_gen = denoising.SubFrameGenerator(paths=train_paths, max_per_file=max_per_file, loc=loc,
                                                input_size=input_size, pre_post_frames=pre_post_frames,
                                                gap_frames=gap_frames, allowed_rotation=train_rotation, padding=padding,
                                                batch_size=batch_size, normalize=normalize, in_memory=in_memory,
                                                allowed_flip=train_flip)

        # Validator
        if validation_files is not None:
            if "*" in validation_files:
                val_str = Path(validation_files)
                val_paths = list(Path(val_str.parent).glob(val_str.name))
            else:
                val_paths = Path(validation_files)

            if isinstance(val_paths, list) and len(val_paths) < 1:
                raise FileNotFoundError(
                    f"No validation files found in folder "
                    f"{Path(val_str.parent)} with search string "
                    f"{val_str.name}"
                )

            val_gen = denoising.SubFrameGenerator(paths=val_paths, max_per_file=max_per_val_file,
                                                  batch_size=batch_size_val, loc=loc, input_size=input_size,
                                                  pre_post_frames=pre_post_frames, gap_frames=gap_frames,
                                                  allowed_rotation=[0], padding=padding, normalize=normalize,
                                                  in_memory=in_memory, allowed_flip=[-1], shuffle=False)
        else:
            val_gen = None

        # Network
        net = denoising.Network(
            train_generator=train_gen, val_generator=val_gen, learning_rate=learning_rate, decay_rate=decay_rate,
            decay_steps=decay_steps, pretrained_weights=pretrained_weights, loss=loss, n_stacks=n_stacks, kernel=kernel,
            batchNormalize=batch_normalize, use_cpu=use_cpu
        )

        net.run(
            batch_size=1, num_epochs=epochs, patience=patience, min_delta=min_delta, save_model=save_path
        )


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option(
    '--model', type=click.Path(), default=None, help='Path to the trained model file or the model object itself.'
)
@click_custom_option(
    '--output-file', type=click.Path(), default=None,
    help='Path to the output file where the results will be saved. If not provided, the result will be returned instead of being saved to a file.'
)
@click_custom_option('--batch-size', type=click.INT, default=8, help='batch size processed in each step.')
@click_custom_option(
    '--input-size', type=(click.INT, click.INT), default=(256, 256), help='size of the denoising window'
)
@click_custom_option(
    '--pre-post-frames', type=click.INT, default=5,
    help='Number of frames before and after the central frame in each data chunk.'
)
@click_custom_option(
    '--gap-frames', type=click.INT, default=0, help='Number of frames to skip in the middle of each data chunk.'
)
@click_custom_option(
    '--z-select', type=(click.INT, click.INT), default=None,
    help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
)
@click_custom_option(
    '--overlap', type=click.INT, default=10, help='Overlap of reconstructed sections to prevent sharp borders.'
)
@click_custom_option('--padding', type=click.STRING, default="edge", help='Padding mode for the data chunks.')
@click_custom_option('--normalize', type=click.STRING, default="global", help='Normalization mode for the data.')
@click_custom_option(
    '--loc', type=click.STRING, default="data/", help='Location in the input file(s) where the data is stored.'
)
@click_custom_option('--in-memory', type=click.BOOL, default=False, help='Whether to store data in memory.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option(
    '--out-loc', type=click.STRING, default=None, help='Location in the output file where the results will be saved.'
)
@click_custom_option(
    '--dtype', type=click.STRING, default="same",
    help='Data type for the output. If "same", the data type of the input will be used.'
)
@click_custom_option(
    "--infer-chunks", is_flag=True, default=False, help="Whether to infer the chunks in the output file."
)
@click_custom_option(
    '--chunks', type=(click.INT, click.INT, click.INT), default=None,
    help='Chunk size for saving the results in the output file. If not provided, a default chunk size will be used.'
)
@click_custom_option('--rescale', type=click.BOOL, default=True, help='Whether to rescale the output values.')
def denoise(
        input_path, batch_size, input_size, pre_post_frames, gap_frames, z_select, logging_level, model, out_loc, dtype,
        infer_chunks, chunks, rescale, overlap, padding, normalize, loc, in_memory, output_file
):
    """
    Denoise the input data using the SubFrameGenerator class and infer method.
    """

    if model is None:
        raise ValueError(f"Please provide a model with '--model'")

    if output_file is None:
        output_file = input_path
        logging.warning(f"No output_file provided. Choosing input_file: {input_path}")

    from astrocast.denoising import SubFrameGenerator

    chunks = parse_chunks(infer_chunks, chunks)

    with UserFeedback(params=locals(), logging_level=logging_level):
        # Initializing the SubFrameGenerator instance
        sub_frame_generator = SubFrameGenerator(paths=input_path, batch_size=batch_size, input_size=input_size,
                                                pre_post_frames=pre_post_frames, gap_frames=gap_frames,
                                                z_select=z_select, allowed_rotation=[0], allowed_flip=[-1],
                                                overlap=overlap, padding=padding, shuffle=False, normalize=normalize,
                                                loc=loc, in_memory=in_memory, save_global_descriptive=False,
                                                logging_level=logging_level)

        # Running the infer method
        result = sub_frame_generator.infer(
            model=model, output=output_file, out_loc=out_loc, dtype=dtype, chunk_size=chunks, rescale=rescale
        )

        logging.info(f"saved inference to: {output_file}")


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--loc', type=click.STRING, default=None, help='Specific dataset to run the process on.')
@click_custom_option(
    '--threshold', type=click.FLOAT, default=None, help='Threshold value to discriminate background from events.'
)
@click_custom_option(
    '--exclude-border', type=click.INT, default=0,
    help='Number of pixels at the image border to exclude from the detection.'
)
@click_custom_option('--use-smoothing', type=click.BOOL, default=True, help='Whether to use smoothing.')
@click_custom_option('--smooth-radius', type=click.INT, default=2, help='Radius for smoothing kernel.')
@click_custom_option('--smooth-sigma', type=click.INT, default=2, help='Sigma value for smoothing kernel.')
@click_custom_option(
    '--use-spatial', type=click.BOOL, default=True, help='Whether to use spatial considerations in the algorithm.'
)
@click_custom_option('--spatial-min-ratio', type=click.INT, default=1, help='Minimum ratio for spatial considerations.')
@click_custom_option('--spatial-z-depth', type=click.INT, default=1, help='Z-depth for spatial considerations.')
@click_custom_option(
    '--use-temporal', type=click.BOOL, default=True, help='Whether to consider temporal elements in the algorithm.'
)
@click_custom_option(
    '--temporal-prominence', type=click.INT, default=10, help='Prominence value for temporal considerations.'
)
@click_custom_option('--temporal-width', type=click.INT, default=3, help='Width value for temporal considerations.')
@click_custom_option(
    '--temporal-rel-height', type=click.FLOAT, default=0.9, help='Relative height for temporal considerations.'
)
@click_custom_option('--temporal-wlen', type=click.INT, default=60, help='Window length for temporal considerations.')
@click_custom_option(
    '--temporal-plateau-size', type=click.INT, default=None, help='Plateau size for temporal considerations.'
)
@click_custom_option(
    '--comb-type', type=click.STRING, default='&', help='Combination type for merging spatial and temporal results.'
)
@click_custom_option(
    '--fill-holes', type=click.BOOL, default=True, help='Whether to fill holes in the detected events.'
)
@click_custom_option('--area-threshold', type=click.INT, default=10, help='Area threshold for hole filling.')
@click_custom_option(
    '--holes-connectivity', type=click.INT, default=1, help='Connectivity consideration for hole filling.'
)
@click_custom_option('--holes-depth', type=click.INT, default=1, help='Depth consideration for hole filling.')
@click_custom_option(
    '--remove-objects', type=click.BOOL, default=True, help='Whether to remove objects during processing.'
)
@click_custom_option('--min-size', type=click.INT, default=20, help='Minimum size for object removal.')
@click_custom_option(
    '--object-connectivity', type=click.INT, default=1, help='Connectivity consideration for object removal.'
)
@click_custom_option('--objects-depth', type=click.INT, default=1, help='Depth consideration for object removal.')
@click_custom_option(
    '--fill-holes-first', type=click.BOOL, default=True, help='Whether to fill holes before other processing steps.'
)
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option(
    '--adjust-for-noise', type=click.BOOL, default=False, help='Whether to adjust event detection for background noise.'
)
@click_custom_option(
    '--subset', type=(click.INT, click.INT), default=None, help='Subset of frames (z0, z1) of the dataset to process.'
)
@click_custom_option(
    '--split-events', type=click.BOOL, default=False,
    help='Whether to split detected events into smaller events if multiple peaks are detected.'
)
@click_custom_option('--debug', type=click.BOOL, default=False, help='Enable debug mode.')
@click_custom_option('--parallel', type=click.BOOL, default=True, help='Enable parallel execution.')
@click_custom_option('--output-path', type=click.STRING, default="infer", help='Path to the output file.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option("--on-disk", type=click.BOOL, default=False,
                     help='Flag to toggle between on-disk (mmap) and in-RAM (shared memory) methods.')
@click_custom_option(
    '--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location'
)
def detect_events(
        input_path, loc, exclude_border, threshold, use_smoothing, smooth_radius, smooth_sigma, use_spatial,
        spatial_min_ratio, spatial_z_depth, use_temporal, temporal_prominence, temporal_width, temporal_rel_height,
        temporal_wlen, temporal_plateau_size, comb_type, fill_holes, area_threshold, holes_connectivity, holes_depth,
        remove_objects, min_size, object_connectivity, objects_depth, fill_holes_first, lazy, adjust_for_noise, subset,
        split_events, debug, parallel, output_path, logging_level, overwrite, on_disk
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
                raise FileExistsError(
                    f"Aborting detection because previous calculation exists ({output_path}."
                    f"Please provide an alternative output path or set '--overwrite True'"
                )

        logging.warning(f"input: {input_path}")

        # check container
        if is_docker() and not on_disk:
            logging.warning("Suspecting to be in container, switching to 'on_disk=True'.")
            on_disk = True

        # Initializing the Detector instance
        detector = Detector(input_path=input_path, output=output_path, logging_level=logging_level)

        # Running the detection
        detector.run(
            loc=loc, exclude_border=exclude_border, threshold=threshold, use_smoothing=use_smoothing,
            smooth_radius=smooth_radius, smooth_sigma=smooth_sigma, use_spatial=use_spatial,
            spatial_min_ratio=spatial_min_ratio, spatial_z_depth=spatial_z_depth, use_temporal=use_temporal,
            temporal_prominence=temporal_prominence, temporal_width=temporal_width,
            temporal_rel_height=temporal_rel_height, temporal_wlen=temporal_wlen,
            temporal_plateau_size=temporal_plateau_size, comb_type=comb_type, fill_holes=fill_holes,
            area_threshold=area_threshold, holes_connectivity=holes_connectivity, holes_depth=holes_depth,
            remove_objects=remove_objects, min_size=min_size, object_connectivity=object_connectivity,
            objects_depth=objects_depth, fill_holes_first=fill_holes_first, lazy=lazy, use_on_disk_sharing=on_disk,
            adjust_for_noise=adjust_for_noise, z_slice=subset, split_events=split_events, debug=debug, parallel=parallel
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
            details = [f"shape: {item.shape}", f"dtype: {item.dtype}", ]
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
    print(f"> {os.path.basename(input_path)} ({file_size})")

    import h5py

    with h5py.File(input_path) as f:
        visualize_h5_recursive(f['/'])


@cli.command()
@click.argument('input-path', type=click.Path())
@click.argument('output-path', type=click.Path())
@click.argument('loc-in', type=click.STRING)
@click.argument('loc-out', type=click.STRING)
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Overwrite output dataset.')
def move_h5_dataset(input_path, output_path, loc_in, loc_out, overwrite):
    import h5py as h5

    with h5.File(input_path) as in_:
        with h5.File(output_path, "a") as out_:

            if loc_in not in in_:
                raise ValueError(f"cannot find {loc_in} in {input_path}. Choose: {list(in_.keys())}")

            if loc_out in out_:

                if not overwrite:
                    raise ValueError(
                        f"{loc_out} exists in {output_path}. "
                        f"Choose different dataset name or overwrite==True"
                    )
                else:
                    del out_[loc_out]

            data = in_[loc_in]
            out_.create_dataset(loc_out, data=data)
            print("done copying")


@cli.command()
@click.argument('input-path', type=click.Path())
@click.argument('loc', nargs=-1)
@click_custom_option('--colormap', type=click.STRING, default="gray", help='Color of the video layer.')
@click_custom_option('--show-trace', type=click.BOOL, default=False, help='Display trace of the video')
@click_custom_option('--window', type=click.INT, default=160, help='window of trace to be shown')
@click_custom_option(
    '--z-select', type=(click.INT, click.INT), default=None,
    help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
)
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option('--testing', type=click.BOOL, default=False, help='Auto closes napari for testing purposes.')
def view_data(input_path, loc, z_select, lazy, show_trace, window, colormap, testing):
    """
    Displays a video from a data file (.h5, .tiff, .tdb).

    This function uses the Video class to create a video object from a dataset  and displays the video using napari.
    The function provides options to select a specific dataset within the h5 file, a range of frames to display,
    and whether to use lazy loading.

    Parameters:
    input_path (str): The path to the h5 file.
    loc (str): The name or identifier of the dataset within the h5 file. Defaults to an empty string, which indicates the root group.
    z_select (tuple of int, optional): A tuple specifying the range of frames to select in the Z dimension. The tuple contains two elements: the start and end frame numbers. Defaults to None, which indicates that all frames should be selected.
    lazy (bool): Whether to implement lazy loading, which can improve performance when working with large datasets by only loading data into memory as it is needed. Defaults to True.

    Returns:
    None

    Examples:
    view_data('path/to/your/file.h5', loc='dataset_name', z_select=(10, 20), lazy=True)
    """

    import napari
    from astrocast.analysis import Video

    if loc is None:
        loc = ""

    vid = Video(data=input_path, z_slice=z_select, loc=loc, lazy=lazy)
    vid.show(show_trace=show_trace, window=window, colormap=colormap)

    if not testing:
        napari.run()


@cli.command()
@click.argument('event_dir', type=click.Path())
@click_custom_option('--video-path', type=click.STRING, default="infer", help='Path to the data used for detection.')
@click_custom_option(
    '--loc', type=click.STRING, default="df/ch0", help='Name or identifier of the dataset used for detection.'
)
@click_custom_option(
    '--z-select', type=(click.INT, click.INT), default=None,
    help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
)
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option(
    '--testing', type=click.BOOL, default=False, help='Automatically closes viewer for testing purposes.'
)
def view_detection_results(event_dir, video_path, loc, z_select, lazy, testing):
    """
    view the detection results; optionally overlayed on the input video.

    Parameters:
    event_dir (str): The path to the directory where the event data is stored. This path must exist.
    video_path (str, optional): The path to the data used for detection. If "infer", the path will be inferred. Defaults to "infer".
    loc (str, optional): The name or identifier of the dataset used for detection within the HDF5 file. Defaults to an empty string.
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

    event = Events(event_dir=event_dir, data=video_path, loc=loc, z_slice=z_select, lazy=lazy)
    viewer = event.show_event_map(z_slice=z_select)

    if not testing:
        viewer.show()
        napari.run()


@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=click.Path(), required=True, help='Path to the output file.')
@click_custom_option(
    '--loc-in', type=click.STRING, default="", help='Name or identifier of the dataset in the h5 file.'
)
@click_custom_option(
    '--loc-out', type=click.STRING, default="", help='Name or identifier of the dataset in the h5 file.'
)
@click_custom_option(
    '--z-select', type=(click.INT, click.INT), default=None,
    help='Range of frames to select in the Z dimension, given as a tuple (start, end).'
)
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option(
    '--chunk-size', type=(click.INT, click.INT, click.INT), default=None,
    help='Chunk size for saving the results in the output file. If not provided, a default chunk '
         'size will be used.'
)
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option(
    '--rescale', type=click.FLOAT, default=None, help='(float): The rescaling factor to '
                                                      'apply to the data array.'
)
@click_custom_option(
    '--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location'
)
def export_video(
        input_path, output_path, loc_in, loc_out, z_select, lazy, chunk_size, compression, rescale, overwrite
):
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
    loc_in (str, optional): The name or identifier of the dataset within the input h5 file. Defaults to an empty string, which indicates the root group.
    loc_out (str, optional): The name or identifier of the dataset within the output file. Defaults to an empty string, which indicates the root group.
    z_select (tuple of int, optional): A tuple specifying the range of frames to select in the Z dimension. The tuple contains two elements: the start and end frame numbers. Defaults to None, which indicates that all frames should be selected.
    lazy (bool, optional): Whether to implement lazy loading, which can improve performance when working with large datasets by only loading data into memory as it is needed. Defaults to True.
    chunk_size (tuple of int, optional): A tuple specifying the chunk size for saving the results in the output file. If not provided, a default chunk size will be used. Defaults to None.
    compression (str, optional): The compression method to use when saving data to the output file. If not provided, no compression is applied. Defaults to None.
    rescale float: The rescaling factor to apply to the data arrays.
    overwrite (bool, optional): Whether to overwrite previous results in the output location if they exist. Defaults to False.

    Returns:
    None

    Example:
    export_video('input.h5', 'output.h5', loc_in='dataset1', loc_out='dataset2', z_select=(10, 20), lazy=True, chunk_size=(100, 100), compression='gzip', overwrite=True)
    """

    if Path(output_path).exists() and not overwrite:
        logging.error(
            f"file already exists {output_path}. Please choose a different output location "
            f"or use '--overwrite True'."
        )

        return 0

    from astrocast.preparation import IO

    io = IO()
    data = io.load(input_path, loc=loc_in, z_slice=z_select, lazy=lazy)

    if rescale is not None and rescale != 1.0:
        data = Input._rescale_data(data, rescale=float(rescale))

    chunks = io.infer_chunks_from_array(arr=data, chunks=chunk_size)
    io.save(output_path, data=data, loc=loc_out, chunks=chunks, compression=compression, overwrite=overwrite)


@cli.command()
@click_custom_option('--input-path', type=click.Path(), default=None, help='Path to input file.')
@click_custom_option(
    '--loc', type=click.STRING, default=None, help='Name or identifier of the dataset in the h5 file.'
)
def explorer(input_path, loc):
    from astrocast.app_preparation import Explorer

    app_instance = Explorer(input_path=input_path, loc=loc)
    app_instance.run()


@cli.command()
@click_custom_option('--input-path', type=click.Path(), default=None, help='Path to event detection output (.roi).')
@click_custom_option('--video-path', type=click.Path(), default=None, help='Path to video file.')
@click_custom_option('--loc', type=click.STRING, default="", help='dataset location for .h5 files')
@click_custom_option('--default-settings', type=dict, default={}, help='settings for app.')
def analysis(input_path, video_path, loc, default_settings):
    """Run interactive analysis of events."""

    from astrocast.app_analysis import Analysis

    app_instance = Analysis(
        input_path=input_path, video_path=video_path, loc=loc, default_settings=default_settings
    )
    app_instance.run()


@cli.command
@click.argument('save-path', type=click.Path())
@click_custom_option('--get-public', type=click.BOOL, default=True, help='Flag to download public datasets.')
@click_custom_option('--get-custom', type=click.BOOL, default=True, help='Flag to download custom datasets.')
def download_datasets(save_path, get_public, get_custom):
    from astrocast.helper import download_sample_data
    download_sample_data(save_path, public_datasets=get_public, custom_datasets=get_custom)


@cli.command
@click.argument('save-path', type=click.Path())
def download_models(save_path):
    from astrocast.helper import download_pretrained_models
    download_pretrained_models(save_path)


@cli.command
@click.argument('input-path', type=click.Path())
@click.argument('z', type=click.INT, nargs=-1)
@click_custom_option('--loc', type=click.STRING, default="data/ch0")
@click_custom_option('--equalize', type=click.BOOL, default=True)
@click_custom_option('--clip-limit', type=click.FLOAT, default=0.01)
@click_custom_option('--size', type=(click.INT, click.INT), default=(50, 50))
def climage(input_path, loc, z, size, equalize, clip_limit):
    import skimage.color as skicol
    from skimage.transform import resize
    from skimage import exposure
    import climage
    from astrocast.preparation import IO

    if isinstance(z, int):
        z = [z]

    z0 = max(0, min(z) - 1)
    z1 = max(z) + 2

    io = IO()
    data = io.load(input_path, loc=loc, lazy=True, z_slice=(z0, z1))
    print(f"input path: {input_path}")

    # enforce even size
    size = [size[0] + size[0] % 2, size[1] + size[1] % 2]

    for zi in z:
        img = data[zi - z0, :, :].astype(float).compute()

        print(f"z:{z1} [{np.min(img):.1f}-{np.max(img):.1f}, {np.mean(img):.1f}+-{np.std(img):.1f}]")

        img = img - np.min(img)
        img = img / np.max(img)

        img = resize(img, size, anti_aliasing=True)

        if equalize:
            img = exposure.equalize_adapthist(img, clip_limit=clip_limit)

        img = skicol.gray2rgb(img)
        img = np.array(img) * 255

        print(climage.convert_array(img, is_unicode=True))


@cli.command
@click.argument('input-path', type=click.Path())
@click_custom_option('--loc', type=click.STRING, default=None)
def delete_h5_dataset(input_path, loc):
    import h5py as h5
    from pathlib import Path

    input_path = Path(input_path)
    assert input_path.is_file()

    with h5.File(input_path.as_posix(), "a") as f:

        if loc is None:

            while True:
                visualize_h5_recursive(f['/'])

                in_ = input("Choose dataset or 'exit': ")
                if in_ in f:
                    del f[in_]
                elif in_ == "exit":
                    break

        else:

            for ds in loc.split(","):

                if ds in f:
                    del f[ds]


@cli.command
@click.argument('data-path', type=click.Path())
@click_custom_option('--cfg-path', type=click.Path(), default=Path("config.yaml"))
@click_custom_option('--log-path', type=click.Path(), default=Path("."))
@click_custom_option('--tasks', default=None)
@click_custom_option('--base-command', type=click.STRING, default="")
@click_custom_option('--account', type=click.STRING)
def push_slurm_tasks(log_path, cfg_path, data_path, tasks, base_command, account):
    import h5py as h5
    from simple_slurm import Slurm

    if tasks is None:
        raise ValueError(f"Please provide a dictionary of tasks.")

    data_path = Path(data_path)
    cfg_path = Path(cfg_path)
    log_path = Path(log_path)

    files = [data_path] if data_path.is_file() else list(data_path.glob("*/*.h5"))

    if isinstance(tasks, str):
        tasks = ast.literal_eval(tasks)

    task_ids = sorted(tasks.keys())
    for f in files:

        last_jobid = None
        with h5.File(f) as file:
            print(f"{f}:")

            for i in task_ids:

                dict_ = tasks[i]

                cmd = base_command[:]

                k = dict_["key"]
                print(f"\tchecking: {k}")

                base_name = f.name.replace(".h5", "")
                log_name = log_path.joinpath(f"slurm-%A-{base_name}-{k}.out").as_posix()
                job_name = f"{k}_{base_name}"

                if (k == "roi" and "df" in file and not f.with_suffix(".roi").exists()) or (k not in file):

                    cmd += f"astrocast --config {cfg_path} {dict_['script']} {f};"
                    print(f"\tcmd {k}>{base_name}:\n {cmd}")

                    dependency = dict(afterok=last_jobid) if last_jobid is not None else None

                    slurm = Slurm()
                    slurm.add_arguments(A=account)
                    slurm.add_arguments(time=dict_["time"])
                    slurm.add_arguments(c=dict_["cores"])

                    if job_name is not None:
                        slurm.add_arguments(J=job_name)

                    if log_path is not None:
                        slurm.add_arguments(output=log_name)

                    if dependency is not None:
                        slurm.add_arguments(dependency=dependency)

                    last_jobid = slurm.sbatch(cmd)


if __name__ == '__main__':
    cli()
