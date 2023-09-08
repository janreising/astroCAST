import datetime as dt
import logging
import os
import shutil
import time
from pathlib import Path

import click
import h5py
import humanize
import napari
import numpy as np
import yaml
from functools import partial

from astrocast.analysis import Video
from astrocast.denoising import SubFrameGenerator
from astrocast.detection import Detector
from astrocast.preparation import MotionCorrection, Delta, Input, IO

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
            with h5py.File(output_path.as_posix(), "r") as f:
                if h5_loc_save in f and not overwrite:
                    logging.error(f"{h5_loc_save} already exists in {output_path}. "
                                          f"Please choose a different output location or use '--overwrite True'")
                    return 0

        else:
            logging.error(f"file already exists {output_path}. Please choose a different output location "
                                  f"or use '--overwrite True'.")
            return 0

    return output_path

def parse_chunks(chunks):

    if chunks is None or chunks == "infer":
        return chunks

    else:
        chunks = tuple(int(c) for c in chunks.split(","))
        if len(chunks) != 3:
            raise ValueError(f"please provide 'chunks' parameter as None, infer or comma-separated list of 3 int values.")
        return chunks

@click.group(context_settings={'auto_envvar_prefix': 'CLI'}, chain=True)
@click.option('--config', default=None, type=click.Path())  # this allows us to change config path
@click.pass_context
def cli(ctx, config):
    if config is not None:

        logging.warning(f"Loading configurations from '{config}'. Please note that parameters specified directly in the command line will override the settings in the YAML configuration file.")

        with open(config, 'r') as file:
            config = yaml.safe_load(file)

        logging.warning(f"yaml-config: {config}")

        ctx.default_map = config

@cli.command()
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--output-path', type=click.Path(), help='Path to save the processed data. If None, the processed data is returned.')
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
@click_custom_option('--chunks', type=click.STRING, default="infer", help='Chunk size to use when saving to HDF5 or TileDB.')
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location')
def convert_input(input_path, logging_level, output_path, sep, channels, z_slice, lazy, subtract_background,
                  subtract_func, rescale, dtype, in_memory, h5_loc, chunks, compression, overwrite):

    """
    Convert user files to astroCAST compatible format using the Input class.
    """

    logging.basicConfig(level=logging_level)
    t0 = time.time()

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
                       lazy=lazy, subtract_background=subtract_background, subtract_func=subtract_func, rescale=rescale,
                       dtype=dtype, in_memory=in_memory, h5_loc=h5_loc, chunks=chunks, compression=compression)

    # logging
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Motion correction finished in {delta}")

@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--working-directory', type=click.Path(), default=None, help='Working directory for temporary files.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--h5-loc', type=click.STRING, default="", help='Dataset name in case of input being an HDF5 file.')
@click_custom_option('--max-shifts', type=click.Tuple([int, int]), default=(50, 50), help='Maximum allowed rigid shift.')
@click_custom_option('--niter-rig', type=click.INT, default=3, help='Maximum number of iterations for rigid motion correction.')
@click_custom_option('--splits-rig', type=click.INT, default=14, help='Number of splits across time for parallelization during rigid motion correction.')
@click_custom_option('--num-splits-to-process-rig', type=click.INT, default=None, help='Number of splits to process during rigid motion correction.')
@click_custom_option('--strides', type=click.Tuple([int, int]), default=(48, 48), help='Intervals at which patches are laid out for motion correction.')
@click_custom_option('--overlaps', type=click.Tuple([int, int]), default=(24, 24), help='Overlap between patches (size of patch strides+overlaps).')
@click_custom_option('--pw-rigid', type=click.BOOL, default=False, help='Flag for performing motion correction when calling motion_correct.')
@click_custom_option('--splits-els', type=click.INT, default=14, help='Number of splits across time for parallelization during elastic motion correction.')
@click_custom_option('--num-splits-to-process-els', type=click.INT, default=None, help='Number of splits to process during elastic motion correction.')
@click_custom_option('--upsample-factor-grid', type=click.INT, default=4, help='Upsample factor of shifts per patches to avoid smearing when merging patches.')
@click_custom_option('--max-deviation-rigid', type=click.INT, default=3, help='Maximum deviation allowed for patch with respect to rigid shift.')
@click_custom_option('--nonneg-movie', type=click.BOOL, default=True, help='Make the saved movie and template mostly nonnegative by removing min_mov from movie.')
@click_custom_option('--gsig-filt', type=click.Tuple([int, int]), default=(20, 20), help='Tuple indicating the size of the filter.')
@click_custom_option('--h5-loc-save', type=click.STRING, default="mc", help='Location within the HDF5 file to save the data.')
@click_custom_option('--chunks', type=click.STRING, default=None, help='Chunk shape for creating a dask array when saving to an HDF5 file.')
@click_custom_option('--compression', type=click.STRING, default=None, help='Compression algorithm to use when saving to an HDF5 file.')
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location')
def motion_correction(input_path, working_directory, logging_level, output_path, h5_loc,
                          max_shifts, niter_rig, splits_rig, num_splits_to_process_rig, strides,
                          overlaps, pw_rigid, splits_els, num_splits_to_process_els, upsample_factor_grid,
                          max_deviation_rigid, nonneg_movie, gsig_filt, h5_loc_save, chunks, compression, overwrite):
    """
    Correct motion artifacts of input data using the MotionCorrection class.
    """

    logging.basicConfig(level=logging_level)
    t0 = time.time()

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
           max_deviation_rigid=max_deviation_rigid, nonneg_movie=nonneg_movie, gSig_filt=gsig_filt)

    # Save the results to the specified output path
    logging.info("saving result ...")
    mc.save(output_path, h5_loc=h5_loc_save, chunks=chunks, compression=compression)

    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Motion correction finished in {delta}")

@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--window', type=click.INT, required=True, help='Size of the window for the minimum filter.')
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--loc', type=click.STRING, default="", help='Location of the data in the HDF5 file (if applicable).')
@click_custom_option('--method', type=click.Choice(['background', 'dF', 'dFF']), default='background', help='Method to use for delta calculation.')
@click_custom_option('--chunks', type=click.STRING, default="infer", help='Chunk size for data processing.')
@click_custom_option('--overwrite-first-frame', type=click.BOOL, default=True, help='Whether to overwrite the first frame with the second frame after delta calculation.')
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Flag for lazy data loading and computation.')
@click_custom_option('--h5-loc', type=click.STRING, default="df", help='Location within the HDF5 file to save the data.')
@click_custom_option('--compression', type=click.STRING, default=None, help='Compression algorithm to use when saving to an HDF5 file.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location')
def subtract_delta(input_path, output_path, loc, method, window, chunks, overwrite_first_frame, lazy, h5_loc,
              compression, logging_level, overwrite):
    """
    Subtract baseline of input using the Delta class.
    """

    logging.basicConfig(level=logging_level)
    t0 = time.time()

    # check output
    output_path = check_output(output_path, input_path, h5_loc, overwrite)
    if output_path == 0:
        logging.warning("skipping this step because output exists.")
        return 0

    # check chunks
    chunks = parse_chunks(chunks)

    # Initialize the Delta instance
    logging.info("creating delta instance ...")
    delta_instance = Delta(input_=input_path, loc=loc)

    # Run the delta calculation
    logging.info("subtracting background ...")
    result = delta_instance.run(method=method, window=window, chunks=chunks, output_path=None, overwrite_first_frame=overwrite_first_frame, lazy=lazy)

    # Save the results to the specified output path
    logging.info("saving result ...")
    delta_instance.save(output_path=output_path, h5_loc=h5_loc, chunks=chunks, compression=compression, overwrite=overwrite)

    # logging
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Motion correction finished in {delta}")

@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--model', type=click.Path(), required=True, help='Path to the trained model file or the model object itself.')
@click_custom_option('--output-file', type=click.Path(), required=True, help='Path to the output file where the results will be saved. If not provided, the result will be returned instead of being saved to a file.')
@click_custom_option('--batch-size', type=click.INT, default=16, help='batch size processed in each step.')
@click_custom_option('--input-size', type=(int, int), default=(100, 100), help='size of the denoising window')
@click_custom_option('--pre-post-frame', type=click.INT, default=5, help='Number of frames before and after the central frame in each data chunk.')
@click_custom_option('--gap-frames', type=click.INT, default=0, help='Number of frames to skip in the middle of each data chunk.')
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None, help='Range of frames to select in the Z dimension, given as a tuple (start, end).')
@click_custom_option('--overlap', type=click.FLOAT, default=None, help='Overlap between data chunks.')
@click_custom_option('--padding', type=click.STRING, default=None, help='Padding mode for the data chunks.')
@click_custom_option('--normalize', type=click.STRING, default=None, help='Normalization mode for the data.')
@click_custom_option('--loc', type=click.STRING, default="data/", help='Location in the input file(s) where the data is stored.')
@click_custom_option('--in-memory', type=click.BOOL, default=False, help='Whether to store data in memory.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--out-loc', type=click.STRING, default=None, help='Location in the output file where the results will be saved.')
@click_custom_option('--dtype', type=click.STRING, default="same", help='Data type for the output. If "same", the data type of the input will be used.')
@click_custom_option('--chunk-size', type=(int, int), default=None, help='Chunk size for saving the results in the output file. If not provided, a default chunk size will be used.')
@click_custom_option('--rescale', type=click.BOOL, default=True, help='Whether to rescale the output values.')
def denoise(input_file, batch_size, input_size, pre_post_frame, gap_frames, z_select,
            logging_level, model, output, out_loc, dtype, chunk_size, rescale,
            overlap, padding,
            normalize, loc, in_memory):
    """
    Denoise the input data using the SubFrameGenerator class and infer method.
    """

    logging.basicConfig(level=logging_level)
    t0 = time.time()

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

    # Logging the time taken
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Denoising finished in {delta}")

@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=click.Path(), default=None, help='Path to the output file.')
@click_custom_option('--indices', type=click.STRING, default=None, help='Indices in a numpy array format.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--h5-loc', type=click.STRING, default=None, help='Name or identifier of the dataset in the h5 file.')
@click_custom_option('--threshold', type=click.FLOAT, default=None, help='Threshold value to discriminate background from events.')
@click_custom_option('--min-size', type=click.INT, default=20, help='Minimum size of an event region.')
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option('--adjust-for-noise', type=click.BOOL, default=False, help='Whether to adjust event detection for background noise.')
@click_custom_option('--subset', type=click.STRING, default=None, help='Subset of the dataset to process.')
@click_custom_option('--split-events', type=click.BOOL, default=True, help='Whether to split detected events into smaller events if multiple peaks are detected.')
@click_custom_option('--binary-struct-iterations', type=click.INT, default=1, help='Number of iterations for binary structuring element.')
@click_custom_option('--binary-struct-connectivity', type=click.INT, default=2, help='Connectivity of binary structuring element.')
@click_custom_option('--save-activepixels', type=click.BOOL, default=False, help='Save active pixels or not.')
@click_custom_option('--parallel', type=click.BOOL, default=True, help='Parallel execution of event characterization.')
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location')
def detect_events(input_path, output_path, indices, logging_level, h5_loc, threshold, min_size, lazy,
                  adjust_for_noise, subset, split_events, binary_struct_iterations, binary_struct_connectivity,
                  save_activepixels, parallel, overwrite):
    """
    Detect events using the Detector class.
    """

    logging.basicConfig(level=logging_level)
    t0 = time.time()

    # check output
    if output_path is not None and Path(output_path).exists():

        if overwrite:
            logging.warning(f"overwrite is {overwrite}, deleting previous result")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Aborting detection because previous calculation exists ({output_path}."
                                  f"Please provide an alternative output path or set '--overwrite True'")

    logging.warning(f"input: {input_path}")

    # Initializing the Detector instance
    detector = Detector(input_path=input_path, output=output_path,
                        indices=np.array(eval(indices)) if indices else None, logging_level=logging_level)

    # Running the detection
    detector.run(dataset=h5_loc, threshold=threshold, min_size=min_size, lazy=lazy,
                 adjust_for_noise=adjust_for_noise, subset=subset, split_events=split_events,
                 binary_struct_iterations=binary_struct_iterations,
                 binary_struct_connectivity=binary_struct_connectivity,
                 save_activepixels=save_activepixels, parallel=parallel)

    # Logging the time taken
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Event detection finished in {delta}")

def visualize_h5_recursive(loc, indent='', prefix=''):
    """Recursive part of the function to visualize the structure."""
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
@click.argument('input-path', type=click.Path(exists=True))
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

    with h5py.File(input_path, 'r') as f:
        visualize_h5_recursive(f['/'])

@cli.command()
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--h5-loc', type=click.STRING, default="", help='Name or identifier of the dataset in the h5 file.')
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None, help='Range of frames to select in the Z dimension, given as a tuple (start, end).')
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
def view_data(input_path, h5_loc, z_select, lazy):
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

    vid = Video(data=input_path, z_slice=z_select, h5_loc=h5_loc, lazy=lazy)
    vid.show()
    napari.run()

@cli.command()
@click.argument('input-path', type=click.Path())
@click_custom_option('--output-path', type=click.Path(), required=True, help='Path to the output file.')
@click_custom_option('--h5-loc-in', type=click.STRING, default="", help='Name or identifier of the dataset in the h5 file.')
@click_custom_option('--h5-loc-out', type=click.STRING, default="", help='Name or identifier of the dataset in the h5 file.')
@click_custom_option('--z-select', type=(click.INT, click.INT), default=None, help='Range of frames to select in the Z dimension, given as a tuple (start, end).')
@click_custom_option('--lazy', type=click.BOOL, default=True, help='Whether to implement lazy loading.')
@click_custom_option('--chunk-size', type=(int, int), default=None, help='Chunk size for saving the results in the output file. If not provided, a default chunk size will be used.')
@click_custom_option('--compression', default=None, help='Compression method to use when saving to HDF5 or TileDB.')
@click_custom_option('--overwrite', type=click.BOOL, default=False, help='Flag for overwriting previous result in output location')
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
                                  f"or use '--overwrite True'.")

        return 0

    io = IO()
    data = io.load(input_path, h5_loc=h5_loc_in, z_slice=z_select, lazy=lazy)

    io.save(output_path, data=data, h5_loc=h5_loc_out, chunks=chunk_size, compression=compression, overwrite=overwrite)

if __name__ == '__main__':
    cli()
