import datetime as dt
import logging
import time
from pathlib import Path

import click
import h5py
import humanize
import numpy as np
import yaml
from functools import partial

from astrocast.denoising import SubFrameGenerator
from astrocast.detection import Detector
from astrocast.preparation import MotionCorrection, Delta, Input

click_custom_option = partial(click.option, show_default=True)

def check_output(output_path, input_path, h5_loc_save, overwrite):

    if output_path is None:
        logging.warning(f"No output_path provided. Assuming input_path: {input_path}")
        output_path = input_path

    output_path = Path(output_path)
    if output_path.exists():

        if output_path.suffix in (".hdf5", ".h5"):
            with h5py.File(output_path.as_posix(), "r") as f:
                if h5_loc_save in f and not overwrite:
                    raise FileExistsError(f"{h5_loc_save} already exists in {output_path}. "
                                          f"Please choose a different output location or use '--overwrite True'")

        else:
            raise FileExistsError(f"file already exists {output_path}. Please choose a different output location "
                                  f"or use '--overwrite True'.")

    return output_path

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
@click_custom_option('--chunks', type=click.Tuple([int, int, int]), default=None, help='Chunk shape for creating a dask array when saving to an HDF5 file.')
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
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--window', type=click.INT, required=True, help='Size of the window for the minimum filter.')
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--loc', type=click.STRING, default="", help='Location of the data in the HDF5 file (if applicable).')
@click_custom_option('--method', type=click.Choice(['background', 'dF', 'dFF']), default='background', help='Method to use for delta calculation.')
@click_custom_option('--chunks', type=click.Tuple([int, str, str]), default=(1, 100, 100), help='Chunk size for data processing.')
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

    # Initialize the Delta instance
    logging.info("creating delta instance ...")
    delta_instance = Delta(input_=input_path, loc=loc)

    # Run the delta calculation
    logging.info("subtracting background ...")
    result = delta_instance.run(method=method, window=window, chunks=chunks, output_path=None, overwrite_first_frame=overwrite_first_frame, lazy=lazy)

    # Save the results to the specified output path
    logging.info("saving result ...")
    delta_instance.save(output_path=output_path, h5_loc=h5_loc, chunks=(1, "auto", "auto"), compression=compression)

    # logging
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Motion correction finished in {delta}")


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
@click_custom_option('--chunks', default=None, help='Chunk size to use when saving to HDF5 or TileDB.')
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

    # convert input
    input_instance = Input(logging_level=logging_level)
    input_instance.run(input_path=input_path, output_path=output_path, sep=sep, channels=channels, z_slice=z_slice,
                       lazy=lazy, subtract_background=subtract_background, subtract_func=subtract_func, rescale=rescale,
                       dtype=dtype, in_memory=in_memory, h5_loc=h5_loc, chunks=chunks, compression=compression)

    # logging
    delta = humanize.naturaldelta(dt.timedelta(seconds=time.time() - t0))
    logging.info(f"Motion correction finished in {delta}")

@cli.command()
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--model', type=click.Path(exists=True), required=True, help='Path to the trained model file or the model object itself.')
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
@click.argument('input-path', type=click.Path(exists=True))
@click_custom_option('--output-path', type=click.Path(), default=None, help='Path to the output file.')
@click_custom_option('--indices', type=click.STRING, default=None, help='Indices in a numpy array format.')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--h5_loc', type=click.STRING, default=None, help='Name or identifier of the dataset in the h5 file.')
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
            Path(output_path).unlink()
        else:
            raise FileExistsError(f"Aborting detection because previous calculation exists ({output_path}."
                                  f"Please provide an alternative output path or set '--overwrite True'")


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


if __name__ == '__main__':
    cli()
