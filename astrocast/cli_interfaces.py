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

@click.group(context_settings={'auto_envvar_prefix': 'CLI'})
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
@click_custom_option('input-path')
@click_custom_option('--config', type=click.Path(exists=True), help='Path to the configuration YAML file.')
@click_custom_option('--window', type=click.INT, required=True, help='Size of the window for the minimum filter.')
@click_custom_option('--output-path', type=None, help='Path to save the output data.')
@click_custom_option('--loc', type=click.STRING, default="", help='Location of the data in the HDF5 file (if applicable).')
@click_custom_option('--method', type=click.Choice(['background', 'dF', 'dFF']), default='background', help='Method to use for delta calculation.')
@click_custom_option('--chunks', type=click.STRING, default='infer', help='Chunk size for data processing.')
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
@click_custom_option('--input-path')
@click_custom_option('--logging-level', type=click.INT, default=logging.INFO, help='Logging level for messages.')
@click_custom_option('--output-path', type=Path, help='Path to save the processed data. If None, the processed data is returned.')
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
def convert_input(logging_level, input_path, output_path, sep, channels, z_slice, lazy, subtract_background,
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


if __name__ == '__main__':
    cli()
