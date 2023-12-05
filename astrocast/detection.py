import json
import logging
import os
import platform
import random
import shutil
import time
import traceback
import warnings
from distutils.version import LooseVersion
from pathlib import Path
from typing import Union, Literal, Tuple, Sequence

import dask.array as da
import numpy as np
import scipy
import scipy.ndimage as ndimage
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask.distributed import progress
from dask_image import ndfilters
from multiprocess import shared_memory
from scipy import signal
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_triangle, gaussian
from skimage.measure import regionprops_table, find_contours
from skimage.segmentation import watershed
from tqdm import tqdm

from astrocast.helper import get_data_dimensions, is_docker
from astrocast.preparation import IO


class Detector:
    # noinspection GrazieInspection
    """
        Detector is a class designed for detecting and analyzing astrocytic events in video datasets,
        particularly focusing on spatial and temporal characteristics of these events.

        The class implements a robust event detection algorithm that leverages both spatial and temporal data to identify
        astrocytic events. The algorithm can be tuned using various parameters to adapt to different datasets and research needs.

        Key Features:
            - Gaussian Smoothing: Enhances events while preserving spatial features. Can be adjusted or omitted based on the dataset.
            - Spatial Thresholding: Utilizes mean fluorescence ratios to differentiate active areas from background, considering the whole frame.
            - Temporal Thresholding: Treats the video as a series of 1D time series, identifying active pixels by peak prominence and other characteristics.
            - Morphological Operations: Corrects for potential artifacts in thresholding, like filling holes or removing noise-based objects.
            - Event Separation: An experimental feature to split closely occurring events for finer analysis.

        .. attention:: Caveats

            - Parameter Sensitivity: The effectiveness of event detection is highly dependent on the choice of parameters, which may need tuning for different datasets.
            - Smoothing Impact: Temporal thresholding is sensitive to the smoothing applied, requiring careful adjustment of smoothing parameters.
            - Noise and Artifacts: The algorithm includes provisions for noise adjustment and artifact removal, but these may not cover all types of dataset-specific noise.
            - Parallel Processing: Default parallel processing can be toggled off for troubleshooting but may affect performance.

        The method `run` executes the event detection process and returns the path to the directory containing the results and metadata.
        It saves all provided arguments for traceability and reproducibility of the analysis.

        Args:
            input_path: Path to the input file.
            output: Path to the output directory. If None, the output directory is created in the input directory.
            logging_level: Sets the level at which information is logged to the console as an integer value.
                    The built-in levels in the logging module are, in increasing order of severity:
                    debug (10), info (20), warning (30), error (40), critical (50).

        Example::

            detector = astrocast.detection.Detector(input_path=/path/to/preprocessed/video)
            detector.run(loc='df/ch0')

        """

    def __init__(
            self, input_path: Union[str, Path], output: Union[str, Path] = None, logging_level: int = logging.INFO
    ):

        # logging
        logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging_level)

        # paths
        self.input_path = Path(input_path)
        self.output = output if output is None else Path(output)
        working_directory = self.input_path.parent

        logging.info(f"working directory: {working_directory}")
        logging.info(f"input file: {self.input_path}")

        # quality check arguments
        if not self.input_path.exists():
            raise FileNotFoundError(f"input file does not exist: {input_path}")
        if (output is not None) and self.output.exists():
            raise FileExistsError(f"output folder already exists: {output}")

        # shared variables
        self.output_directory = None
        self.file = None
        self.data = None
        self.Z, self.X, self.Y = None, None, None
        self.meta = {}

    def run(
            self, loc: str = None, exclude_border: int = 0, threshold: Union[int, float] = None,
            use_smoothing: bool = True, smooth_radius: int = 2, smooth_sigma: int = 2, use_spatial: bool = True,
            spatial_min_ratio: Union[int, float] = 1, spatial_z_depth: int = 1, use_temporal: bool = True,
            temporal_prominence: Union[int, float] = 10, temporal_width: int = 3,
            temporal_rel_height: Union[int, float] = 0.9, temporal_wlen: int = 60, temporal_plateau_size: int = None,
            comb_type: Literal['&', '|'] = "&", fill_holes: bool = True, area_threshold: int = 10,
            holes_connectivity: int = 1, holes_depth: int = 1, remove_objects: bool = True, min_size: int = 20,
            object_connectivity: int = 1, objects_depth: int = 1, fill_holes_first: bool = True, lazy: bool = True,
            adjust_for_noise: bool = False, z_slice: Tuple[int, int] = None, split_events: bool = False,
            debug: bool = False, event_map_export_format: Literal['.tiff', '.h5', '.tdb'] = ".tiff",
            parallel: bool = True, use_on_disk_sharing=False
    ) -> Path:

        # noinspection GrazieInspection
        """
                Executes the AstroCAST event detection algorithm on a specified video dataset.

                Args:
                    loc: Identifier of the dataset within an HDF5 file.
                    exclude_border: Exclude the border pixels to mitigate motion correction artifacts.
                    threshold: Absolute value for simple thresholding; uses automatic thresholding if None.
                    use_smoothing: Apply Gaussian smoothing to enhance events while preserving spatial features.
                    smooth_radius: Radius for the Gaussian smoothing kernel.
                    smooth_sigma: Sigma value for the Gaussian smoothing kernel.
                    use_spatial: Enable spatial thresholding based on the mean fluorescence ratio.
                    spatial_min_ratio: Minimum ratio of active to inactive pixels for spatial thresholding.
                    spatial_z_depth: Number of frames considered for automatic spatial thresholding.
                    use_temporal: Enable temporal thresholding to identify active pixels in timeseries.
                    temporal_prominence: Minimum prominence of peaks for temporal thresholding.
                    temporal_width: Minimum width of peaks to exclude short-duration noise.
                    temporal_rel_height: Defines boundaries of events relative to peak height.
                    temporal_wlen: Window length for prominence calculation in temporal thresholding.
                    temporal_plateau_size: Minimum size of a plateau to be considered an event.
                    comb_type: Combination type for spatial and temporal thresholding ('&' or '|').
                    fill_holes: Apply morphological operations to fill holes in detected events.
                    area_threshold: Maximum size of holes to be filled.
                    remove_objects: Apply morphological operations to remove small objects.
                    objects_depth: Number of frames considered for automatic object removal.
                    min_size: Minimum size of an event region for inclusion in the results.
                    holes_depth: Number of frames considered for automatic temporal thresholding.
                    holes_connectivity: Modifies shape of the element used to fill holes.
                    object_connectivity: Modifies shape of the element used to remove small objects.
                    fill_holes_first: Determines whether holes are filled before removing small objects.
                    lazy: Implement lazy loading of data for efficient memory usage.
                    adjust_for_noise: Adjust event detection for background noise, used with `threshold`.
                    z_slice: Selection of frames that are processed.
                    split_events: Experimental feature to split incorrectly connected events.
                    event_map_export_format: Suffix of the output file for the event map.
                    debug: Enable debug mode to export intermediary steps for troubleshooting.
                    parallel: Enable parallel execution for event characterization.
                    use_on_disk_sharing: Flag to toggle between on-disk (mmap) and in-RAM (shared memory) methods.

                .. warning::

                   The `use_on_disk_sharing` parameter enables the use of on-disk memory mapping (mmap) instead of in-RAM
                   shared memory. While this method ensures compatibility in environments where in-RAM sharing (e.g.,
                   Docker containers) may cause crashes, it is generally slower due to disk I/O operations.
                   Use this method if you encounter issues with shared memory, particularly in containerized environments.

                .. note::

                    - Smoothing parameters (sigma and radius) enhance events while preserving spatial features.
                    - Spatial and temporal thresholding classify pixels as active, potentially belonging to astrocytic events.
                    - Outputs include the event map, time map, and metadata, saved in specified formats.
                    - Debug mode is useful for troubleshooting unsatisfactory event detection results.
                """

        # save function parameters
        function_arguments = locals().copy()  # Create a copy of the local variables

        if 'self' in function_arguments:
            del function_arguments['self']

        self.meta.update(function_arguments)

        # output folder
        self.output_directory = self.output if self.output is not None else self.input_path.with_suffix(
            ".roi"
        ) if loc is None else self.input_path.with_suffix(".{}.roi".format(loc.split("/")[-1]))

        if not self.output_directory.is_dir():
            self.output_directory.mkdir()
        logging.info(f"output directory: {self.output_directory}")

        # profiling
        pbar = ProgressBar(minimum=10)
        pbar.register()

        # load data
        io = IO()
        data = io.load(path=self.input_path, loc=loc, z_slice=z_slice, lazy=lazy)
        self.Z, self.X, self.Y = data.shape
        self.data = data
        logging.info(f"data: {data.shape}") if lazy else logging.info(f"data: {data}")

        # calculate event map
        event_map_path = self.output_directory.joinpath(f"event_map{event_map_export_format}")
        if not event_map_path.exists():
            logging.info("Estimating noise")
            # TODO maybe should be adjusted since it might already be calculated
            noise = self._estimate_background(data) if adjust_for_noise else 1

            logging.info("Thresholding events")
            event_map = self._get_events(
                data, exclude_border=exclude_border, roi_threshold=threshold, var_estimate=noise,
                use_smoothing=use_smoothing, smooth_radius=smooth_radius, smooth_sigma=smooth_sigma,
                use_spatial=use_spatial, spatial_min_ratio=spatial_min_ratio, spatial_z_depth=spatial_z_depth,
                use_temporal=use_temporal, temporal_prominence=temporal_prominence, temporal_width=temporal_width,
                temporal_rel_height=temporal_rel_height, temporal_wlen=temporal_wlen,
                temporal_plateau_size=temporal_plateau_size, comb_type=comb_type, fill_holes=fill_holes,
                area_threshold=area_threshold, holes_connectivity=holes_connectivity, holes_depth=holes_depth,
                remove_objects=remove_objects, min_size=min_size, object_connectivity=object_connectivity,
                objects_depth=objects_depth, fill_holes_first=fill_holes_first, morph_dtype=np.bool_, debug=debug
            )

            logging.info(f"Saving event map to: {event_map_path}")
            io.save(event_map_path, data=event_map)

        else:
            logging.info(f"Loading event map from: {event_map_path}")
            event_map = io.load(event_map_path, lazy=lazy)

        # calculate time map
        logging.info("Calculating time map")
        time_map_path = self.output_directory.joinpath("time_map.npy")
        if not time_map_path.is_file():
            time_map = self._get_time_map(event_map)

            logging.info(f"Saving event map to: {time_map_path}")
            np.save(time_map_path, time_map)
        else:
            logging.info(f"Loading time map from {time_map_path}")
            time_map = np.load(time_map_path.as_posix())

        # calculate features
        logging.info("Calculating features")

        if not use_on_disk_sharing and is_docker():
            logging.warning(
                "Detected execution within a Docker environment. In-Docker shared memory usage can lead to failures. "
                "Automatically switching to on-disk sharing method. To suppress this message, explicitly set "
                "'use_on_disk_sharing' to True when running in Docker."
            )
            use_on_disk_sharing = True

        if use_on_disk_sharing:
            _ = self._custom_slim_features_mmap(time_map, event_map_path, split_events=split_events, parallel=parallel)
        else:
            _ = self._custom_slim_features(time_map, event_map_path, split_events=split_events, parallel=parallel)

        logging.info("Saving meta file")
        with open(self.output_directory.joinpath("meta.json"), 'w') as outfile:
            json.dump(self.meta, outfile)

        logging.info("Run complete! [{}]".format(self.input_path))

        return self.output_directory

    @staticmethod
    def _estimate_background(data: np.array, mask_xy: np.array = None) -> float:

        """ estimates overall noise level

        :param data: numpy.array
        :param mask_xy: binary 2D mask to ignore certain pixel
        :return: estimated standard error
        """
        xx = np.power(data[1:, :, :] - data[:-1, :, :], 2)  # dim: Z, X, Y
        stdMap = np.sqrt(np.median(xx, 0) / 0.9133)  # dim: X, Y

        if mask_xy is not None:
            stdMap[~mask_xy] = None

        stdEst = np.nanmedian(stdMap.flatten(), axis=0)  # dim: 1

        return stdEst

    def _get_events(
            self, data: np.array, roi_threshold: float, var_estimate: float, mask_xy: np.array = None, exclude_border=0,
            use_smoothing=True, smooth_radius=2, smooth_sigma=2, use_spatial=True, spatial_min_ratio=1,
            spatial_z_depth=1, use_temporal=True, temporal_prominence=10, temporal_width=3, temporal_rel_height=0.9,
            temporal_wlen=60, temporal_plateau_size=None, comb_type="&", fill_holes=True, area_threshold=10,
            holes_connectivity=1, holes_depth=1, remove_objects=True, min_size=0, object_connectivity=1,
            objects_depth=1, fill_holes_first=True, morph_dtype=np.bool_, debug: bool = False
    ) -> (np.array, dict):

        """ identifies events in data based on threshold

        :param data: 3D array with dimensions Z, X, Y of dtype float.
                    expected to be photobleach corrected.
        :param roi_threshold: minimum threshold to be considered an active pixel.
        :param var_estimate: estimated variance of data.
        :param mask_xy: (optional) 2D binary array masking pixels.
        :return:
            event_map: 3D array in which pixels are labelled with event identifier.
        """

        io = IO()

        # threshold data by significance value
        if roi_threshold is not None:
            active_pixels = da.from_array(np.zeros(data.shape, dtype=np.bool_))

            # Abs. threshold is roi_threshold * np.sqrt(var_estimate) when var_estimate != None.
            absolute_threshold = roi_threshold * np.sqrt(var_estimate) if var_estimate is not None else roi_threshold
            # Active pixels are those whose gaussian filter-processed intensities, sigma=smoXY
            # are higher than then calculated absolute threshold.
            active_pixels[:] = ndfilters.gaussian_filter(data, smooth_sigma) > absolute_threshold

        else:
            logging.info("Dynamically choosing filtering threshold (threshold: None)")

            if not isinstance(data, da.Array):
                data = da.from_array(data)

            # 3D smooth
            if use_smoothing:
                data = self._gaussian_smooth_3d(
                    data, sigma=smooth_sigma, radius=smooth_radius, mode='nearest', rechunk=True
                )

                if debug:
                    io.save(self.output_directory.joinpath("debug_smoothed_input.tiff"), data=data)

            # Threshold
            if use_spatial:
                active_pixels_spatial = self._spatial_threshold(
                    data, min_ratio=spatial_min_ratio, threshold_z_depth=spatial_z_depth
                )
            if use_temporal:
                active_pixels_temporal = self._temporal_threshold(
                    data, prominence=temporal_prominence, width=temporal_width, rel_height=temporal_rel_height,
                    wlen=temporal_wlen, plateau_size=temporal_plateau_size
                )

            if use_spatial and use_temporal:

                if comb_type == "&":
                    active_pixels = np.minimum(active_pixels_spatial, active_pixels_temporal)
                elif comb_type == "|":
                    active_pixels = np.maximum(active_pixels_spatial, active_pixels_temporal)
                else:
                    raise ValueError(f"please provide comb_type as one of: ('&', '|')")

                active_pixels = active_pixels.astype(morph_dtype)

            elif use_spatial:
                active_pixels = active_pixels_spatial

            elif use_temporal:
                active_pixels = active_pixels_temporal
            else:
                raise ValueError(f"please choose at least one of (use_spatial, use_temporal) or roi_threshold ")

            if debug:
                io.save(self.output_directory.joinpath("debug_active_pixels.tiff"), data=active_pixels)

        logging.info("identified active pixels")

        # mask inactive pixels (accelerates subsequent computation)
        if exclude_border > 0:

            if mask_xy is None:
                mask_xy = np.ones((active_pixels.shape[1], active_pixels.shape[2]), dtype=morph_dtype)

            mask_xy[:exclude_border, :] = 0
            mask_xy[-exclude_border:, :] = 0
            mask_xy[:, :exclude_border] = 0
            mask_xy[:, -exclude_border:] = 0

        if mask_xy is not None:
            np.multiply(active_pixels, mask_xy, out=active_pixels)
            active_pixels = active_pixels.astype(morph_dtype)
            logging.info("Masked inactive pixels")

        # Morphological operation
        if fill_holes and remove_objects:

            if fill_holes_first:
                active_pixels = self._fill_holes(
                    active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity, depth=holes_depth,
                    dtype=np.bool_
                )
                active_pixels = self._remove_objects(
                    active_pixels, min_size=min_size, connectivity=object_connectivity, depth=objects_depth,
                    dtype=morph_dtype
                )

                logging.info("Applied morphologic operations")

            else:
                active_pixels = self._remove_objects(
                    active_pixels, min_size=min_size, connectivity=object_connectivity, depth=objects_depth,
                    dtype=morph_dtype
                )
                active_pixels = self._fill_holes(
                    active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity, depth=holes_depth,
                    dtype=morph_dtype
                )
                logging.info("Applied morphologic operations")

        elif fill_holes:
            active_pixels = self._fill_holes(
                active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity, depth=holes_depth,
                dtype=morph_dtype
            )
            logging.info("Applied morphologic operations")

        elif remove_objects:
            active_pixels = self._remove_objects(
                active_pixels, min_size=min_size, connectivity=object_connectivity, depth=objects_depth,
                dtype=morph_dtype
            )
            logging.info("Applied morphologic operations")

        if debug and (fill_holes or remove_objects):
            io.save(self.output_directory.joinpath("debug_active_pixels_morphed.tiff"), data=active_pixels)

        # label connected pixels
        event_map = da.from_array(np.zeros(data.shape, dtype=int))
        event_map[:], num_events = ndimage.label(active_pixels)
        logging.info("labelled connected pixel. #events: {}".format(num_events))

        # characterize each event
        event_map = event_map.astype(int)

        logging.info("event_map dype: {}".format(event_map.dtype))

        return event_map

    @staticmethod
    def _gaussian_smooth_3d(
            arr, sigma=3, radius=2, mode='nearest', rechunk=True, chunks=('auto', 'auto', 'auto')
    ):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        if rechunk:
            arr = arr.rechunk(chunks)

        depth = {i: radius * 2 + 1 for i in range(3)}
        overlap = da.overlap.overlap(arr, depth=depth, boundary=mode)

        if LooseVersion(scipy.__version__) >= LooseVersion("1.10.0"):
            parameters = dict(sigma=sigma, radius=radius, mode=mode)
        else:
            logging.warning(f"scipy version {scipy.__version__} < 1.10.x. Ignoring radius parameter.")
            parameters = dict(sigma=sigma, mode=mode)

        mapped = overlap.map_blocks(
            lambda x: ndimage.gaussian_filter(x, **parameters), dtype=arr.dtype
        )
        arr = da.overlap.trim_internal(mapped, depth, boundary=mode)

        return arr

    @staticmethod
    def _spatial_threshold(arr, min_ratio=1, threshold_z_depth=1):

        def threshold(arr, min_ratio=min_ratio, depth=threshold_z_depth):

            Z, X, Y = arr.shape
            binary_mask = np.zeros(arr.shape, dtype=np.bool_)

            for i in range(depth, Z - depth):
                z0, z1 = i - depth, i + depth + 1
                arr_s = arr[z0:z1, :, :]

                # calculate threshold
                threshold = threshold_triangle(arr_s)

                # threshold image
                center_index = (len(arr_s) - 1) // 2
                imc = arr_s[center_index, :, :]
                binary_mask_s = imc > threshold

                # calculate ratio foreground/background
                active_ind = np.where(binary_mask_s == 1)  # TODO more efficient solution?
                inactive_ind = np.where(binary_mask_s == 0)

                fg = np.mean(imc[active_ind])
                bg = abs(np.mean(imc[inactive_ind]))
                ratio = fg / bg

                if ratio > min_ratio:
                    binary_mask[i, :, :] = binary_mask_s

            return binary_mask

        data = arr.rechunk((1, -1, -1))
        depth = {0: threshold_z_depth, 1: 0, 2: 0}  # (threshold_z_depth, 0, 0)

        binary_mask = data.map_overlap(threshold, boundary="nearest", depth=depth, trim=True, dtype=np.bool_)

        return binary_mask

    @staticmethod
    def _temporal_threshold(arr, prominence=10, width=3, rel_height=0.9, wlen=60, plateau_size=None):

        """

        :param arr: numpy.ndarray or da.Array
        :param prominence:prominencenumber or ndarray or sequence, optional
                Required prominence of peaks. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required prominence.
        :param width: number or ndarray or sequence, optional
                Required width of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required width.
        :param rel_height: float, optional
                Used for calculation of the peaks width, thus it is only used if width is given. See argument rel_height in peak_widths for a full description of its effects.
        :param wlen: int, optional
                Used for calculation of the peaks prominences, thus it is only used if one of the arguments prominence or width is given. See argument wlen in peak_prominences for a full description of its effects.
        :param plateau_size: number or ndarray or sequence, optional
                Required size of the flat top of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied as the maximal required plateau size.

        :return: binary mask
        """

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        arr = arr.rechunk((-1, "auto", "auto"))

        def find_peaks(
                arr, prominence=prominence, width=width, rel_height=rel_height, wlen=wlen, plateau_size=plateau_size
        ):

            binary_mask = np.zeros(arr.shape, dtype=int)

            _, X, Y = arr.shape
            for x in range(X):
                for y in range(Y):

                    peaks, prominences = signal.find_peaks(
                        arr[:, x, y], prominence=prominence, wlen=wlen, width=width, rel_height=rel_height,
                        plateau_size=plateau_size
                    )

                    for (left, right, prom) in list(
                            zip(prominences['left_ips'], prominences['right_ips'], prominences['prominences'])
                    ):
                        binary_mask[int(left):int(right), x, y] = 1  # prom

            return binary_mask

        binary_mask = da.map_blocks(find_peaks, arr, dtype=int)
        return binary_mask

    @staticmethod
    def _remove_objects(arr, min_size=10, connectivity=1, depth=0, dtype=np.bool_):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        arr = arr.astype(dtype)

        arr = arr.rechunk(("auto", -1, -1))
        depth_dict = {0: 1 + depth, 1: 0, 2: 0}

        def rm_small(frame):

            Z, X, Y = frame.shape
            binary_mask = np.zeros(frame.shape, dtype=dtype)

            for i in range(depth, Z - depth):
                z0, z1 = i - depth, i + depth + 1
                removed = morphology.remove_small_objects(
                    frame[z0:z1, :, :], min_size=min_size, connectivity=connectivity
                )
                binary_mask[i, :, :] = removed[depth:-depth, :, :]

            return binary_mask

        binary_mask = arr.map_overlap(rm_small, boundary="nearest", depth=depth_dict, trim=True, dtype=dtype)

        return binary_mask

    @staticmethod
    def _fill_holes(arr, area_threshold=10, connectivity=1, depth=0, dtype=np.bool_):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        arr = arr.astype(dtype)
        arr = arr.rechunk(("auto", -1, -1))
        depth_dict = {0: 1 + depth, 1: 0, 2: 0}

        def rm_small(frame):

            Z, X, Y = frame.shape
            binary_mask = np.zeros(frame.shape, dtype=dtype)

            for i in range(depth, Z - depth):
                z0, z1 = i - depth, i + depth + 1
                removed = morphology.remove_small_holes(
                    frame[z0:z1, :, :], area_threshold=area_threshold, connectivity=connectivity
                )
                binary_mask[i, :, :] = removed[depth:-depth, :, :]

            return binary_mask

        binary_mask = arr.map_overlap(rm_small, boundary="nearest", depth=depth_dict, trim=True, dtype=dtype)

        return binary_mask

    @staticmethod
    def _get_time_map(event_map, chunk: int = 200):

        time_map = np.zeros((event_map.shape[0], np.max(event_map) + 1), dtype=np.bool_)

        Z = event_map.shape[0]
        if type(event_map) == da.core.Array:
            for c in tqdm(range(0, Z, chunk)):

                cmax = min(Z, c + chunk)
                event_map_memory = event_map[c:cmax, :, :].compute()

                for z in range(c, cmax):
                    time_map[z, np.unique(event_map_memory[z - c, :, :])] = 1

        else:

            logging.warning("Assuming event_map is in RAM. Otherwise slow execution.")
            for z in tqdm(range(Z)):
                time_map[z, np.unique(event_map[z, :, :])] = 1

        return time_map

    def _custom_slim_features(self, time_map, event_path, split_events: bool = True, parallel=True,
                              dashboard_address=None):

        io = IO()

        # define event output path
        combined_path = event_path.parent.joinpath("events.npy")
        if combined_path.is_file():
            print("combined event path already exists! moving on ...")
            return True

        # create shared memory array for the fluorescence data
        logging.info("Creating shared memory arrays...")
        data_shape, _, data_dtype = get_data_dimensions(self.data, return_dtype=True)
        # Calculate n_bytes needed for data.
        n_bytes = data_shape[0] * data_shape[1] * data_shape[2] * data_dtype.itemsize
        # Create shared buffer
        data_sh = shared_memory.SharedMemory(create=True, size=n_bytes)
        # Buffer to array
        data_ = np.ndarray(data_shape, data_dtype, buffer=data_sh.buf)
        data_[:] = self.data[:]
        # save data info for use in task
        data_info = (data_shape, data_dtype, data_sh.name)

        # create shared memory array for the event map
        event_shape, event_chunksize, event_dtype = get_data_dimensions(event_path, return_dtype=True)
        event_map = io.load(event_path, lazy=False)
        n_bytes = event_shape[0] * event_shape[1] * event_shape[2] * event_dtype.itemsize
        event_sh = shared_memory.SharedMemory(create=True, size=n_bytes)
        event_ = np.ndarray(event_shape, event_dtype, buffer=event_sh.buf)
        event_[:] = event_map[:]
        event_info = (event_shape, event_dtype, event_sh.name)
        del event_map

        logging.info("data_.dtype: {}".format(data_.dtype))
        logging.info("event_.dtype: {}".format(event_.dtype))

        # collecting tasks
        logging.info("Collecting tasks...")

        e_start = np.argmax(time_map, axis=0)
        e_stop = time_map.shape[0] - np.argmax(time_map[::-1, :], axis=0)

        out_path = event_path.parent.joinpath("events/")
        if not out_path.is_dir():
            os.mkdir(out_path)

        # push tasks to client
        e_ids = list(range(1, len(e_start)))

        logging.info("#tasks: {}".format(len(e_ids)))
        random.shuffle(e_ids)
        futures = []

        if parallel:

            with Client(
                    memory_limit='auto', processes=False, silence_logs=logging.WARN,
                    dashboard_address=dashboard_address,
            ) as client:
                for e_id in e_ids:
                    futures.append(
                        client.submit(
                            self.characterize_event,  # function to run
                            e_id, e_start[e_id], e_stop[e_id],  # event_id, start and stop frame of event
                            data_info,  # information about data array (shape, dtype, name/path)
                            event_info,  # information about event_mask (shape, dtype, name/path)
                            out_path,  # path to save result to
                            split_events,  # try to decode overlapping events
                            False  # use mmap `use_on_disk_sharing`
                        )
                    )
                progress(futures)

                client.gather(futures)
                # client.close()

        else:

            for event_id in e_ids:
                npy_path = self.characterize_event(
                    event_id, t0=e_start[event_id], t1=e_stop[event_id], data_info=data_info, event_info=event_info,
                    out_path=out_path.as_posix(), split_events=split_events, use_on_disk_sharing=False
                )
                futures.append(npy_path)

        # close shared memory
        try:
            data_sh.close()
            data_sh.unlink()

            event_sh.close()
            event_sh.unlink()
        except FileNotFoundError as err:
            print("An error occured during shared memory closing: ")
            print(err)

        # combine results
        events = {}
        for e in os.listdir(out_path):
            events.update(np.load(out_path.joinpath(e), allow_pickle=True)[()])
        np.save(combined_path, events)
        shutil.rmtree(out_path)

        return events

    def _custom_slim_features_mmap(self, time_map, event_path, split_events: bool = True, parallel=True):
        """
        Custom feature extraction using memory-mapped files for data sharing.

        Args:
            time_map: Time map for the events.
            event_path: Path to the event data.
            split_events: Flag to split events.
            parallel: Flag to run in parallel mode.

        Returns:
            Extracted events.
        """
        io = IO()

        # define event output path
        combined_path = event_path.parent.joinpath("events.npy")
        if combined_path.is_file():
            print("combined event path already exists! moving on ...")
            return True

        # Get data dimensions and types for both the data and the event map.
        data_shape, _, data_dtype = get_data_dimensions(self.data, return_dtype=True)
        event_shape, event_chunksize, event_dtype = get_data_dimensions(event_path, return_dtype=True)

        # Set file paths for memory-mapped files.
        data_file_path = self.output_directory.joinpath("data_mmap_file.mmap")
        event_file_path = self.output_directory.joinpath("event_mmap_file.mmap")

        # Create and populate the memory-mapped array for the data.
        data_ = self._create_mmap_array(data_file_path, data_shape, data_dtype)
        data_[:] = self.data[:]
        data_info = [data_shape, data_dtype, data_file_path]

        # Load the event map, create and populate the memory-mapped array for the event map.
        event_map = io.load(event_path, lazy=False)
        event_ = self._create_mmap_array(event_file_path, event_shape, event_dtype)
        event_[:] = event_map[:]
        del event_map
        event_info = [event_shape, event_dtype, event_file_path]

        # collecting tasks
        logging.info("Collecting tasks...")

        e_start = np.argmax(time_map, axis=0)
        e_stop = time_map.shape[0] - np.argmax(time_map[::-1, :], axis=0)

        out_path = event_path.parent.joinpath("events/")
        if not out_path.is_dir():
            os.mkdir(out_path)

        # push tasks to client
        e_ids = list(range(1, len(e_start)))

        logging.info("#tasks: {}".format(len(e_ids)))
        random.shuffle(e_ids)
        futures = []

        if parallel:

            # with Client(memory_limit='auto', processes=False, silence_logs=logging.ERROR) as client:
            with Client(memory_limit='auto', processes=False) as client:
                for e_id in e_ids:
                    futures.append(
                        client.submit(
                            self.characterize_event,  # function to run
                            e_id, e_start[e_id], e_stop[e_id],  # event_id, start and stop frame of event
                            data_info,  # information about data array (shape, dtype, name/path)
                            event_info,  # information about event_mask (shape, dtype, name/path)
                            out_path,  # path to save result to
                            split_events,  # try to decode overlapping events
                            True  # use mmap `use_on_disk_sharing`
                        )
                    )
                progress(futures)

                client.gather(futures)

        else:

            for event_id in e_ids:
                npy_path = self.characterize_event(
                    event_id, t0=e_start[event_id], t1=e_stop[event_id], data_info=data_info, event_info=event_info,
                    out_path=out_path.as_posix(), split_events=split_events, use_on_disk_sharing=True
                )
                futures.append(npy_path)

        # Cleanup: Ensure the memory-mapped files are closed and deleted properly.
        del data_
        del event_
        self.cleanup_mmap(data_file_path)
        self.cleanup_mmap(event_file_path)

        # combine results
        events = {}
        for e in os.listdir(out_path):
            events.update(np.load(out_path.joinpath(e), allow_pickle=True)[()])
        np.save(combined_path, events)
        shutil.rmtree(out_path)

        return events

    @staticmethod
    def _create_mmap_array(file_path: Union[str, Path], shape: Tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
        """
        Create a memory-mapped array from the given file.

        Args:
            file_path: Path to the file used for memory-mapping.
            shape: Shape of the numpy array.
            dtype: Data type of the numpy array.

        Returns:
            A numpy array memory-mapped to the specified file.
        """

        file_mmap = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)

        return file_mmap

    @staticmethod
    def cleanup_mmap(file_path):
        """
        Closes the memory-mapped object and deletes the associated file.

        Args:
            file_path: The file path of the memory-mapped file.
        """

        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.is_file():
            try:
                if platform.system() in ["Windows", "win32"]:
                    time.sleep(5)

                file_path.unlink()
            except PermissionError as err:
                logging.error(f"Unable to delete mmap file due to permissions: {err.strerror}")

    def characterize_event(
            self, event_id: int, t0: int, t1: int, data_info: Tuple[Sequence[int], np.dtype, str],
            event_info: Tuple[Sequence[int], np.dtype, str], out_path: Union[str, Path], split_events: bool = True,
            use_on_disk_sharing: bool = False
    ) -> Union[int, None]:
        # noinspection GrazieInspection
        """
                Characterizes an event by computing various properties and metrics.

                This function analyzes a specific event in a dataset by calculating properties such as bounding box dimensions, area, shape, and signal traces. It supports handling split events and saves the results to a specified path.

                Args:
                    event_id: The unique identifier of the event to characterize.
                    t0: The starting time index for the event.
                    t1: The ending time index for the event.
                    data_info: Information about the data, including shape and type.
                    event_info: Information about the event, including shape and type.
                    out_path: The path where the results will be saved.
                    split_events: Flag to determine if events should be split.
                    use_on_disk_sharing: Flag to toggle between on-disk (mmap) and in-RAM (shared memory) methods.

                .. warning::

                   The `use_on_disk_sharing` parameter enables the use of on-disk memory mapping (mmap) instead of in-RAM
                   shared memory. While this method ensures compatibility in environments where in-RAM sharing (e.g.,
                   Docker containers) may cause crashes, it is generally slower due to disk I/O operations.
                   Use this method if you encounter issues with shared memory, particularly in containerized environments.

                .. note::

                    .. list-table:: Event Properties Explained
                        :widths: 10 10 10
                        :header-rows: 1

                        *   - Property
                            - Brief Description
                            - In-Depth Explanation & Formula
                        *   - z0, z1
                            - Z-index bounds
                            - Start (z0) and end (z1) indices in the z-dimension.
                        *   - x0, x1, y0, y1
                            - XY bounding box
                            - Coordinates defining the bounding box in x (x0, x1) and y (y0, y1) dimensions.
                        *   - dz, dx, dy
                            - Bounding box size
                            - Dimensions of the bounding box: depth (dz), width (dx), and height (dy).
                        *   - v_length
                            - Event length
                            - Length of the event in the z-dimension. Calculated as :math:`z1 - z0`.
                        *   - v_diameter
                            - Event diameter
                            - Diameter of the event. Calculated as :math:`\\sqrt{dx^2 + dy^2}`.
                        *   - v_area
                            - Event area
                            - Total area covered by the event. Calculated as the count of z-indices where event_id is present.
                        *   - v_bbox_pix_num
                            - Bounding box pixel count
                            - Total number of pixels within the bounding box. Calculated as :math:` dz * dx * dy `.
                        *   - mask
                            - Event mask
                            - Binary mask indicating the presence (1) or absence (0) of the event.
                        *   - v_mask_centroid_local
                            - Local centroid
                            - The local centroid coordinates of the event mask. Calculated for each dimension and normalized by the size of the bounding box in the respective dimension. Formula: :math:`\\text{centroid}_{local-i} = \\frac{\\text{centroid}_{local-i}}{d_i}` corresponding to z, x, y dimensions.
                        *   - v_mask_axis_major_length
                            - Major axis length
                            - The length of the major axis of the ellipse that has the same normalized second central moments as the region. ???
                        *   - v_mask_axis_minor_length
                            - Minor axis length
                            - The length of the minor axis of the ellipse that has the same normalized second central moments as the region. ???
                        *   - v_mask_extent
                            - Extent
                            - The ratio of pixels in the region to pixels in the total bounding box. Calculated as :math:`\\frac{\\text{area}}{dx \\times dy \\times dz}`.
                        *   - v_mask_solidity
                            - Solidity
                            - The proportion of the pixels in the convex hull that are also in the region. Calculated as :math:`\\frac{\\text{area}}{\\text{area of convex hull}}`. ???
                        *   - v_mask_area
                            - Area
                            - The number of pixels in the region.
                        *   - v_mask_equivalent_diameter_area
                            - Equivalent diameter
                            - The diameter of a circle with the same area as the region. Calculated as :math:`\\sqrt{\\frac{4 \\times \\text{area}}{\\pi}}`.
                        *   - contours
                            - Event contours
                            - Contours extracted from each frame of the event. ???
                        *   - footprint
                            - 2D event footprint
                            - The 2D representation of the event, capturing its extent in the XY plane.
                        *   - v_fp_<property>
                            - Footprint properties
                            - Properties such as centroid, eccentricity, perimeter calculated from the 2D footprint. ???
                        *   - trace
                            - Signal trace
                            - The average signal intensity of the event across the z-dimension.
                        *   - v_max_height
                            - Maximum trace height
                            - The peak signal intensity in the trace. Calculated as :math:`\\max(\\text{trace}) - \\min(\\text{trace})`.
                        *   - v_max_gradient
                            - Maximum trace gradient
                            - The steepest gradient in the trace. Calculated as :math:`\\max(\\Delta \\text{trace})`.
                        *   - noise_mask_trace
                            - Noise mask trace
                            - The trace calculated from the noise mask area. ???
                        *   - v_noise_mask_mean
                            - Noise mean
                            - The mean value of the noise mask trace. Calculated as :math:`\\mu_{\\text{noise}}`.
                        *   - v_noise_mask_std
                            - Noise standard deviation
                            - The standard deviation of the noise mask trace. Calculated as :math:`\\sigma_{\\text{noise}}`.
                        *   - v_signal_to_noise_ratio
                            - Signal-to-noise ratio
                            - Ratio of signal intensity to noise. Calculated as :math:`\\frac{v_{\\text{max height}}}{\\mu_{\\text{noise}}}`.
                        *   - v_signal_to_noise_ratio_fold
                            - Signal-to-noise fold change
                            - Signal-to-noise ratio adjusted for noise standard deviation. Calculated as :math:`\\frac{(v_{\\text{max height}} - \\mu_{\\text{noise}})}{\\sigma_{\\text{noise}}}`.
                        *   - error
                            - Error flag
                            - Indicates any computational errors during property calculation. `0` for no error, `1` for error.

                Returns:
                    An integer indicating the status (e.g., 2 for existing results) or None if the process completes.
                """

        data_buffer = None
        event_buffer = None

        # check if result already exists
        if not isinstance(out_path, Path):
            out_path = Path(out_path)

        res_path = out_path.joinpath(f"events{event_id}.npy")
        if os.path.isfile(res_path):
            return 2

        # buffer
        t1 = t1 + 1

        # get event map
        e_shape, e_dtype, e_name = event_info
        if use_on_disk_sharing:

            if isinstance(e_name, Path):
                e_name = e_name.as_posix()

            # Access memory-mapped files
            event_map = np.memmap(e_name, dtype=e_dtype, mode='r', shape=e_shape)

        else:
            # Access shared memory
            event_buffer = shared_memory.SharedMemory(name=e_name)
            event_map = np.ndarray(e_shape, e_dtype, buffer=event_buffer.buf)

        event_map = event_map[t0:t1, :, :]

        # select volume with data
        z, x, y = np.where(event_map == event_id)
        gx0, gx1 = np.min(x), np.max(x) + 1
        gy0, gy1 = np.min(y), np.max(y) + 1
        event_map = event_map[:, gx0:gx1, gy0:gy1]

        # index data volume
        d_shape, d_dtype, d_name = data_info
        if use_on_disk_sharing:

            if isinstance(d_name, Path):
                d_name = d_name.as_posix()

            # Access memory-mapped files
            data = np.memmap(d_name, dtype=d_dtype, mode='r', shape=d_shape)

        else:
            # Access shared memory
            data_buffer = shared_memory.SharedMemory(name=d_name)
            data = np.ndarray(d_shape, d_dtype, buffer=data_buffer.buf)

        data = data[t0:t1, gx0:gx1, gy0:gy1]

        if data.shape != event_map.shape:
            raise ValueError(f"data and event_map do not have the same size: {data.shape} vs. {event_map.shape}")

        if split_events:
            mask = np.where(event_map == event_id)
            event_map, _ = self._detect_sub_events(data, mask)

        res = {}
        for em_id in np.unique(event_map):

            if em_id < 1:
                continue

            em_id = int(em_id)
            event_id_key = None
            error = 0
            try:
                event_id_key = f"{event_id}_{em_id}" if split_events else event_id
                res[event_id_key] = {}

                z, x, y = np.where(event_map == em_id)
                z0, z1 = np.min(z), np.max(z) + 1
                x0, x1 = np.min(x), np.max(x) + 1
                y0, y1 = np.min(y), np.max(y) + 1

                # local bounding box + global bounding box
                res[event_id_key]["z0"] = t0 + z0
                res[event_id_key]["z1"] = t0 + z1

                res[event_id_key]["x0"] = gx0 + x0
                res[event_id_key]["x1"] = gx0 + x1

                res[event_id_key]["y0"] = gy0 + y0
                res[event_id_key]["y1"] = gy0 + y1

                # bbox dimensions
                dz, dx, dy = z1 - z0, x1 - x0, y1 - y0
                res[event_id_key]["dz"] = dz
                res[event_id_key]["v_length"] = dz
                res[event_id_key]["dx"] = dx
                res[event_id_key]["dy"] = dy
                res[event_id_key]["v_diameter"] = np.sqrt(dx ** 2 + dy ** 2)

                # area
                res[event_id_key]["v_area"] = len(z)
                res[event_id_key]["v_bbox_pix_num"] = int(dz * dx * dy)

                # shape
                mask = np.ones((dz, dx, dy), dtype=np.bool_)
                mask[(z - z0, x - x0, y - y0)] = 0
                res[event_id_key]["mask"] = np.invert(mask).flatten()

                if dz > 1:

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        try:
                            props = regionprops_table(
                                np.invert(mask).astype(np.uint8),
                                properties=['centroid_local', 'axis_major_length', "axis_minor_length", 'extent',
                                            'solidity', 'area', 'equivalent_diameter_area']
                            )

                            props["centroid_local-0"] = props["centroid_local-0"] / dz
                            props["centroid_local-1"] = props["centroid_local-1"] / dx
                            props["centroid_local-2"] = props["centroid_local-2"] / dy

                            for k in props.keys():
                                res[event_id_key][f"v_mask_{k}"] = props[k][0]

                        except ValueError as err:
                            error = 1
                            logging.warning(f"Error encountered: {err}")

                # contour
                mask_padded = np.pad(
                    np.invert(mask), pad_width=((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0
                )

                contours = []
                for cz in range(1, mask_padded.shape[0] - 1):

                    frame = mask_padded[cz, :, :]

                    # find countours in frame
                    contour_ = find_contours(frame, level=0.9)

                    for contour in contour_:
                        # create z axis
                        z_column = np.zeros((contour.shape[0], 1))
                        z_column[:] = cz

                        # add extra dimension for z axis
                        contour = np.append(z_column, contour, axis=1)

                        # adjust for padding
                        contour = np.subtract(contour, 1)

                        contours.append(contour)

                res[event_id_key]["contours"] = contours

                # calculate footprint features
                fp = np.invert(np.min(mask, axis=0))
                res[event_id_key]["footprint"] = fp.flatten()

                props = regionprops_table(
                    fp.astype(np.uint8),
                    properties=['centroid_local', 'axis_major_length', 'axis_minor_length', 'eccentricity',
                                'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'orientation', 'perimeter',
                                'solidity', 'area', 'area_convex']
                )

                props["cx"] = gx0 + props["centroid_local-0"]
                props["cy"] = gy0 + props["centroid_local-1"]

                props["centroid_local-0"] = props["centroid_local-0"] / dx
                props["centroid_local-1"] = props["centroid_local-1"] / dy

                for k in props.keys():
                    res[event_id_key][f"v_fp_{k}"] = props[k][0]

                # trace
                signal = data[z0:z1, x0:x1, y0:y1]
                masked_signal = np.ma.masked_array(signal, mask)
                temp_trace = np.ma.filled(np.nanmean(masked_signal, axis=(1, 2)), fill_value=0)
                res[event_id_key]["trace"] = temp_trace

                # trace characteristics
                trace = temp_trace
                temp_v_max_height = np.max(trace) - np.min(trace)
                res[event_id_key]["v_max_height"] = temp_v_max_height

                temp_v_max_gradient = np.max(np.diff(trace)) if len(trace) > 1 else np.nan
                res[event_id_key]["v_max_gradient"] = temp_v_max_gradient

                # approximating noise
                masked_noise = np.ma.masked_array(signal, np.invert(mask))
                temp_noise_mask_trace = np.ma.filled(np.nanmean(masked_noise, axis=(1, 2)), fill_value=0)
                res[event_id_key]["noise_mask_trace"] = temp_noise_mask_trace

                # noise characteristics
                noise_mask_trace = temp_noise_mask_trace
                temp_v_noise_mask_mean = np.mean(noise_mask_trace)
                res[event_id_key]["v_noise_mask_mean"] = temp_v_noise_mask_mean

                temp_v_noise_mask_std = np.std(noise_mask_trace)
                res[event_id_key]["v_noise_mask_std"] = temp_v_noise_mask_std

                # signal-to-noise characteristics
                temp_v_signal_to_noise_ratio = abs(temp_v_max_height / temp_v_noise_mask_mean)

                if np.isnan(temp_v_signal_to_noise_ratio) or np.isinf(temp_v_signal_to_noise_ratio):
                    temp_v_signal_to_noise_ratio = np.nan

                res[event_id_key]["v_signal_to_noise_ratio"] = temp_v_signal_to_noise_ratio

                if temp_v_noise_mask_std != 0:
                    temp_v_signal_to_noise_ratio_fold = abs(
                        (temp_v_max_height - temp_v_noise_mask_mean) / temp_v_noise_mask_std
                    )
                    res[event_id_key]["v_signal_to_noise_ratio_fold"] = temp_v_signal_to_noise_ratio_fold
                else:
                    res[event_id_key]["v_signal_to_noise_ratio_fold"] = np.nan

                # clean up
                del signal
                del masked_signal
                del fp
                del mask

                # error messages
                res[event_id_key]["error"] = error

            except ValueError as err:
                logging.error(f"{event_id_key} failed: {err}\n{traceback.format_exc()}")
                res[event_id_key]["error"] = 1

        np.save(res_path.as_posix(), res)

        if not use_on_disk_sharing:
            try:
                data_buffer.close()
                event_buffer.close()

            except FileNotFoundError as err:
                print("An error occured during shared memory closing: {}".format(err))

        else:
            del data
            del event_map

    @staticmethod
    def _detect_sub_events(
            img, mask_indices, sigma: int = 2, min_local_max_distance: int = 5, local_max_threshold: float = 0.5,
            min_roi_frame_area: int = 5, reject_if_original: bool = True
    ):

        mask = np.ones(img.shape, dtype=bool)
        mask[mask_indices] = 0  # TODO does this need to be inverted?
        Z, X, Y = mask.shape

        new_mask = np.zeros((Z, X, Y), dtype="i2")
        last_mask_frame = np.zeros((X, Y), dtype="i2")
        next_event_id = 1
        local_max_container = []

        for cz in range(Z):
            frame_mask = mask[cz, :, :]
            frame_raw = img[cz, :, :] if sigma is None else gaussian(img[cz, :, :], sigma=sigma)
            frame_raw = np.ma.masked_array(frame_raw, frame_mask)

            # Find Local Maxima
            local_maxima = peak_local_max(
                frame_raw, min_distance=min_local_max_distance, threshold_rel=local_max_threshold
            )
            local_maxima = np.array(
                [(lmx, lmy, last_mask_frame[lmx, lmy]) for (lmx, lmy) in zip(local_maxima[:, 0], local_maxima[:, 1])],
                dtype="i2"
            )

            # Try to find global maximum if no local maxima were found
            if len(local_maxima) == 0:

                mask_area = np.sum(frame_mask)  # Look for global max
                glob_max = np.unravel_index(np.argmax(frame_raw), (X, Y))  # Look for global max

                if (mask_area > 0) and (glob_max != (0, 0)):
                    local_maxima = np.array([[glob_max[0], glob_max[1], 0]])
                else:
                    local_max_container.append(local_maxima)
                    last_mask_frame = np.zeros((X, Y), dtype="i2")
                    continue

            # assign new label to new local maxima (maxima with '0' label)
            for i in range(local_maxima.shape[0]):

                if local_maxima[i, 2] == 0:
                    local_maxima[i, 2] = next_event_id
                    next_event_id += 1

            # Local Dropouts
            # sometimes local maxima drop below threshold
            # but the event still exists at lower intensity
            # re-introduce those local maxima if the intensity
            # is above threshold value (mask > 0) and the event
            # area of the previous frame is sufficient (area > min_roi_frame_area)
            last_local_max_labels = np.unique(last_mask_frame)
            curr_local_max_labels = np.unique(local_maxima[:, 2])

            for last_local_max_label in last_local_max_labels:

                if (last_local_max_label != 0) and (last_local_max_label not in curr_local_max_labels):

                    prev_local_maxima = local_max_container[-1]
                    missing_local_maxima = prev_local_maxima[prev_local_maxima[:, 2] == last_local_max_label]
                    prev_area = np.sum(new_mask[cz - 1, :, :] == last_local_max_label)

                    # print("missing peak: ", lp, missing_peak)
                    if (len(missing_local_maxima) < 1) or (prev_area < min_roi_frame_area):
                        continue

                    lmx, lmy, _ = missing_local_maxima[0]
                    if ~ frame_mask[lmx, lmy]:  # check that local max still has ongoing event
                        # print("missing peak: ", missing_peak, missing_peak.shape)
                        local_maxima = np.append(local_maxima, missing_local_maxima, axis=0)

            # Local Maximum Uniqueness
            # When a new local maxima appears in a region that was
            # previously occupied, two local maxima receive the same
            # label. Keep label of local maximum closest to previous maximum
            # and assign all local maxima which are further away with
            # new label.
            local_maxima_labels, local_maxima_counts = np.unique(local_maxima[:, 2], return_counts=True)
            for label, count in zip(local_maxima_labels, local_maxima_counts):

                if count > 1:
                    # find duplicated local maxima
                    duplicate_local_maxima_indices = np.where(local_maxima[:, 2] == label)[0]
                    duplicate_local_maxima = local_maxima[local_maxima[:, 2] == label]  # TODO use index instead

                    # get reference local maximum
                    prev_local_max = local_max_container[-1]
                    ref_local_max = prev_local_max[prev_local_max[:, 2] == label]

                    # euclidean distance
                    distance_to_ref = [np.linalg.norm(local_max_xy - ref_local_max[0, :2]) for local_max_xy in
                                       duplicate_local_maxima[:, :2]]
                    min_dist = np.argmin(distance_to_ref)

                    # relabel all local maxima that are further away
                    to_relabel = list(range(len(distance_to_ref)))
                    del to_relabel[min_dist]
                    for to_rel in to_relabel:
                        dup_index = duplicate_local_maxima_indices[to_rel]
                        local_maxima[dup_index, 2] = next_event_id
                        next_event_id += 1

            # save current detected peaks
            local_max_container.append(local_maxima)

            # Separate overlaying events
            if local_maxima.shape[0] == 1:

                # Single Local Maximum (== global maximum)
                last_mask_frame = np.zeros((X, Y), dtype="i2")
                last_mask_frame[~frame_mask] = local_maxima[0, 2]

            else:
                # Multiple Local Maxima
                # use watershed algorithm to separate multiple overlaying events
                # with location of local maxima as seeds

                # create seeds
                seeds = np.zeros((X, Y))
                for i in range(local_maxima.shape[0]):
                    lmx, lmy, lbl = local_maxima[i, :]
                    seeds[lmx, lmy] = lbl

                # run watershed on inverse intensity image
                basin = -1 * frame_raw
                basin[frame_mask] = 0

                # watershed requires type conversion
                basin = basin.filled(fill_value=0).astype(int)
                seeds = seeds.astype(np.int64)

                last_mask_frame = watershed(basin, seeds).astype("i2")
                last_mask_frame[frame_mask] = 0

            # save results of current run
            new_mask[cz, :, :] = last_mask_frame

        unique_elements = np.unique(new_mask)
        if reject_if_original & (np.array_equal(unique_elements, [0, 1]) | np.array_equal(unique_elements, [0])):
            # did not split original event
            return ~mask, None
        else:
            return new_mask, local_max_container
