import argparse
import os
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional

import tifffile as tf
import dask.array as da
import random
import shutil
import tiledb
import warnings

from scipy import signal
from skimage import morphology
from skimage.filters import threshold_triangle, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table, find_contours
import scipy.ndimage as ndimage
from dask_image import ndmorph, ndfilters
from dask.distributed import progress
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from multiprocess import shared_memory

from astrocast.preparation import IO


class Detector:

    """
    The detector class provides a method and dependency functions to detect events, estimate background noise, identify methods based on a threshold (user provided or calculated), characterize events...

    Methods:
     - run(self, dataset: Optional[str] = None, threshold: Optional[float] = None, min_size: int = 20,
        lazy: bool = True, adjust_for_noise: bool = False, subset: Optional[str] = None, split_events: bool = True,
        binary_struct_iterations: int = 1, binary_struct_connectivity: int = 2, save_activepixels: bool = False):
        the 'dataset' parameter corresponds to the name or identifier of the dataset in the h5 file.
        the 'threshold' value to discriminate background from events. If None, automatic thresholding is performed.
         the 'min_size' refers to Minimum size of an event region. Events with size < min_size will be excluded.
        the 'lazy' parameter, if set to True, lazy loading is implemented.
        the 'adjust_for_noise', indicates if event detection for background noise adjustment must be carried out.
        the 'subset' indicates which subset of the dataset to process.
        the 'split_events' indicates  Whether to split detected events into smaller events
                if multiple peaks are detected.
        binary_struct_iterations (int): Number of iterations for binary structuring element.
        binary_struct_connectivity (int): Connectivity of binary structuring element.

    """
    def __init__(self, input_path: str, output=None, logging_level=logging.INFO):

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
        self.file = None
        self.data = None
        self.Z, self.X, self.Y = None, None, None
        self.meta = {}

    def run(self, h5_loc: Optional[str] = None, exclude_border=0,
            threshold: Optional[float] = None,
            use_smoothing=True, smooth_radius=2, smooth_sigma=2,
            use_spatial=True, spatial_min_ratio=1, spatial_z_depth=1,
            use_temporal=True, temporal_prominence=10, temporal_width=3, temporal_rel_height=0.9, temporal_wlen=60,
            temporal_plateau_size=None,
            comb_type="&",
            fill_holes=True, area_threshold=10, holes_connectivity=1, holes_depth=1,
            remove_objects=True, min_size=20, object_connectivity=1, objects_depth=1,
            fill_holes_first=True,
            lazy: bool = True, adjust_for_noise: bool = False,
            subset: Optional[str] = None, split_events: bool = False,
            debug: bool = False,
            parallel=True
            ) -> None:
        """
        Runs the event detection process on the specified dataset.

        Args:
            h5_loc (Optional[str]): Name or identifier of the dataset in the h5 file.
            threshold (Optional[float]): Threshold value to discriminate background from events. 
                If None, automatic thresholding is performed.
            min_size (int): Minimum size of an event region. 
                Events with size < min_size will be excluded.
            lazy (bool): Whether to implement lazy loading.
            adjust_for_noise (bool): Whether to adjust event detection for background noise.
            subset (Optional[str]): Subset of the dataset to process.
            split_events (bool): Whether to split detected events into smaller events 
                if multiple peaks are detected.
            binary_struct_iterations (int): Number of iterations for binary structuring element. 
            binary_struct_connectivity (int): Connectivity of binary structuring element.
            parallel: parallel execution of event characterization. Recommended to be true.

        Returns:
            dictionary of events

        Notes:
            - The output and intermediate results are stored in the output directory.
            - If the output directory does not exist, it will be created.
            - The event map and time map are saved as numpy arrays.
            - The detected events are processed to calculate features.
            - The metadata is saved in a JSON file.
        """

        self.meta.update({
            "subset": subset,
            "threshold": threshold,
            "min_size": min_size,
            "adjust_for_noise": adjust_for_noise,
        })

        # output folder
        self.output_directory = self.output if self.output is not None \
            else self.input_path.with_suffix(".roi") if h5_loc is None \
            else self.input_path.with_suffix(".{}.roi".format(h5_loc.split("/")[-1]))

        if not self.output_directory.is_dir():
            self.output_directory.mkdir()
        logging.info(f"output directory: {self.output_directory}")

        # profiling
        pbar = ProgressBar(minimum=10)
        pbar.register()

        # load data
        io = IO()
        # todo: add chunks flag and/or re-chunking
        data = io.load(path=self.input_path, h5_loc=h5_loc, z_slice=subset, lazy=lazy)
        self.Z, self.X, self.Y = data.shape
        self.data = data
        logging.info(f"data: {data.shape}") if lazy else logging.info(f"data: {data}")

        # calculate event map
        event_map_path = self.output_directory.joinpath("event_map.tdb")
        if not os.path.isdir(event_map_path):
            logging.info("Estimating noise")
            # TODO maybe should be adjusted since it might already be calculated
            noise = self.estimate_background(data) if adjust_for_noise else 1

            logging.info("Thresholding events")
            event_map = self.get_events(data, exclude_border=exclude_border, roi_threshold=threshold, var_estimate=noise,
                                        use_smoothing=use_smoothing, smooth_radius=smooth_radius, smooth_sigma=smooth_sigma,
                                        use_spatial=use_spatial, spatial_min_ratio=spatial_min_ratio, spatial_z_depth=spatial_z_depth,
                                        use_temporal=use_temporal, temporal_prominence=temporal_prominence, temporal_width=temporal_width,
                                        temporal_rel_height=temporal_rel_height, temporal_wlen=temporal_wlen, temporal_plateau_size=temporal_plateau_size,
                                        comb_type=comb_type,
                                        fill_holes=fill_holes, area_threshold=area_threshold, holes_connectivity=holes_connectivity, holes_depth=holes_depth,
                                        remove_objects=remove_objects, min_size=min_size, object_connectivity=object_connectivity, objects_depth=objects_depth,
                                        fill_holes_first=fill_holes_first, morph_dtype=np.bool_,
                                        debug=debug
                                        )

            logging.info(f"Saving event map to: {event_map_path}")
            event_map.rechunk((100, 100, 100)).to_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            logging.info(f"Saving tiff to: {tiff_path}")
            tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        else:
            logging.info(f"Loading event map from: {event_map_path}")
            event_map = da.from_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            if not tiff_path.is_file():
                logging.info(f"Saving tiff to: {tiff_path}")
                tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        # calculate time map
        logging.info("Calculating time map")
        time_map_path = self.output_directory.joinpath("time_map.npy")
        if not time_map_path.is_file():
            time_map = self.get_time_map(event_map)

            logging.info(f"Saving event map to: {time_map_path}")
            np.save(time_map_path, time_map)
        else:
            logging.info(f"Loading time map from {time_map_path}")
            time_map = np.load(time_map_path.as_posix())

        # calculate features
        logging.info("Calculating features")
        events = self.custom_slim_features(time_map, event_map_path, split_events=split_events, parallel=parallel)

        logging.info("Saving features")
        with open(self.output_directory.joinpath("meta.json"), 'w') as outfile:
            json.dump(self.meta, outfile)

        logging.info("Run complete! [{}]".format(self.input_path))

        return events

    @staticmethod
    def estimate_background(data: np.array, mask_xy: np.array = None) -> float:

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

    def get_events(self, data: np.array, roi_threshold: float, var_estimate: float,
                   mask_xy: np.array = None, exclude_border=0,
                   use_smoothing=True, smooth_radius = 2, smooth_sigma = 2,
                   use_spatial=True, spatial_min_ratio=1, spatial_z_depth=1,
                   use_temporal=True, temporal_prominence=10, temporal_width=3, temporal_rel_height=0.9, temporal_wlen=60, temporal_plateau_size=None,
                   comb_type="&",
                   fill_holes=True, area_threshold=10, holes_connectivity=1, holes_depth=1,
                   remove_objects=True, min_size=0, object_connectivity=1, objects_depth=1,
                   fill_holes_first=True, morph_dtype=np.bool_,
                   debug: bool = False) -> (np.array, dict):

        """ identifies events in data based on threshold

        :param data: 3D array with dimensions Z, X, Y of dtype float.
                    expected to be photobleach corrected.
        :param roi_threshold: minimum threshold to be considered an active pixel.
        :param var_estimate: estimated variance of data.
        :param min_roi_size: minimum size of active regions of interest.
        :param mask_xy: (optional) 2D binary array masking pixels.
        :return:
            event_map: 3D array in which pixels are labelled with event identifier.
        """

        io = IO()

        # threshold data by significance value
        if roi_threshold is not None:
            active_pixels = da.from_array(np.zeros(data.shape, dtype=np.bool_))

            # Abs. threshold is roi_threshold * np.sqrt(var_estimate) when var_estimate != None.
            absolute_threshold = roi_threshold * np.sqrt(var_estimate) \
                if var_estimate is not None else roi_threshold
            # Active pixels are those whose gaussian filter-processed intensities, sigma=smoXY
            # are higher than then calculated absolute threshold.
            active_pixels[:] = ndfilters.gaussian_filter(data, smooth_sigma) > absolute_threshold

        else:
            logging.info("Dynamically choosing filtering threshold (threshold: None)")

            if not isinstance(data, da.Array):
                data = da.from_array(data)

            # 3D smooth
            if use_smoothing:
                data = self.gaussian_smooth_3d(data, sigma=smooth_sigma, radius=smooth_radius, mode='nearest', rechunk=True)

                if debug:
                    io.save(self.output_directory.joinpath("debug_smoothed_input.tiff"), data=data)

            # Threshold
            if use_spatial:
                active_pixels_spatial = self.spatial_threshold(data, min_ratio=spatial_min_ratio,
                                                               threshold_z_depth=spatial_z_depth)
            if use_temporal:
                active_pixels_temporal = self.temporal_threshold(data, prominence=temporal_prominence, width=temporal_width, rel_height=temporal_rel_height,
                                                                 wlen=temporal_wlen, plateau_size=temporal_plateau_size)

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
                active_pixels = self.fill_holes(active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity,
                                                depth=holes_depth, dtype=np.bool_)
                active_pixels = self.remove_objects(active_pixels, min_size=min_size, connectivity=object_connectivity,
                                                depth=objects_depth, dtype=morph_dtype)

                logging.info("Applied morphologic operations")

            else:
                active_pixels = self.remove_objects(active_pixels, min_size=min_size, connectivity=object_connectivity,
                                                depth=objects_depth, dtype=morph_dtype)
                active_pixels = self.fill_holes(active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity,
                                                depth=holes_depth, dtype=morph_dtype)
                logging.info("Applied morphologic operations")

        elif fill_holes:
            active_pixels = self.fill_holes(active_pixels, area_threshold=area_threshold, connectivity=holes_connectivity,
                                                depth=holes_depth, dtype=morph_dtype)
            logging.info("Applied morphologic operations")

        elif remove_objects:
            active_pixels = self.remove_objects(active_pixels, min_size=min_size, connectivity=object_connectivity,
                                            depth=objects_depth, dtype=morph_dtype)
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
    def gaussian_smooth_3d(arr, sigma=3, radius=2, mode='nearest',
                           rechunk=True, chunks=('auto', 'auto', 'auto')):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        if rechunk:
            arr = arr.rechunk(chunks)


        depth = {i:radius*2+1 for i in range(3)}
        overlap = da.overlap.overlap(arr, depth=depth, boundary=mode)
        mapped = overlap.map_blocks(lambda x: ndimage.gaussian_filter(x,
                                                                      sigma=sigma, radius=radius, mode=mode),
                                    dtype=arr.dtype)
        arr = da.overlap.trim_internal(mapped, depth, boundary=mode)

        return arr

    @staticmethod
    def spatial_threshold(arr, min_ratio=1, threshold_z_depth=1):

        def threshold(arr, min_ratio=min_ratio, depth=threshold_z_depth):

                Z, X, Y = arr.shape
                binary_mask = np.zeros(arr.shape, dtype=np.bool_)

                for i in range(depth, Z-depth):
                    z0, z1 = i-depth, i+depth+1
                    arr_s = arr[z0:z1, :, :]

                    # calculate threshold
                    threshold = threshold_triangle(arr_s)

                    # threshold image
                    center_index = (len(arr_s) - 1) // 2
                    imc = arr_s[center_index, :, :]
                    binary_mask_s = imc > threshold

                    # calculate ratio foreground/background
                    active_ind = np.where(binary_mask_s == 1) # TODO more efficient solution?
                    inactive_ind = np.where(binary_mask_s == 0)

                    fg = np.mean(imc[active_ind])
                    bg = abs(np.mean(imc[inactive_ind]))
                    ratio = fg/bg

                    if ratio > min_ratio:
                        binary_mask[i, :, :] = binary_mask_s

                return binary_mask

        data = arr.rechunk((1, -1, -1))
        depth = {0:threshold_z_depth, 1:0, 2:0} #(threshold_z_depth, 0, 0)

        binary_mask = data.map_overlap(threshold, boundary="nearest", depth=depth, trim=True, dtype=np.bool_)

        return binary_mask

    @staticmethod
    def temporal_threshold(arr,  prominence=10, width=3, rel_height=0.9, wlen=60, plateau_size=None):

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

        def find_peaks(arr, prominence=prominence, width=width, rel_height=rel_height, wlen=wlen, plateau_size=plateau_size):

            binary_mask = np.zeros(arr.shape, dtype=int)

            _, X, Y = arr.shape
            for x in range(X):
                for y in range(Y):

                    peaks, prominences = signal.find_peaks(arr[:, x, y], prominence=prominence, wlen=wlen,
                                                           width=width, rel_height=rel_height, plateau_size=plateau_size)


                    for (left, right, prom) in list(zip(prominences['left_ips'], prominences['right_ips'], prominences['prominences'])):
                        binary_mask[int(left):int(right), x, y] = 1 # prom

            return binary_mask

        binary_mask = da.map_blocks(find_peaks, arr, dtype=int)
        return binary_mask

    @staticmethod
    def remove_objects(arr, min_size=10, connectivity=1, depth=0, dtype=np.bool_):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        arr = arr.astype(dtype)

        arr = arr.rechunk(("auto", -1, -1))
        depth_dict = {0: 1 + depth, 1: 0, 2: 0}

        def rm_small(frame):

            Z, X, Y = frame.shape
            binary_mask = np.zeros(frame.shape, dtype=dtype)

            for i in range(depth, Z-depth):
                z0, z1 = i-depth, i+depth+1
                removed = morphology.remove_small_objects(frame[z0:z1, :, :],
                                                          min_size=min_size, connectivity=connectivity)
                binary_mask[i, :, :] = removed[depth:-depth, :, :]

            return binary_mask

        binary_mask = arr.map_overlap(rm_small, boundary="nearest", depth=depth_dict, trim=True, dtype=dtype)

        return binary_mask

    @staticmethod
    def fill_holes(arr, area_threshold=10, connectivity=1, depth=0, dtype=np.bool_):

        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        arr = arr.astype(dtype)
        arr = arr.rechunk(("auto", -1, -1))
        depth_dict = {0: 1 + depth, 1: 0, 2: 0}

        def rm_small(frame):

            Z, X, Y = frame.shape
            binary_mask = np.zeros(frame.shape, dtype=dtype)

            for i in range(depth, Z-depth):
                z0, z1 = i-depth, i+depth+1
                removed = morphology.remove_small_holes(frame[z0:z1, :, :],
                                                        area_threshold=area_threshold, connectivity=connectivity)
                binary_mask[i, :, :] = removed[depth:-depth, :, :]

            return binary_mask

        binary_mask = arr.map_overlap(rm_small, boundary="nearest", depth=depth_dict, trim=True, dtype=dtype)

        return binary_mask

    @staticmethod
    def get_time_map(event_map, chunk: int = 200):

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

    def custom_slim_features(self, time_map, event_path, split_events: bool = True, parallel=True):

        # print(event_map)
        # sh_em = shared_memory.SharedMemory(create=True, size=event_map.nbytes)
        # shn_em = np.ndarray(event_map.shape, dtype=event_map.dtype, buffer=sh_em.buf)
        # shn_em[:] = event_map
        #
        # num_events = np.max(shn_em)

        # create chunks

        # shared memory

        combined_path = event_path.parent.joinpath("events.npy")
        if combined_path.is_file():
            print("combined event path already exists! moving on ...")
            return True

        # self.vprint("creating shared memory arrays ...", 3)
        logging.info("Creating shared memory arrays...")
        data = self.data
        # Calculate n_bytes needed for data.
        n_bytes = data.shape[0] * data.shape[1] * data.shape[2] * data.dtype.itemsize
        # Create shared buffer
        data_sh = shared_memory.SharedMemory(create=True, size=n_bytes)
        # Buffer to array
        data_ = np.ndarray(data.shape, data.dtype, buffer=data_sh.buf)
        data_[:] = data[:]
        # save data info for use in task
        data_info = (data.shape, data.dtype, data_sh.name)

        event = tiledb.open(event_path.as_posix())
        n_bytes = event.shape[0] * event.shape[1] * event.shape[2] * event.dtype.itemsize
        event_sh = shared_memory.SharedMemory(create=True, size=n_bytes)
        event_ = np.ndarray(event.shape, event.dtype, buffer=event_sh.buf)
        event_[:] = event[:]
        event_info = (event.shape, event.dtype, event_sh.name)
        del event

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

            with Client(memory_limit='auto', processes=False,
                        silence_logs=logging.ERROR) as client:
                for e_id in e_ids:
                    futures.append(
                        client.submit(
                            self.characterize_event,
                            e_id, e_start[e_id], e_stop[e_id],
                            data_info, event_info, out_path, split_events
                        )
                    )
                progress(futures)

                client.gather(futures)

        else:

            for event_id in e_ids:
                npy_path = self.characterize_event(event_id,
                                                   t0=e_start[event_id], t1=e_stop[event_id],
                                                   data_info=data_info, event_info=event_info,
                                                   out_path=out_path.as_posix(), split_events=split_events)
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

    def characterize_event(self, event_id, t0, t1, data_info,
                           event_info, out_path, split_events=True):

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
        data_buffer = shared_memory.SharedMemory(name=d_name)
        data = np.ndarray(d_shape, d_dtype, buffer=data_buffer.buf)
        data = data[t0:t1, gx0:gx1, gy0:gy1]

        if data.shape != event_map.shape:
            raise ValueError(f"data and event_map do not have the same size: {data.shape} vs. {event_map.shape}")

        if split_events:
            mask = np.where(event_map == event_id)
            event_map, _ = self.detect_subevents(data, mask)

        res = {}
        for em_id in np.unique(event_map):

            if em_id < 1:
                continue

            em_id = int(em_id)
            event_id_key = None
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
                res[event_id_key]["dx"] = dx
                res[event_id_key]["dy"] = dy

                # area
                res[event_id_key]["area"] = len(z)
                res[event_id_key]["bbox_pix_num"] = int(dz * dx * dy)

                # shape
                mask = np.ones((dz, dx, dy), dtype=np.bool_)
                mask[(z - z0, x - x0, y - y0)] = 0
                res[event_id_key]["mask"] = np.invert(mask).flatten()

                if dz > 1:

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        try:
                            props = regionprops_table(np.invert(mask).astype(np.uint8),
                                                      properties=['centroid_local', 'axis_major_length',
                                                                  "axis_minor_length", 'extent', 'solidity', 'area',
                                                                  'equivalent_diameter_area'])

                            props["centroid_local-0"] = props["centroid_local-0"] / dz
                            props["centroid_local-1"] = props["centroid_local-1"] / dx
                            props["centroid_local-2"] = props["centroid_local-2"] / dy

                            for k in props.keys():
                                res[event_id_key][f"mask_{k}"] = props[k][0]

                        except ValueError as err:
                            print("\t Error in ", event_id_key)
                            print("\t", err)

                # contour
                mask_padded = np.pad(np.invert(mask), pad_width=((1, 1), (1, 1), (1, 1)), mode="constant",
                                     constant_values=0)

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

                props = regionprops_table(fp.astype(np.uint8), properties=['centroid_local', 'axis_major_length',
                                                                           'axis_minor_length', 'eccentricity',
                                                                           'equivalent_diameter_area', 'extent',
                                                                           'feret_diameter_max', 'orientation',
                                                                           'perimeter', 'solidity', 'area',
                                                                           'area_convex'])

                props["cx"] = gx0 + props["centroid_local-0"]
                props["cy"] = gy0 + props["centroid_local-1"]

                props["centroid_local-0"] = props["centroid_local-0"] / dx
                props["centroid_local-1"] = props["centroid_local-1"] / dy

                for k in props.keys():
                    res[event_id_key][f"fp_{k}"] = props[k][0]

                # trace
                signal = data[z0:z1, x0:x1, y0:y1]
                masked_signal = np.ma.masked_array(signal, mask)
                res[event_id_key]["trace"] = np.ma.filled(np.nanmean(masked_signal, axis=(1, 2)), fill_value=0)

                # clean up
                del signal
                del masked_signal
                del fp
                del mask

                # error messages
                res[event_id_key]["error"] = 0

            except ValueError as err:
                print("\t Error in ", event_id_key)
                print("\t", err)
                res[event_id_key]["error"] = 1

        np.save(res_path.as_posix(), res)

        try:
            data_buffer.close()
            event_buffer.close()

        except FileNotFoundError as err:
            print("An error occured during shared memory closing: {}".format(err))

    @staticmethod
    def detect_subevents(img, mask_indices, sigma: int = 2,
                         min_local_max_distance: int = 5, local_max_threshold: float = 0.5, min_roi_frame_area: int = 5,
                         reject_if_original: bool = True):

        mask = np.ones(img.shape, dtype=bool)
        mask[mask_indices] = 0 # TODO does this need to be inverted?
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
            local_maxima = peak_local_max(frame_raw, min_distance=min_local_max_distance,
                                          threshold_rel=local_max_threshold)
            local_maxima = np.array(
                [(lmx, lmy, last_mask_frame[lmx, lmy]) for (lmx, lmy) in zip(local_maxima[:, 0], local_maxima[:, 1])],
                dtype="i2")

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
