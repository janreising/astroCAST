import argparse, os, logging
import json
from dask.diagnostics import ProgressBar
import numpy as np
from pathlib import Path
from typing import Optional
import tifffile as tf
import dask.array as da
import h5py as h5
import random
import shutil
import tiledb
import warnings
from skimage import measure, morphology  # , segmentation
from skimage.filters import threshold_triangle, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from dask_image import ndmorph, ndfilters, ndmeasure
from dask.distributed import progress
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from tqdm import tqdm
from multiprocess import shared_memory


class Detector:

    def __init__(self, input_path: str, output = None,
                 indices: np.array = None, verbosity: int = 1):

        logging.basicConfig(format = "%(asctime)s %(levelname)s: %(message)s", level = verbosity)

        # paths
        self.input_path = Path(input_path)
        self.output = output if output is None else Path(output)
        working_directory = self.input_path.parent

        logging.info(f"working directory: {working_directory}")
        logging.info(f"input file: {self.input_path}")

        # quality check arguments
        assert os.path.isfile(input_path) or os.path.isdir(input_path), \
            f"input file does not exist: {input_path}"
        assert (output is None) or (~ self.output.is_dir()), \
            f"output file already exists: {output}"
        assert indices is None or indices.shape == (3, 2), \
            "indices must be np.arry of shape (3, 2) -> ((z0, z1), " \
            "(x0, x1), (y0, y1)). Found: " + indices
        # shared variables
        self.file = None
        self.Z, self.X, self.Y = None, None, None
        self.meta = {}

    def run(self, dataset: Optional[str] = None,
            threshold: Optional[float] = None, min_size: int = 20,
            use_dask: bool = False, adjust_for_noise: bool = False,
            subset: Optional[str] = None, split_events: bool = True,
            binary_struct_iterations: int = 1,
            binary_struct_connectivity: int = 2, # TODO better way to do this
            save_activepixels: bool = False) -> None:
        """
        Runs the event detection process on the specified dataset.

        Args:
            dataset (Optional[str]): Name or identifier of the dataset in the h5 file.
            threshold (Optional[float]): Threshold value to discriminate background from events. 
                If None, automatic thresholding is performed.
            min_size (int): Minimum size of an event region. 
                Events with size < min_size will be excluded.
            use_dask (bool): Whether to use Dask for parallel computation.
            adjust_for_noise (bool): Whether to adjust event detection for background noise.
            subset (Optional[str]): Subset of the dataset to process.
            split_events (bool): Whether to split detected events into smaller events 
                if multiple peaks are detected.
            binary_struct_iterations (int): Number of iterations for binary structuring element. 
            binary_struct_connectivity (int): Connectivity of binary structuring element.

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
            else self.input_path.with_suffix(".roi") if dataset is None \
            else self.input_path.with_suffix(
            ".{}.roi".format(dataset.split("/")[-1]))

        if not self.output_directory.is_dir():
            self.output_directory.mkdir()
        logging.info(f"output directory: {self.output_directory}")

        # profiling
        pbar = ProgressBar(minimum = 10)
        pbar.register()

        # TODO save this information somewhere
        #resources = ResourceProfiler()
        #resources.register()
        # load data
        data = self._load(dataset_name = dataset, use_dask = use_dask, subset = subset)
        self.data = data
        self.Z, self.X, self.Y = data.shape
        logging.info(f"data: {data}") if use_dask else logging.info(f"data: {data.shape}")

        #self.vprint(data if use_dask else data.shape, 2)

        # calculate event map
        event_map_path = self.output_directory.joinpath("event_map.tdb")
        if not os.path.isdir(event_map_path):
            logging.info("Estimating noise")
            # self.vprint("Estimating noise", 2)
            # TODO maybe should be adjusted since it might already be calculated
            noise = self.estimate_background(data) if adjust_for_noise else 1

            #self.vprint("Thresholding events", 2)
            logging.info("Thresholding events")
            event_map = self.get_events(data, roi_threshold = threshold, var_estimate = noise,
                                        min_roi_size = min_size,
                                        binary_struct_iterations = binary_struct_iterations,
                                        binary_struct_connectivity = binary_struct_connectivity,
                                        save_activepixels = save_activepixels)

            #self.vprint(f"Saving event map to: {event_map_path}", 2)
            logging.info(f"Saving event map to: {event_map_path}")
            event_map.rechunk((100, 100, 100)).to_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            #self.vprint(f"Saving tiff to : {tiff_path}", 2)
            logging.info(f"Saving tiff to: {tiff_path}")
            tf.imwrite(tiff_path, event_map, dtype = event_map.dtype)

        else:
            #self.vprint(f"Loading event map from: {event_map_path}", 2)
            logging.info(f"Loading event map from: {event_map_path}")
            event_map = da.from_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            if not tiff_path.is_file():
                #self.vprint(f"Saving tiff to : {tiff_path}", 2)
                logging.info(f"Saving tiff to: {tiff_path}")
                tf.imwrite(tiff_path, event_map, dtype = event_map.dtype)

        # calculate time map
        #self.vprint("Calculating time map", 2)
        logging.info("Calculating time map")
        time_map_path = self.output_directory.joinpath("time_map.npy")
        if not time_map_path.is_file():
            time_map = self.get_time_map(event_map)

            #self.vprint(f"Saving event map to: {time_map_path}", 2)
            logging.info(f"Saving event map to: {time_map_path}")
            np.save(time_map_path, time_map)
        else:
            #self.vprint(f"Loading time map from: {time_map_path}", 2)
            logging.info(f"Loading time map from {time_map_path}")
            time_map = np.load(time_map_path.as_posix())

        # calculate features
        #self.vprint("Calculating features", 2)
        logging.info("Calculating features")
        events = self.custom_slim_features(time_map, self.input_path, event_map_path,
                                  split_events = split_events)

        #self.vprint("saving features", 2)
        logging.info("Saving features")
        with open(self.output_directory.joinpath("meta.json"), 'w') as outfile:
            json.dump(self.meta, outfile)

        #self.vprint("Run complete! [{}]".format(self.input_path, 1), 1)
        logging.info("Run complete! [{}]".format(self.input_path))

        return events

    def _load(self, dataset_name: str = None, use_dask: bool = False, subset = None):

        """ loads data from file

        :param dataset_name: name of the dataset in stored in an hdf file
        :param in_memory: flag whether to load full dataset in memory or not
        :return:
        """
        # TODO instead of self reference; prob better to explicitly give path as argument
        if self.input_path.suffix == ".h5":

            assert dataset_name is not None, "'dataset_name' required if providing an hdf file"

            file = h5.File(self.input_path, "r")
            assert dataset_name in file, "dataset '{}' does not exist in file".format(dataset_name)

            data = da.from_array(file[dataset_name], chunks = 'auto') \
                if use_dask else file[dataset_name]

        elif self.input_path.suffix in (".tdb", ".delta"):

            data = tiledb.open(self.input_path.as_posix())

            if use_dask:
                data = da.from_array(data, chunks = 'auto')

        else:
            logging.error(f"unknown file type: {self.input_path}")
            #self.vprint(f"unknown file type: {self.input_path}", 0)

        if subset is not None:
            assert len(subset) == 6, "please provide a subset for all dimensions"
            z0, z1, x0, x1, y0, y1 = subset

            z0 = z0 if z0 is not None else 0
            x0 = x0 if x0 is not None else 0
            y0 = y0 if y0 is not None else 0

            Z, X, Y = data.shape
            z1 = z1 if z1 is not None else Z
            x1 = x1 if x1 is not None else X
            y1 = y1 if y1 is not None else Y

            data = data[z0:z1, x0:x1, y0:y1]

        if not use_dask:
            data = data[:]
        return data
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
                   min_roi_size: int = 10, mask_xy: np.array = None, smoXY = 2,
                   remove_small_object_framewise = False,
                   binary_struct_iterations = 1,
                   binary_struct_connectivity = 2,
                   save_activepixels: bool = False) -> (np.array, dict):

        """ identifies events in data based on threshold

        :param data: 3D array with dimensions Z, X, Y of dtype float.
                    expected to be photobleach corrected.
        :param roi_threshold: minimum threshold to be considered an active pixel.
        :param var_estimate: estimated variance of data.
        :param min_roi_size: minimum size of active regions of interest.
        :param mask_xy: (optional) 2D binary array masking pixels.
        :return:
            event_map: 3D array in which pixels are labelled with event identifier.
            event_properties: list of scipy.regionprops items.
        """

        # threshold data by significance value
        if roi_threshold is not None:
            active_pixels = da.from_array(np.zeros(data.shape, dtype = np.bool_))
            # self.vprint("Noise threshold:
            # {:.2f}".format(roi_threshold * np.sqrt(var_estimate)), 4)

            # Abs. threshold is roi_threshold * np.sqrt(var_estimate) when var_estimate != None.
            absolute_threshold = roi_threshold * np.sqrt(var_estimate) \
                if var_estimate is not None else roi_threshold
            # Active pixels are those whose gaussian filter-processed intensities, sigma=smoXY
            # are higher than then calculated absolute threshold.
            active_pixels[:] = ndfilters.gaussian_filter(data, smoXY) > absolute_threshold

        else:
            # self.vprint("no threshold defined. Using skimage.threshold
            # to define threshold dynamically ...", 3)
            logging.warning("no threshold defined. Using skimage.threshold \
                            to define threshold dynamically...")
            # Executed when a roi_threshold has not been provided.
            def dynamic_threshold(img): # Add paramters
                smooth = gaussian(img, sigma = smoXY, channel_axis = None)
                thr = 1 if np.sum(img) == 0 else threshold_triangle(smooth)
                img_thr = smooth > thr
                img_thr = img_thr.astype(np.bool_)

                return img_thr

            data_rechunked = data.rechunk((1, -1, -1))
            active_pixels = data_rechunked.map_blocks(dynamic_threshold, dtype = np.bool_)

            # TODO add flag whether to save or not
            if save_activepixels == True:
                tiff_path = self.output_directory.joinpath("active_pixels.tiff")
                #self.vprint(f"Saving active_pixels to : {tiff_path}", 2)
                logging.info(f"Saving active pixels to: {tiff_path}")
                tf.imwrite(tiff_path, active_pixels, dtype = active_pixels.dtype)

        #self.vprint("identified active pixels", 3)
        logging.info("identified active pixels")
        # mask inactive pixels (accelerates subsequent computation)
        if mask_xy is not None:
            np.multiply(active_pixels, mask_xy, out=active_pixels)
            #self.vprint("masked inactive pixels", 3)
            logging.info("Masked inactive pixels")

        # subsequent analysis hard to parallelize; save in memory
        if remove_small_object_framewise:
            active_pixels = active_pixels.compute() if type(active_pixels) == \
                da.core.Array else active_pixels

            # remove small objects
            for cz in range(active_pixels.shape[0]):
                active_pixels[cz, :, :] = morphology.remove_small_objects(
                    active_pixels[cz, :, :], min_size = min_roi_size, connectivity = 4)
            #self.vprint("removed small objects framewise", 3)
            logging.info("Removed small objects framewise")

            # label connected pixels
            # event_map, num_events = measure.label(active_pixels, return_num=True)

            # TODO this fails for very large images
            event_map = np.zeros(data.shape, dtype=np.uintc)
            event_map[:], num_events = ndimage.label(active_pixels)
            #self.vprint("labelled connected pixel framewise. #events: {}".format(num_events), 3)
            logging.info("labelled connected pixel framewise. #events: {}".format(num_events))

        else:

            # remove small objects
            struct = ndimage.generate_binary_structure(3, binary_struct_connectivity)
            struct = ndimage.iterate_structure(struct, binary_struct_iterations).astype(int)

            active_pixels = ndmorph.binary_opening(active_pixels, structure=struct)
            active_pixels = ndmorph.binary_closing(active_pixels, structure=struct)
            #self.vprint("removed small objects", 3)
            logging.info("removed small objects")

            # label connected pixels
            event_map = da.from_array(np.zeros(data.shape, dtype = np.uintc))
            event_map[:], num_events = ndimage.label(active_pixels)
            #self.vprint("labelled connected pixel. #events: {}".format(num_events), 3)
            logging.info("labelled connected pixel. #events: {}".format(num_events))

        # characterize each event

        # event_properties = measure.regionprops(event_map, intensity_image=data, cache=True,
        #                                        extra_properties=[self.trace, self.footprint]
        #                                        )
        # self.vprint("events collected", 3)

        if num_events < 2 * 32767:
            event_map = event_map.astype("uint16")
        else:
            event_map = event_map.astype("uint32")

        #self.vprint("event_map dtype: {}".format(event_map.dtype), 4)
        logging.info("event_map dype: {}".format(event_map.dtype))

        return event_map  # , event_properties

    def get_time_map(self, event_map, chunk:int = 200):

        time_map = np.zeros((event_map.shape[0], np.max(event_map) + 1), dtype = np.bool_)

        Z = event_map.shape[0]
        if type(event_map) == da.core.Array:
            for c in tqdm(range(0, Z, chunk)):

                cmax = min(Z, c + chunk)
                event_map_memory = event_map[c:cmax, :, :].compute()

                for z in range(c, cmax):
                    time_map[z, np.unique(event_map_memory[z - c, :, :])] = 1

        else:

            print("Assuming event_map is in RAM ... ")
            for z in tqdm(range(Z)):
                time_map[z, np.unique(event_map[z, :, :])] = 1

        return time_map

    def custom_slim_features(self, time_map, data_path, event_path, split_events: bool = True):

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

        #self.vprint("creating shared memory arrays ...", 3)
        logging.info("Creating shared memory arrays...")
        data = self.data
        # Calculate n_bytes needed for data.
        n_bytes = data.shape[0] * data.shape[1] * data.shape[2] * data.dtype.itemsize
        # Create shared buffer
        data_sh = shared_memory.SharedMemory(create = True, size = n_bytes)
        # Buffer to array
        data_ = np.ndarray(data.shape, data.dtype, buffer = data_sh.buf)
        data_[:] = data[:]
        # save data info for use in task
        data_info = (data.shape, data.dtype, data_sh.name)

        event = tiledb.open(event_path.as_posix())
        n_bytes = event.shape[0] * event.shape[1] * event.shape[2] * event.dtype.itemsize
        event_sh = shared_memory.SharedMemory(create = True, size = n_bytes)
        event_ = np.ndarray(event.shape, event.dtype, buffer = event_sh.buf)
        event_[:] = event[:]
        event_info = (event.shape, event.dtype, event_sh.name)
        del event
        #self.vprint("data_.dtype: {}".format(data_.dtype), 5)
        #self.vprint("event_.dtype: {}".format(event_.dtype), 5)
        logging.info("data_.dtype: {}".format(data_.dtype))
        logging.info("event_.dtype: {}".format(event_.dtype))

        # collecting tasks
        #self.vprint("collecting tasks ...", 3)
        logging.info("Collecting tasks...")

        e_start = np.argmax(time_map, axis = 0)
        e_stop = time_map.shape[0] - np.argmax(time_map[::-1, :], axis = 0)

        out_path = event_path.parent.joinpath("events/")
        if not out_path.is_dir():
            os.mkdir(out_path)

        # push tasks to client
        e_ids = list(range(1, len(e_start)))
        #self.vprint("#tasks: {}".format(len(e_ids)), 4)
        logging.info("#tasks: {}".format(len(e_ids)))
        random.shuffle(e_ids)
        futures = []
        with Client(memory_limit = 'auto', processes = False,
                    silence_logs = logging.ERROR) as client:
            for e_id in e_ids:
                futures.append(
                    client.submit(
                        characterize_event,
                        e_id, e_start[e_id], e_stop[e_id],
                        data_info, event_info, out_path, split_events
                    )
                )
            progress(futures)

            client.gather(futures)

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
            events.update(np.load(out_path.joinpath(e), allow_pickle = True)[()])
        np.save(combined_path, events)
        shutil.rmtree(out_path)

        return events

def characterize_event(event_id, t0, t1, data_info,
                       event_info, out_path, split_events = True):

    # check if result already exists
    res_path = out_path.joinpath(f"events{event_id}.npy")
    if os.path.isfile(res_path):
        return 2

    # buffer
    t1 = t1+1

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

    if split_events:
        event_map, _ = detect_subevents(data, event_map == event_id)  # TODO inverse map?

    res = {}
    for em_id in np.unique(event_map):

        if em_id < 1:
            continue

        em_id = int(em_id)
        try:
            event_id_key = f"{event_id}_{em_id}" if split_events else event_id
            res[event_id_key] = {}

            z, x, y = np.where(event_map == em_id)
            z0, z1 = np.min(z), np.max(z) + 1
            x0, x1 = np.min(x), np.max(x) + 1
            y0, y1 = np.min(y), np.max(y) + 1

            # local bounding box + global bounding box
            res[event_id_key]["z0"] = t0+z0
            res[event_id_key]["z1"] = t0+z1

            res[event_id_key]["x0"] = gx0+x0
            res[event_id_key]["x1"] = gx0+x1

            res[event_id_key]["y0"] = gy0+y0
            res[event_id_key]["y1"] = gy0+y1

            # bbox dimensions
            dz, dx, dy = z1 - z0, x1 - x0, y1 - y0
            res[event_id_key]["dz"] = dz
            res[event_id_key]["dx"] = dx
            res[event_id_key]["dy"] = dy

            # area
            res[event_id_key]["area"] = len(z)
            res[event_id_key]["bbox_pix_num"] = int(dz * dx * dy)

            # shape
            mask = np.ones((dz, dx, dy), dtype = np.bool_)
            mask[(z-z0, x-x0, y-y0)] = 0
            res[event_id_key]["mask"] = np.invert(mask).flatten()

            if dz > 1:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    try:
                        props = regionprops_table(np.invert(mask).astype(np.uint8), properties =
                            ['centroid_local',
                            'axis_major_length',
                            "axis_minor_length",
                            'extent',
                            'solidity',
                            'area',
                            # 'area_convex',  # too much memory
                            'equivalent_diameter_area',
                            # 'feret_diameter_max'  # too much memory
                            ])

                        props["centroid_local-0"] = props["centroid_local-0"] / dz
                        props["centroid_local-1"] = props["centroid_local-1"] / dx
                        props["centroid_local-2"] = props["centroid_local-2"] / dy

                        for k in props.keys():
                            res[event_id_key][f"mask_{k}"] = props[k][0]

                    except ValueError as err:
                        print("\t Error in ", event_id_key)
                        print("\t", err)

            # calculate footprint features
            fp = np.invert(np.min(mask, axis = 0))
            res[event_id_key]["footprint"] = fp.flatten()

            props = regionprops_table(fp.astype(np.uint8), properties =
                ['centroid_local','axis_major_length', "axis_minor_length",
                'eccentricity', 'equivalent_diameter_area', 'extent', 'feret_diameter_max',
                'orientation', 'perimeter', 'solidity', 'area', 'area_convex']
                                    )

            props["cx"] = gx0 + props["centroid_local-0"]
            props["cy"] = gy0 + props["centroid_local-1"]

            props["centroid_local-0"] = props["centroid_local-0"] / dx
            props["centroid_local-1"] = props["centroid_local-1"] / dy

            for k in props.keys():
                res[event_id_key][f"fp_{k}"] = props[k][0]

            #trace
            signal = data[z0:z1, x0:x1, y0:y1]
            masked_signal = np.ma.masked_array(signal, mask)
            res[event_id_key]["trace"] = np.ma.filled(np.nanmean(masked_signal, axis = (1, 2)), fill_value = 0)

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

def detect_subevents(img, mask, sigma:int = 2,
                    min_local_max_distance:int = 5, local_max_threshold:float = 0.5, min_roi_frame_area:int = 5,
                    reject_if_original:bool = True, verbosity:int = 0):
    assert img.shape == mask.shape, "image ({}) and mask ({}) don't have the same dimension!".format(img.shape,
                                                                                                    mask.shape)

    mask = ~mask
    Z, X, Y = mask.shape

    new_mask = np.zeros((Z, X, Y), dtype = "i2")
    last_mask_frame = np.zeros((X, Y), dtype = "i2")
    next_event_id = 1
    local_max_container = []

    for cz in range(Z):
        frame_mask = mask[cz, :, :]
        frame_raw = img[cz, :, :] if sigma is None else gaussian(img[cz, :, :], sigma = sigma)
        frame_raw = np.ma.masked_array(frame_raw, frame_mask)

        if verbosity > 3: print(f"\n----- {cz} -----")

        ####################
        ## Find Local Maxima

        local_maxima = peak_local_max(frame_raw, min_distance = min_local_max_distance, threshold_rel=local_max_threshold)
        local_maxima = np.array(
            [(lmx, lmy, last_mask_frame[lmx, lmy]) for (lmx, lmy) in zip(local_maxima[:, 0], local_maxima[:, 1])],
            dtype = "i2")

        # Try to find global maximum if no local maxima were found
        if len(local_maxima) == 0:

            mask_area = np.sum(frame_mask)  # Look for global max
            glob_max = np.unravel_index(np.argmax(frame_raw), (X, Y))  # Look for global max

            if (mask_area > 0) and (glob_max != (0, 0)):
                local_maxima = np.array([[glob_max[0], glob_max[1], 0]])
            else:
                local_max_container.append(local_maxima)
                last_mask_frame = np.zeros((X, Y), dtype = "i2")
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
        if verbosity > 3: print("last_local_max_labels:\n", last_local_max_labels)
        if verbosity > 3: print("curr_local_max_labels:\n", curr_local_max_labels)

        for last_local_max_label in last_local_max_labels:

            if (last_local_max_label != 0) and (last_local_max_label not in curr_local_max_labels):

                prev_local_maxima = local_max_container[-1]
                if verbosity > 5: print("prev_local_maxima: ", prev_local_maxima)
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
        local_maxima_labels, local_maxima_counts = np.unique(local_maxima[:, 2], return_counts = True)
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

        #############################
        ## Separate overlaying events

        if local_maxima.shape[0] == 1:

            # Single Local Maximum (== global maximum)
            last_mask_frame = np.zeros((X, Y), dtype = "i2")
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

            last_mask_frame = watershed(basin, seeds).astype("i2")
            last_mask_frame[frame_mask] = 0

        # save results of current run
        new_mask[cz, :, :] = last_mask_frame

    unique_elements = np.unique(new_mask)
    if reject_if_original & (np.array_equal(unique_elements, [0, 1]) | np.array_equal(unique_elements,[0])):
        # did not split original event
        return ~mask, None
    else:
        return new_mask, local_max_container

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required = True, type = str, default = None, help = "Input file")
    parser.add_argument("-k", "--key", type = str, default = None, help = "dataset name")
    parser.add_argument("-t", "--threshold", type = int, default = None, help = "use -1 for automatic thresholding")
    parser.add_argument("-v", "--verbosity", type = int, default = 1)
    parser.add_argument("--binarystruct", type = int, default = 1)
    parser.add_argument("--binaryconnect", type = int, default = 2)
    parser.add_argument("--splitevents", type = bool, const = True, default = True, nargs = '?', 
                        help = "splits detected events into smaller events if multiple peaks are detected")
    parser.add_argument("--usedask", type = bool, default = True)
    parser.add_argument("--output", type = str, default = None, help = "output folder name. If output = None, output is set to input_path + .roi") # Added option
    parser.add_argument("--saveactpixels", type = bool, default = False, help = "Save active pixels file")


    args = parser.parse_args()

    args.threshold = args.threshold if args.threshold != -1 else None

    # logging
    # TODO fill in

    # deal with data input
    ed = Detector(args.input, verbosity = args.verbosity, output = args.output)
    ed.run(dataset = args.key, use_dask = args.usedask, subset = None,
           split_events = args.splitevents,
           binary_struct_connectivity = args.binaryconnect,
           binary_struct_iterations = args.binarystruct,
           save_activepixels = args.saveactpixels)
