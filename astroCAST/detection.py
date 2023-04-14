import argparse
import logging
import os
import numpy as np
from pathlib import Path

class Detector:

    def __init__(self, input_path, output=None, indices=None, verbosity=1):

        logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=verbosity)

        # paths
        self.input_path = Path(input_path)
        self.output = output if output is None else Path(output)
        working_directory = self.input_path.parent

        logging.info(f"working directory: {working_directory}")
        logging.info(f"input file: {self.input_path}")

        # quality check arguments
        assert os.path.isfile(input_path) or os.path.isdir(input_path), f"input file does not exist: {input_path}"
        assert (output is None) or (~ output.isdir()), f"output file already exists: {output}"
        assert indices is None or indices.shape == (3, 2), "indices must be np.arry of shape (3, 2) -> ((z0, z1), " \
                                                           "(x0, x1), (y0, y1)). Found: " + indices

        # shared variables
        self.file = None
        self.Z, self.X, self.Y = None, None, None

    def run(self, dataset=None,
            threshold=None, min_size=20, use_dask=False, adjust_for_noise=False,
            subset=None, split_events=True,
            binary_struct_iterations=1, binary_struct_connectivity=2, # TODO better way to do this
            ):

        # output folder
        output_directory = self.output if self.output is not None else self.input_path.with_suffix(
                ".roi") if dataset is None else self.input_path.with_suffix(".{}.roi".format(dataset.split("/")[-1]))

        if not output_directory.is_dir():
            output_directory.mkdir()

        logging.info(f"output directory: {output_directory}")

        # profiling
        pbar = ProgressBar(minimum=10)
        pbar.register()

        # TODO save this information somewhere
        # resources = ResourceProfiler()
        # resources.register()

        # load data
        data = self._load(dataset_name=dataset, use_dask=use_dask, subset=subset)
        self.data = data
        self.Z, self.X, self.Y = data.shape
        self.vprint(data if use_dask else data.shape, 2)

        # calculate event map
        event_map_path = self.output_directory.joinpath("event_map.tdb")
        if not os.path.isdir(event_map_path):
            self.vprint("Estimating noise", 2)
            # TODO maybe should be adjusted since it might already be calculated
            noise = self.estimate_background(data) if adjust_for_noise else 1

            self.vprint("Thresholding events", 2)
            event_map = self.get_events(data, roi_threshold=threshold, var_estimate=noise, min_roi_size=min_size,
                                        binary_struct_iterations=binary_struct_iterations, binary_struct_connectivity=binary_struct_connectivity)

            self.vprint(f"Saving event map to: {event_map_path}", 2)
            event_map.rechunk((100, 100, 100)).to_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            self.vprint(f"Saving tiff to : {tiff_path}", 2)
            tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        else:
            self.vprint(f"Loading event map from: {event_map_path}", 2)
            event_map = da.from_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            if not tiff_path.is_file():
                self.vprint(f"Saving tiff to : {tiff_path}", 2)
                tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        # calculate time map
        self.vprint("Calculating time map", 2)
        time_map_path = self.output_directory.joinpath("time_map.npy")
        if not time_map_path.is_file():
            time_map = self.get_time_map(event_map)

            self.vprint(f"Saving event map to: {time_map_path}", 2)
            np.save(time_map_path, time_map)
        else:
            self.vprint(f"Loading time map from: {time_map_path}", 2)
            time_map = np.load(time_map_path.as_posix())

        # calculate features
        self.vprint("Calculating features", 2)
        self.custom_slim_features(time_map, self.input_path, event_map_path, split_events=split_events)

        self.vprint("saving features", 2)
        with open(self.output_directory.joinpath("meta.json"), 'w') as outfile:
            json.dump(self.meta, outfile)

        self.vprint("Run complete! [{}]".format(self.input_path, 1), 1)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-k", "--key", type="str", default=None)
    parser.add_argument("-t", "--threshold", type=int, default=None, help="use -1 for automatic thresholding")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--binarystruct", type=int, default=1)
    parser.add_argument("--binaryconnect", type=int, default=2)
    parser.add_argument("--splitevents", type=bool, const=True, default=True,
                        help="splits detected events into smaller events if multiple peaks are detected")
    parser.add_argument("--usedask", type=bool, default=True)

    args = parser.parse_args()

    args.threshold = args.threshold if args.threshold != -1 else None

    # logging
    # TODO fill in

    # deal with data input
    ed = Detector(args.input, verbosity=args.verbosity)
    ed.run(dataset=args.key, use_dask=args.usedask, subset=None,
           split_events=args.splitevents,
           binary_struct_connectivity=args.binaryconnect,
           binary_struct_iterations=args.binarystruct)
