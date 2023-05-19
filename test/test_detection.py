from astroCAST.detection import Detector
import astroCAST
from pathlib import Path
import os

def test_run_function(threshold = None, save_activepixels = False, output = None):
    det = Detector("/Volumes/tank/groups/herlenius/anagon/22A7x7-1.h5", output = output)
    det.run(dataset = "dff/ast", use_dask = True, save_activepixels = save_activepixels)

    # Assert function's expected behaviour
    assert os.path.exists(f"{det.output_directory}/event_map.tdb"), "event map file does not exist in output_directory"
    assert os.path.exists(f"{det.output_directory}/event_map.tiff"), "tiff output file does not exist in output_directory"
    if save_activepixels == True: 
        assert os.path.exists(f"{det.output_directory}/active_pixels.tiff"), "Active pixels file does not exist in output directory"
    assert os.path.exists(f"{det.output_directory}/time_map.npy"), "time_map.npy file does not exist in output directory"
    assert os.path.exists(f"{det.output_directory}/events.npy"), "Combined events file does not exist in the output directory"
    assert os.path.exists(f"{det.output_directory}/meta.json"), "meta.json file does not exist in the output directory." 
    assert det.output_directory.exists(), "Output folder does not exist"
    assert bool(det.meta), "metadata dictionary is empty"
    assert det.data.size != 0, "data object is empty"
    assert det.data.shape is not None, "data has no dimensions"
test_run_function()

test_run_function(save_activepixels = True, output = "/Users/anagon/Desktop/output.ast.roi")
