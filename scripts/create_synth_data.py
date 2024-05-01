from pathlib import Path
from typing import List, Tuple, Union

import h5py as h5
import tifffile as tif
from tqdm.auto import tqdm

import astrocast.helper as helper
import astrocast.simulation as simulator


def create_synth_data(output_dir: Union[str, Path],
                      signal_generator: Union[helper.SignalGenerator, List[helper.SignalGenerator]] = None,
                      video_size: Tuple[int, int, int] = (200, 200, 200),
                      num_cells: int = 40, max_astrocytes: int = None,
                      blur: Tuple[int, int] = None, event_probability: float = 0.01,
                      noise: Tuple[int, int] = None, signal_noise: Tuple[int, int] = None,
                      n_repetitions: int = 1,
                      ):
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    frames, X, Y = video_size
    for exp in tqdm(range(n_repetitions), desc="Repetition"):
        
        sim = simulator.SimData(frames=frames, X=X, Y=Y)
        sim.generate_voronoi(num_cells=num_cells)
        _ = sim.populate_astrocytes(show_progress=False, plot=False,
                                    max_astrocytes=max_astrocytes)
        
        activity = sim.generate_activity(signal_generator=signal_generator,
                                         blur=blur, noise=noise, signal_noise=signal_noise,
                                         event_probability=event_probability)
        
        tiff_path = output_dir.joinpath(f"{exp}.tiff")
        tif.imwrite(tiff_path.as_posix(), activity)
        
        h5_path = output_dir.joinpath(f"{exp}.h5")
        with h5.File(h5_path, "w") as h:
            h.create_dataset("data/ch0", data=activity)
        
        p_path = output_dir.joinpath(f"{exp}.p")
        _ = sim.get_events(save_path=p_path)


if __name__ == "__main__":
    
    print(f"Parent directory:")
    p_dir = Path(input())
    
    def_ = dict(parameter_fluctuations=0.01, trace_length=100)
    
    generators = [
        helper.SignalGenerator(**def_),
        helper.SignalGenerator(num_peaks=2, peak_rebounce_ratio=0.8, **def_),
        helper.SignalGenerator(num_peaks=3, peak_rebounce_ratio=0.9, **def_),
        helper.SignalGenerator(leaky_k=0.12, k=3, num_peaks=1, **def_)
        ]
    
    experiments = [
        dict(blur=(5, 5), noise=(0.1, 0.01), event_probability=0.001, output_dir=p_dir.joinpath("easy"),
             video_size=(2000, 400, 400), num_cells=100, signal_generator=generators),
        dict(blur=(5, 5), noise=(2, 0.5), event_probability=0.001, output_dir=p_dir.joinpath("noise"),
             video_size=(2000, 400, 400), num_cells=100, signal_generator=generators),
        dict(blur=(5, 5), noise=(0.1, 0.01), event_probability=0.001, output_dir=p_dir.joinpath("noise"),
             video_size=(10000, 200, 200), num_cells=100, signal_generator=generators),
        ]
    
    for exp in tqdm(experiments, total=len(experiments), desc="Experiments"):
        create_synth_data(**exp)
