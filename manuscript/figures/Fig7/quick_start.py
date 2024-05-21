from pathlib import Path
import tracemalloc
import humanize

from astrocast.preparation import MotionCorrection, Input, Delta
from astrocast.denoising import SubFrameDataset, PyTorchNetwork
from astrocast.detection import Detector
import time
import shutil

for test_path in Path.cwd().glob("*.h5"):
    for use_dn in [True, False]:
        for use_sub in [True, False]:
            
            txt_path = Path(f"{test_path.stem}_{use_dn}_{use_sub}.txt")
            if txt_path.exists():
                continue
            
            temp_path = Path("current_test_file.h5")
            if temp_path.exists():
                temp_path.unlink()
            
            shutil.copy(src=test_path, dst=temp_path)
            
            t0 = time.time()
            tracemalloc.start()
            
            error_occured = False
            try:
                # # 1.2 Motion correction
                
                loc_in = "data/ch0"
                loc_out = "mc/ch0"
                
                MC = MotionCorrection()
                MC.run(path=temp_path, loc=loc_in)
                MC.save(output=temp_path, loc=loc_out, chunk_strategy="Z", compression="gzip")
                
                # Model parameters
                if use_dn:
                    
                    # ## 1.3 Denoising
                    
                    # Inference dataset
                    input_size = (128, 128)
                    loc_in = loc_out
                    loc_out = "inf/ch0"
                    use_cpu = True  # switch to False, if cuda is available
                    
                    infer_dataset = SubFrameDataset(paths=temp_path, loc=loc_in, input_size=input_size,
                                                    allowed_rotation=0,
                                                    allowed_flip=-1,
                                                    shuffle=False, normalize="global", overlap=10, padding="edge")
                    
                    model_path = "../../denoiser_models/1p_exvivo_input_size_128_128_pre_post_frame_5-gap_frames_0-train_rotation_1_2_3-architecture_3_64_epochs-50.pth"
                    pre_post_frames = 5
                    gap_frames = 0
                    n_stacks, kernels = (3, 64)
                    
                    net = PyTorchNetwork(infer_dataset, load_model=model_path, val_dataset=None, batch_size=16,
                                         n_stacks=n_stacks,
                                         kernels=kernels, kernel_size=3, batch_normalize=False,
                                         use_cpu=use_cpu)
                    
                    # Denoise data
                    net.infer(dataset=infer_dataset, output=temp_path, out_loc=loc_out, batch_size=1, dtype=float)
                
                # ## 1.3 Subtraction
                if use_sub:
                    
                    loc_in = loc_out
                    loc_out = "df/ch0"
                    
                    delta = Delta(temp_path, loc=loc_in)
                    res_delta = delta.run(method="dF", scale_factor=0.25, neighbors=100, wlen=50, distance=5,
                                          max_chunk_size_mb=10, width=5)
                    delta.save(output_path=temp_path, loc=loc_out, chunk_strategy="balanced", compression="gzip")
                
                # # 2. Event detection
                event_path = temp_path.with_suffix(".ch0.roi")
                if event_path.exists():
                    import shutil
                    
                    shutil.rmtree(event_path)
                
                detector = Detector(input_path=test_path)
                event_dictionary = detector.run(loc=loc_out, debug=True, exclude_border=20, temporal_prominence=0.3)
            
            except Exception as e:
                
                error_occured = True
                err_msg = f"Error ({humanize.naturaldelta(t1 - t0)}): {str(e)}"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            t1 = time.time()
            
            with open(txt_path, "w") as f:
                f.write(f"{test_path}\n"
                        f"{use_dn}\n"
                        f"{use_sub}\n{t1 - t0}")
