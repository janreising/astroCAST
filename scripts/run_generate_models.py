import itertools
import sys
import traceback
from pathlib import Path

import yaml

from astrocast.denoising import Network, PyTorchNetwork, SubFrameDataset, SubFrameGenerator

with open("generate_pretrain_models.yaml", "r") as file:
    config = yaml.safe_load(file)

params = config["param"]
use_pytorch = params["use_pytorch"]
use_cpu = params["use_cpu"]

root = Path(config["root_path"])
model_path = Path(config["model_path"])

# Convert single values to lists and keep lists/tuples as-is
params = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in params.items()}

# Generate all combinations
keys, values = zip(*params.items())
combinations = [dict(zip(keys, prod)) for prod in itertools.product(*values)]

# Loop over all combinations
for param_set in combinations:
    
    name_parts = []
    for key, value in param_set.items():
        
        if key in ["epochs", "patience", "min_delta"]:
            continue
        
        if isinstance(value, (list, tuple)):
            value_str = '_'.join(map(str, value))
        else:
            value_str = str(value)
        
        name_parts.append(f"{key}_{value_str}")
    name = '-'.join(name_parts)
    
    for k, v in config["data"].items():
        
        print(f"{k}:{name}")
        
        save_model_path = model_path.joinpath(f"{k}_{name}")
        if save_model_path.joinpath("model.h5").is_file():
            print(f"Skipping > model exists: {save_model_path}")
            continue
        
        try:
            input_size = param_set["input_size"]
            pre_post_frames = param_set["pre_post_frames"]
            gap_frames = param_set["gap_frames"]
            train_rotation = param_set["train_rotation"]
            
            n_stacks, kernel = param_set["architecture"]
            epochs = param_set["epochs"]
            patience = param_set["patience"]
            min_delta = param_set["min_delta"]
            
            # Trainer
            train_str = v["train"]
            if "*" in train_str:
                train_paths = list(root.glob(train_str))
            else:
                train_paths = root.joinpath(train_str)
            
            # Validator
            val_paths = None
            if "val" in v:
                val_str = v["train"]
                if "*" in val_str:
                    val_paths = list(root.glob(val_str))
                else:
                    val_paths = root.joinpath(val_str)
            
            if not use_pytorch:
                
                train_gen = SubFrameGenerator(paths=train_paths, max_per_file=v["max_per_file"], loc=v["loc"],
                                              input_size=input_size,
                                              pre_post_frames=pre_post_frames, gap_frames=gap_frames,
                                              allowed_rotation=train_rotation,
                                              padding=None, batch_size=8, normalize="global", in_memory=False,
                                              allowed_flip=[0, 1], shuffle=True)
                
                # Validator
                if val_paths is not None:
                    
                    val_gen = SubFrameGenerator(
                            paths=val_paths, max_per_file=3, loc=v["loc"], input_size=input_size,
                            pre_post_frames=pre_post_frames, gap_frames=gap_frames, allowed_rotation=[0],
                            padding=None, batch_size=16, normalize="global", in_memory=False,
                            cache_results=True,
                            allowed_flip=[-1], shuffle=True)
                
                else:
                    val_gen = None
                
                # Network
                
                net = Network(train_generator=train_gen, val_generator=val_gen, learning_rate=0.001, decay_rate=0.99,
                              pretrained_weights=None,
                              n_stacks=n_stacks, kernel=kernel,
                              batchNormalize=False, use_cpu=use_cpu)
                
                net.run(batch_size=1, num_epochs=epochs, patience=patience, min_delta=min_delta,
                        save_model=save_model_path)
            
            else:
                
                train_dataset = SubFrameDataset(paths=train_paths, input_size=input_size, loc=v["loc"],
                                                pre_post_frames=pre_post_frames, max_per_file=v["max_per_file"],
                                                gap_frames=gap_frames, allowed_rotation=train_rotation, padding=None,
                                                normalize="global", in_memory=False, allowed_flip=[0, 1], shuffle=True)
                
                val_dataset = None
                if "val" in v:
                    val_dataset = SubFrameDataset(paths=val_paths, input_size=input_size, loc=v["loc"],
                                                  pre_post_frames=pre_post_frames, max_per_file=3,
                                                  gap_frames=gap_frames, allowed_rotation=0, padding=None,
                                                  normalize="global", in_memory=False, allowed_flip=-1, shuffle=True)
                
                net = PyTorchNetwork(train_dataset, val_dataset=val_dataset, batch_size=16, shuffle=True, num_workers=4,
                                     learning_rate=0.001, momentum=0.9, decay_rate=0.1, decay_steps=30,
                                     n_stacks=n_stacks, kernels=kernel, kernel_size=3, batch_normalize=False,
                                     use_cpu=use_cpu)
                
                net.run(num_epochs=epochs, save_model=save_model_path, patience=patience, min_delta=min_delta)
        
        except KeyboardInterrupt:
            sys.exit(2)
        
        except Exception as err:
            print(f"Error in {k}:{name}: {err}")
            traceback.print_exc()
