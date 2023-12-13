# Import necessary modules of astroCAST
import os
import pathlib
from pathlib import Path

from astrocast.denoising import Network, SubFrameGenerator

root = Path("/tf/data/")

model_path = root.joinpath("model")
save_path = root.joinpath("model_retrain")
train_paths = root.joinpath("pub_data/InVivoSuppRaw.h5")
infer_input = train_paths

infer_output = infer_input.parent.with_suffix(".retrain.tiff")

# find model
if type(model_path) in [str, pathlib.PosixPath]:
    model_path = Path(model_path)

if os.path.isdir(model_path):
    
    models = list(model_path.glob("*.h*5"))
    
    if len(models) < 1:
        raise FileNotFoundError(f"cannot find model in provided directory: {model_path}")
    
    models.sort(key=lambda x: os.path.getmtime(x))
    model_path = models[0]
    print(f"directory provided. Selected most recent model: {model_path}")

elif os.path.isfile(model_path):
    model_path = model_path

else:
    raise FileNotFoundError(f"cannot find model: {model_path}")

# Generators
loc = "mc/ch0"  # data/
param = dict(loc=loc, input_size=(256, 256), pre_post_frame=5, gap_frames=0, normalize="global", in_memory=True)

train_gen = SubFrameGenerator(paths=train_paths, padding=None, batch_size=16, max_per_file=100,
                              allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)

# Network
net = Network(train_generator=train_gen, val_generator=None, learning_rate=0.001, decay_rate=None,
              pretrained_weights=model_path.as_posix(),
              n_stacks=3, kernel=64,
              batchNormalize=False, use_cpu=False)

# Train
net.retrain_model(batch_size=1,
                  frozen_epochs=50, unfrozen_epochs=50,
                  patience=20, min_delta=0.0001, monitor="loss",
                  save_model=save_path)

# if infer_output.is_file():
#     infer_output.unlink()
#
# inf_param = param.copy()
# inf_param["paths"] = infer_input
# inf_gen = SubFrameGenerator(padding="edge", batch_size=32, allowed_rotation=[0], allowed_flip=[-1], z_select=(0, 25),
#                             overlap=10,
#                                 shuffle=False, max_per_file=None, **inf_param)
#
# inf_gen.infer(model=model_path, output=infer_output.as_posix(), out_loc="", rescale=True, dtype=float)
