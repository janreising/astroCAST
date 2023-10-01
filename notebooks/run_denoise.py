# Import necessary modules of astroCAST
from astrocast.denoising import SubFrameGenerator, Network
from astrocast.analysis import Video
from pathlib import Path

root = Path("/tf/data/")

model_path = root.joinpath("model")
train_paths = list(root.joinpath("ast").glob("train*.h5"))
val_paths = list(root.joinpath("ast").glob("test*.h5"))
print(len(train_paths), len(val_paths))

infer_input = root.joinpath("ast/true_test_70v89olb.h5")
infer_output = infer_input.parent.with_suffix(".tiff")

param = dict(loc="data", input_size=(256, 256), pre_post_frame=5, gap_frames=0, normalize="global", in_memory=True)

train_gen = SubFrameGenerator(paths=train_paths, padding=None, batch_size=12 ,max_per_file=16,
                               allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)

val_gen = SubFrameGenerator(paths=val_paths, padding=None, batch_size=16, max_per_file=1, cache_results=True,
                                   allowed_rotation=[0], allowed_flip=[-1], shuffle=False, **param)

net = Network(train_generator=train_gen, val_generator=val_gen,
              pretrained_weights = model_path.as_posix(),
              n_stacks=3, kernel=64,
              batchNormalize=False, use_cpu=False)
net.run(batch_size=1,
        num_epochs=50,
        patience=3, min_delta=0.0001,
        save_model=model_path)

if infer_output.is_file():
    infer_output.unlink()

inf_param = param.copy()
inf_param["paths"] = infer_input
inf_gen = SubFrameGenerator(padding="edge", batch_size=32, allowed_rotation=[0], allowed_flip=[-1], z_select=(0, 25),
                            overlap=10,
                                shuffle=False, max_per_file=None, **inf_param)

inf_gen.infer(model=model_path, output=infer_output.as_posix(), out_loc="", rescale=True, dtype=float)

