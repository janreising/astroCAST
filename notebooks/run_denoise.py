# Import necessary modules of astroCAST
from astrocast.denoising import SubFrameGenerator, Network
from astrocast.analysis import Video
from pathlib import Path

model_path = Path("./data/model")
train_paths = list(Path("./data/ast").glob("*.h5"))
print(len(train_paths))

# infer_input = Path("/media/janrei1/data/deep/to_infer.h5")
# infer_output = infer_input.with_suffix(".tiff")
# out_loc = "inf/ch0"