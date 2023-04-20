from astroCAST.denoising import FullFrameGenerator, SubFrameGenerator
from astroCAST.denoising import DeepInterpolate, Network
import os

def test_generator_full_frame(file_path, pre_post_frame):

    gen = FullFrameGenerator(file_path=file_path, loc="data/ch0", pre_post_frame=pre_post_frame,
                             batch_size=2, steps_per_epoch=1, )
    item = gen[0]

def test_generator_sub_frame(file_path, pre_post_frame):

    gen = SubFrameGenerator(paths=file_path, loc="data/ch0", pre_post_frame=pre_post_frame,
                             input_size=(25, 25), batch_size=2,)
    item = gen[0]

def test_network():

    gen = FullFrameGenerator(pre_post_frame=5, batch_size=5, steps_per_epoch=1, file_path="testdata/sample_0.tiff")
    net = Network(train_generator=gen, n_stacks=1, kernel=16)

def test_inference(file_path, model="testdata/test_model.h5", out_path=None):

    di = DeepInterpolate()

    # out_path = "testdata/sample_0_inf.tiff"
    di.infer(file_path, output=out_path, model=model)

    if out_path is not None:

        assert os.path.isfile(out_path), "cannot find output tiff: {}".format(out_path)
        os.remove(out_path)


