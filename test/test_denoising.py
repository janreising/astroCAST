import logging
import tempfile
from pathlib import Path

import pytest

from astrocast.denoising import FullFrameGenerator, SubFrameGenerator
from astrocast.denoising import Network
import os

from astrocast.helper import SampleInput
from astrocast.preparation import IO

class Test_Generators:

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("pre_post_frame", [5, (3, 2)])
    def test_generator_full_frame(self, extension, pre_post_frame):

        si = SampleInput()
        file_path = si.get_test_data(extension=extension)
        loc = si.get_h5_loc()

        gen = FullFrameGenerator(file_path=file_path, loc=loc, pre_post_frame=pre_post_frame,
                                 batch_size=25)

        for ep in range(2):
            for item in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("normalize", [None, "local", "global"])
    @pytest.mark.parametrize("pre_post_frame", [5, (3, 2)])
    def test_generator_sub_frame(self, extension, pre_post_frame, normalize):

        si = SampleInput()
        file_path = si.get_test_data(extension=extension)
        loc = si.get_h5_loc()

        gen = SubFrameGenerator(paths=file_path, loc=loc, pre_post_frame=pre_post_frame,
                                 input_size=(25, 25), batch_size=25, normalize=normalize)

        for ep in range(2):
            for item in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.xdist_group(name="tensorflow")
    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_network(self, extension):

        si = SampleInput()
        file_path = si.get_test_data(extension=extension)
        loc = si.get_h5_loc()

        param = dict(paths=file_path, loc=loc, input_size=(25, 25), pre_post_frame=5, gap_frames=0,
                     normalize="global", cache_results=True, in_memory=True)

        train_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=50,
                                       allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)

        net = Network(train_generator=train_gen, val_generator=train_gen, n_stacks=1, kernel=4, batchNormalize=False, use_cpu=True)
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)

    @pytest.mark.xdist_group(name="tensorflow")
    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_inference_full(self, extension, out_path=None):

        si = SampleInput()
        file_path = si.get_test_data(extension=extension)
        loc = si.get_h5_loc()

        gen = FullFrameGenerator(file_path=file_path, loc=loc, pre_post_frame=5,
                                 batch_size=25, total_samples=50)

        net = Network(train_generator=gen, val_generator=gen, n_stacks=1, kernel=8, batchNormalize=False, use_cpu=True)
        net.run(batch_size=gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)
        model = net.model

        # out_path = "testdata/sample_0_inf.tiff"
        gen.infer(model=model, output=out_path, )

        if out_path is not None:
            assert os.path.isfile(out_path), "cannot find output tiff: {}".format(out_path)
            os.remove(out_path)

    @pytest.mark.xdist_group(name="tensorflow")
    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("output_file", [None, "inf.tiff", "inf.h5"])
    def test_inference_sub(self, extension, output_file):

        si = SampleInput()
        file_path = si.get_test_data(extension=extension)
        loc = si.get_h5_loc()

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)
            assert tmpdir.is_dir()

            if output_file is None:
                out_path = None
            else:
                out_path = tmpdir.joinpath(output_file)

            param = dict(paths=file_path, loc=loc, input_size=(25, 25), pre_post_frame=5, gap_frames=0,
                         normalize="global", cache_results=True, in_memory=True)

            train_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=50,
                                           allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)
            val_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=5,
                                           allowed_rotation=[0], allowed_flip=[-1], shuffle=True, **param)

            net = Network(train_generator=train_gen, val_generator=val_gen, n_stacks=1, kernel=8, batchNormalize=False, use_cpu=True)
            net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)
            model = net.model

            inf_gen = SubFrameGenerator(padding="edge", batch_size=25,
                                        allowed_rotation=[0], allowed_flip=[-1],
                                        shuffle=False,
                                        logging_level=logging.DEBUG,
                                        **param)

            inf_loc = "inf/ch0"
            res = inf_gen.infer(model=model, output=out_path, out_loc=inf_loc, rescale=False)

            # Check result
            io = IO()
            data = io.load(file_path, h5_loc=loc, lazy=True)

            if out_path is not None:
                assert out_path.is_file(), "cannot find output file: {}".format(out_path)
                res = io.load(out_path, h5_loc=inf_loc, lazy=True)

            assert res.shape == data.shape, f"inferred output has incorrect shape: " \
                                                f"orig>{data.shape} vs. inf>{res.shape}"



