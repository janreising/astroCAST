import logging
from pathlib import Path

import pytest

from astrocast.denoising import Network
from astrocast.denoising import SubFrameGenerator
from astrocast.helper import SampleInput
from astrocast.preparation import IO


class TestGenerators:
    data = None
    si_objects = None

    @classmethod
    def setup_class(cls):

        cls.data = {}
        cls.si_objects = []
        for extension in [".h5", ".tiff"]:
            si = SampleInput(tmp_dir=None)
            file_path = si.get_test_data(extension=extension)
            loc = si.get_loc()

            cls.data[extension] = (file_path, loc)
            cls.si_objects.append(si)

    @classmethod
    def teardown_class(cls):

        for si in cls.si_objects:
            si.clean_up()

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("pre_post_frame", [5, (3, 2)])
    @pytest.mark.parametrize("gap_frames", [0, 2, (2, 1)])
    def test_generator_sub_vanilla(
            self, tmpdir, extension, pre_post_frame, gap_frames, normalize=None, random_offset=False, overlap=None,
            z_select=None, drop_frame_probability=None,
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("drop_frame_probability", [0.05, 0.5])
    def test_generator_sub_drop_frame(
            self, drop_frame_probability, extension=".h5", pre_post_frame=5, gap_frames=0, normalize=None,
            random_offset=False, overlap=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("normalize", ["local", "global"])
    def test_generator_sub_normalize(
            self, normalize, drop_frame_probability=None, extension=".h5", pre_post_frame=5, gap_frames=0,
            random_offset=False, overlap=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("overlap", [0, 2, 0.1])
    def test_generator_sub_overlap(
            self, overlap, drop_frame_probability=None, extension=".h5", pre_post_frame=5, gap_frames=0,
            random_offset=False, normalize=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("random_offset", [True])
    def test_generator_sub_random_offset(
            self, random_offset, drop_frame_probability=None, extension=".h5", pre_post_frame=5, gap_frames=0,
            overlap=None, normalize=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("z_select", [(0, 25), (0, 100)])
    def test_generator_sub_z_select(
            self, z_select, drop_frame_probability=None, extension=".h5", pre_post_frame=5, gap_frames=0, overlap=None,
            normalize=None, random_offset=False
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frame=pre_post_frame, gap_frames=gap_frames, random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.tensorflow
    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("n_stacks", [1, 2])
    def test_network(self, extension, n_stacks):

        file_path, loc = self.data[extension]

        param = dict(
            paths=file_path, loc=loc, input_size=(32, 32), pre_post_frame=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True,
            **param
        )

        net = Network(
            train_generator=train_gen, val_generator=train_gen, n_stacks=n_stacks, kernel=4, batchNormalize=False,
            use_cpu=True
        )
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01)

    @pytest.mark.xdist_group(name="tensorflow")
    def test_network_retrain(self, extension=".h5"):

        file_path, loc = self.data[extension]

        param = dict(
            paths=file_path, loc=loc, input_size=(25, 25), pre_post_frame=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True,
            **param
        )

        net = Network(
            train_generator=train_gen, val_generator=train_gen, n_stacks=1, kernel=4, batchNormalize=False, use_cpu=True
        )
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01)

        net.retrain_model(5, 5)

    @pytest.mark.xdist_group(name="tensorflow")
    def test_network_save(self, tmpdir, extension=".h5"):

        file_path, loc = self.data[extension]

        save_model_dir = Path(tmpdir.strpath)
        save_model_path = save_model_dir.joinpath("model.h5")

        param = dict(
            paths=file_path, loc=loc, input_size=(25, 25), pre_post_frame=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
            shuffle=True, **param
        )

        net = Network(
            train_generator=train_gen, val_generator=train_gen, n_stacks=1, kernel=4, batchNormalize=False,
            use_cpu=True
        )
        net.run(
            batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=save_model_dir
        )

        res = train_gen.infer(model=save_model_path, output=None, out_loc="inf/ch0", rescale=False)

    @pytest.mark.tensorflow
    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("output_file", [None, "inf.tiff", "inf.h5"])
    @pytest.mark.parametrize("rescale", [True, False])
    def test_inference_sub(self, tmpdir, extension, output_file, rescale):

        file_path, loc = self.data[extension]

        tmpdir = Path(tmpdir.strpath)
        assert tmpdir.is_dir()

        if output_file is None:
            out_path = None
        else:
            out_path = tmpdir.joinpath(output_file)

        param = dict(
            paths=file_path, loc=loc, input_size=(25, 25), pre_post_frame=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
            shuffle=True, **param
        )
        val_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=5, allowed_rotation=[0], allowed_flip=[-1], shuffle=True,
            **param
        )

        net = Network(
            train_generator=train_gen, val_generator=val_gen, n_stacks=1, kernel=8, batchNormalize=False,
            use_cpu=True
        )
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None)
        model = net.model

        inf_gen = SubFrameGenerator(
            padding="edge", batch_size=25, allowed_rotation=[0], allowed_flip=[-1], shuffle=False,
            logging_level=logging.DEBUG, **param
        )

        inf_loc = "inf/ch0"
        res = inf_gen.infer(model=model, output=out_path, out_loc=inf_loc, rescale=rescale)

        # Check result
        io = IO()
        data = io.load(file_path, loc=loc, lazy=True)

        if out_path is not None:
            assert out_path.is_file(), "cannot find output file: {}".format(out_path)
            res = io.load(out_path, loc=inf_loc, lazy=True)

        assert res.shape == data.shape, f"inferred output has incorrect shape: " \
                                        f"orig>{data.shape} vs. inf>{res.shape}"
