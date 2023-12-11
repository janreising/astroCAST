import logging
from pathlib import Path

import pytest

from astrocast.denoising import Network, PyTorchNetwork, SubFrameDataset
from astrocast.denoising import SubFrameGenerator
from astrocast.helper import SampleInput
from astrocast.preparation import IO


class TestDenosingTensorflow:
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
    @pytest.mark.parametrize("pre_post_frames", [5, (3, 2)])
    @pytest.mark.parametrize("gap_frames", [0, 2, (2, 1)])
    def test_generator_sub_vanilla(
            self, tmpdir, extension, pre_post_frames, gap_frames, normalize=None, random_offset=False, overlap=None,
            z_select=None, drop_frame_probability=None,
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("drop_frame_probability", [0.05, 0.5])
    def test_generator_sub_drop_frame(
            self, drop_frame_probability, extension=".h5", pre_post_frames=5, gap_frames=0, normalize=None,
            random_offset=False, overlap=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("normalize", ["local", "global"])
    def test_generator_sub_normalize(
            self, normalize, drop_frame_probability=None, extension=".h5", pre_post_frames=5, gap_frames=0,
            random_offset=False, overlap=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("overlap", [0, 2, 0.1])
    def test_generator_sub_overlap(
            self, overlap, drop_frame_probability=None, extension=".h5", pre_post_frames=5, gap_frames=0,
            random_offset=False, normalize=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("random_offset", [True])
    def test_generator_sub_random_offset(
            self, random_offset, drop_frame_probability=None, extension=".h5", pre_post_frames=5, gap_frames=0,
            overlap=None, normalize=None, z_select=None
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("z_select", [(0, 25), (0, 100)])
    def test_generator_sub_z_select(
            self, z_select, drop_frame_probability=None, extension=".h5", pre_post_frames=5, gap_frames=0, overlap=None,
            normalize=None, random_offset=False
    ):

        file_path, loc = self.data[extension]
        gen = SubFrameGenerator(
            paths=file_path, loc=loc, pre_post_frames=pre_post_frames, gap_frames=gap_frames,
            random_offset=random_offset,
            overlap=overlap, z_select=z_select, drop_frame_probability=drop_frame_probability, input_size=(25, 25),
            batch_size=25, normalize=normalize
        )

        for ep in range(2):
            for _ in gen:
                pass

            gen.on_epoch_end()

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("n_stacks", [1, 3])
    def test_network(self, extension, n_stacks, kernels=4):

        file_path, loc = self.data[extension]

        param = dict(
            paths=file_path, loc=loc, input_size=(32, 32), pre_post_frames=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
            shuffle=True,
            **param
        )

        net = Network(train_generator=train_gen, val_generator=train_gen, n_stacks=n_stacks, kernel=kernels,
                      batchNormalize=False, use_cpu=True)

        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01)

    def test_network_retrain(self, extension=".h5"):

        file_path, loc = self.data[extension]

        param = dict(
            paths=file_path, loc=loc, input_size=(25, 25), pre_post_frames=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
            shuffle=True,
            **param
        )

        net = Network(train_generator=train_gen, val_generator=train_gen, n_stacks=1, kernel=4,
                      batchNormalize=False, use_cpu=True)
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01)

        net.retrain_model(5, 5)

    def test_network_save(self, tmpdir, extension=".h5", n_stacks=2, kernels=4, kernel_size=3):

        file_path, loc = self.data[extension]

        save_model_dir = Path(tmpdir.strpath)

        param = dict(
            paths=file_path, loc=loc, input_size=(25, 25), pre_post_frames=5, gap_frames=0, normalize="global",
            cache_results=True, in_memory=True
        )

        save_model_path = save_model_dir.joinpath("checkpoints/")

        train_gen = SubFrameGenerator(
            padding=None, batch_size=25, max_per_file=50, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
            shuffle=True, **param
        )

        net = Network(train_generator=train_gen, val_generator=train_gen, n_stacks=n_stacks, kernel=kernels,
                      batchNormalize=False, use_cpu=True)
        net.run(
            batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=save_model_path
        )

        assert save_model_path.exists()

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("output_file", [None, "inf.tiff", "inf.h5"])
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize("n_stacks", [1, 2])
    def test_inference_sub(self, tmpdir, extension, output_file, rescale, n_stacks, inf_loc="inf/ch0"):

        file_path, loc = self.data[extension]

        tmpdir = Path(tmpdir.strpath)
        assert tmpdir.is_dir()

        if output_file is None:
            out_path = None
        else:
            out_path = tmpdir.joinpath(output_file)

        param = dict(
            paths=file_path, loc=loc, input_size=(32, 32), pre_post_frames=5, gap_frames=0, normalize="global",
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

        net = Network(train_generator=train_gen, val_generator=val_gen, n_stacks=n_stacks, kernel=8,
                      batchNormalize=False, use_cpu=True)
        net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None)

        model = net.model

        inf_gen = SubFrameGenerator(
            padding="edge", batch_size=25, allowed_rotation=[0], allowed_flip=[-1], shuffle=False,
            logging_level=logging.DEBUG, **param
        )

        res = inf_gen.infer(model=model, output=out_path, out_loc=inf_loc, rescale=rescale)

        # Check result
        io = IO()
        data = io.load(file_path, loc=loc, lazy=True)

        if out_path is not None:
            assert out_path.is_file(), "cannot find output file: {}".format(out_path)
            res = io.load(out_path, loc=inf_loc, lazy=True)

        assert res.shape == data.shape, f"inferred output has incorrect shape: " \
                                        f"orig>{data.shape} vs. inf>{res.shape}"


class TestDatasetPytorch:
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

    def _meta_generator(self, extension=".h5", params=None):

        file_path, loc = self.data[extension]
        param = dict(paths=file_path, loc=loc, padding=None, max_per_file=50, allowed_rotation=[1, 2, 3],
                     allowed_flip=[0, 1], pre_post_frames=5, gap_frames=0, normalize=None,
                     random_offset=False, overlap=None, z_select=None)

        if params is not None:
            for key, val in params.items():
                logging.warning(f"Parameter {key}: {val}")
                param[key] = val

        train_dataset = SubFrameDataset(**param)

        for ep in range(2):
            for _ in train_dataset:
                pass

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    @pytest.mark.parametrize("pre_post_frames", [5, (3, 2)])
    @pytest.mark.parametrize("gap_frames", [0, 2, (2, 1)])
    def test_extensions_and_input_stack(self, extension, pre_post_frames, gap_frames):
        params = dict(pre_post_frames=pre_post_frames, gap_frames=gap_frames)
        self._meta_generator(extension=extension, params=params)

    @pytest.mark.parametrize("drop_frame_probability", [0.05, 0.5])
    def test_generator_sub_drop_frame(self, drop_frame_probability):
        params = dict(drop_frame_probability=drop_frame_probability)
        self._meta_generator(params=params)

    @pytest.mark.parametrize("normalize", ["local", "global"])
    def test_generator_sub_normalize(self, normalize):
        params = dict(normalize=normalize)
        self._meta_generator(params=params)

    @pytest.mark.parametrize("overlap", [0, 2, 0.1])
    def test_generator_sub_overlap(self, overlap):
        params = dict(overlap=overlap)
        self._meta_generator(params=params)

    @pytest.mark.parametrize("random_offset", [True])
    def test_generator_sub_random_offset(self, random_offset):
        params = dict(random_offset=random_offset)
        self._meta_generator(params=params)

    @pytest.mark.parametrize("z_select", [(0, 25), (0, 100)])
    def test_generator_sub_z_select(self, z_select):
        params = dict(z_select=z_select)
        self._meta_generator(params=params)


class TestNetworkPytorch:
    data = None
    si_objects = None
    default_network = None
    base_param = dict(input_size=(32, 32), pre_post_frames=5, gap_frames=0, normalize="global", cache_results=True,
                      in_memory=True, padding=None, max_per_file=12, allowed_rotation=[1, 2, 3], allowed_flip=[0, 1],
                      random_offset=False, overlap=None, z_select=None)

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

    def _meta_network(self, net=None, n_stacks=2, kernels=4, batch_size=4, extension=".h5", train_params=None,
                      val_params=None, test_params=None, inf_params=None, num_epochs=2, patience=1, min_delta=0.01,
                      save_model=None, load_model=None, output=None, rescale=False):

        file_path, loc = self.data[extension]

        # create base param
        base_param = self.base_param.copy()
        base_param["paths"] = file_path
        base_param["loc"] = loc
        base_param["paths"] = file_path

        # create train param
        train_param = base_param.copy()
        if train_params is not None:
            for key, val in train_params.items():
                train_param[key] = val

        train_dataset = SubFrameDataset(**train_param)

        # create val_dataset
        if val_params is not None:
            val_param = base_param.copy()
            for key, val in val_params.items():
                val_param[key] = val

            val_param['input_size'] = train_param['input_size']
            val_param['pre_post_frames'] = train_param['pre_post_frames']
            val_param['gap_frames'] = train_param['gap_frames']
            val_dataset = SubFrameDataset(**val_param)

        else:
            val_dataset = None

        # create train_dataset
        if test_params is not None:
            test_param = base_param.copy()
            for key, val in val_params.items():
                test_param[key] = val

            test_param['input_size'] = train_param['input_size']
            test_param['pre_post_frames'] = train_param['pre_post_frames']
            test_param['gap_frames'] = train_param['gap_frames']
            test_dataset = SubFrameDataset(**test_param)

        else:
            test_dataset = None

        if net is None:
            net = PyTorchNetwork(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                                 shuffle=True, n_stacks=n_stacks, kernels=kernels, batch_size=batch_size,
                                 batch_normalize=False, use_cpu=True, load_model=load_model)

            net.run(num_epochs=num_epochs, patience=patience, min_delta=min_delta, save_model=save_model)

        if inf_params is not None:
            inf_param = base_param.copy()
            for key, val in inf_params.items():
                inf_param[key] = val

            inf_param['input_size'] = train_param['input_size']
            inf_param['pre_post_frames'] = train_param['pre_post_frames']
            inf_param['gap_frames'] = train_param['gap_frames']

            inf_dataset = SubFrameDataset(**inf_param)
            inf_loc = "inf/ch0"
            res = net.infer(inf_dataset, output=output, out_loc=inf_loc, batch_size=batch_size, rescale=rescale)

            # Check result
            io = IO()
            data = io.load(file_path, loc=loc, lazy=True)

            if output is not None:
                assert output.is_file(), "cannot find output file: {}".format(output)
                res = io.load(output, loc=inf_loc, lazy=True)

            assert res.shape == data.shape, f"inferred output has incorrect shape: " \
                                            f"orig>{data.shape} vs. inf>{res.shape}"

        else:
            return net

    def test_train_val_test(self):

        val_params = dict(max_per_file=5, allowed_rotation=0, allowed_flip=-1)
        test_params = dict(max_per_file=5, allowed_rotation=0, allowed_flip=-1)

        self._meta_network(val_params=val_params, test_params=test_params)

    @pytest.mark.parametrize("n_stacks", [1, 3])
    def test_n_stacks(self, n_stacks):
        self._meta_network(n_stacks=n_stacks)

    @pytest.mark.parametrize("kernels", [8, 15])
    def test_n_stacks(self, kernels):
        self._meta_network(kernels=kernels)

    @pytest.mark.parametrize("input_size", [(16, 16), (30, 30), (16, 32)])
    def test_input_size(self, input_size):
        train_params = dict(input_size=input_size)
        self._meta_network(train_params=train_params)

    @pytest.mark.parametrize("pre_post_frames", [2, (1, 2)])
    def test_pre_post_frames(self, pre_post_frames):
        train_params = dict(pre_post_frames=pre_post_frames)
        self._meta_network(train_params=train_params)

    @pytest.mark.parametrize("gap_frames", [2, (1, 2)])
    def test_gap_frames(self, gap_frames):
        train_params = dict(gap_frames=gap_frames)
        self._meta_network(train_params=train_params)

    @pytest.mark.parametrize("normalize", ["local", None])
    def test_input_size(self, normalize):
        train_params = dict(normalize=normalize)
        self._meta_network(train_params=train_params)

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_extension(self, extension):
        self._meta_network(extension=extension)

    def test_retrain(self):
        net = self._meta_network()
        net.retrain_model(5, 5)

    def test_save_load(self, tmpdir):

        save_model_dir = Path(tmpdir.strpath)
        save_model_path = save_model_dir.joinpath("model.pth")

        net = self._meta_network(save_model=save_model_path)
        assert save_model_path.exists()

        net2 = self._meta_network(load_model=save_model_path)

    @pytest.mark.parametrize("extension", [".h5", ".tiff"])
    def test_infer_from_extension(self, extension):

        if self.default_network is None:
            self.default_network = self._meta_network()

        net = self.default_network

        # create base param
        inf_param = dict(padding="edge", allowed_rotation=0, allowed_flip=-1, max_per_file=None)
        self._meta_network(net=net, inf_params=inf_param)

    @pytest.mark.parametrize("output_file", ["inf.tiff", "inf.h5"])
    def test_infer_output(self, tmpdir, output_file):

        if self.default_network is None:
            self.default_network = self._meta_network()

        net = self.default_network

        # create base param
        inf_param = dict(padding="edge", allowed_rotation=0, allowed_flip=-1, max_per_file=None)
        output_file = Path(tmpdir.strpath).joinpath(output_file)
        self._meta_network(net=net, inf_params=inf_param, output=output_file)

    def test_infer_rescale(self):

        if self.default_network is None:
            self.default_network = self._meta_network()

        net = self.default_network

        # create base param
        inf_param = dict(padding="edge", allowed_rotation=0, allowed_flip=-1, max_per_file=None)
        self._meta_network(net=net, inf_params=inf_param, rescale=True)
