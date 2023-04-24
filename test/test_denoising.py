from astroCAST.denoising import FullFrameGenerator, SubFrameGenerator
from astroCAST.denoising import Network
import os

def test_generator_full_frame(file_path, pre_post_frame):

    gen = FullFrameGenerator(file_path=file_path, loc="data/ch0", pre_post_frame=pre_post_frame,
                             batch_size=25)

    for ep in range(2):
        for item in gen:
            pass

        gen.on_epoch_end()

def test_generator_sub_frame(file_path, pre_post_frame):

    gen = SubFrameGenerator(paths=file_path, loc="data/ch0", pre_post_frame=pre_post_frame,
                             input_size=(25, 25), batch_size=25,)

    for ep in range(2):
        for item in gen:
            pass

        gen.on_epoch_end()

def test_network(file_path):

    param = dict(paths=file_path, loc="data/ch0", input_size=(25, 25), pre_post_frame=5, gap_frames=0,
                 normalize="global", cache_results=True, in_memory=True)

    train_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=50,
                                   allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)

    net = Network(train_generator=train_gen, val_generator=train_gen, n_stacks=1, kernel=4, batchNormalize=False, use_cpu=True)
    net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)

def test_inference_full(file_path, out_path=None):

    gen = FullFrameGenerator(file_path=file_path, loc="data/ch0", pre_post_frame=5,
                             batch_size=25, total_samples=50)

    net = Network(train_generator=gen, val_generator=gen, n_stacks=1, kernel=8, batchNormalize=False, use_cpu=True)
    net.run(batch_size=gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)
    model = net.model

    # out_path = "testdata/sample_0_inf.tiff"
    gen.infer(model=model, output=out_path, )

    if out_path is not None:
        assert os.path.isfile(out_path), "cannot find output tiff: {}".format(out_path)
        os.remove(out_path)

def test_inference_sub(file_path, out_path=None):

    param = dict(paths=file_path, loc="data/ch0", input_size=(25, 25), pre_post_frame=5, gap_frames=0,
                 normalize="global", cache_results=True, in_memory=True)

    train_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=50,
                                   allowed_rotation=[1, 2, 3], allowed_flip=[0, 1], shuffle=True, **param)
    val_gen = SubFrameGenerator(padding=None, batch_size=25, max_per_file=5,
                                   allowed_rotation=[0], allowed_flip=[-1], shuffle=True, **param)

    net = Network(train_generator=train_gen, val_generator=val_gen, n_stacks=1, kernel=8, batchNormalize=False, use_cpu=True)
    net.run(batch_size=train_gen.batch_size, num_epochs=2, patience=1, min_delta=0.01, save_model=None, load_weights=False)
    model = net.model

    inf_gen = SubFrameGenerator(padding="edge", batch_size=25, allowed_rotation=[0], allowed_flip=[-1],
                                shuffle=False, max_per_file=2, **param)

    # out_path = "testdata/sample_0_inf.tiff"
    inf_gen.infer(model=model, output=out_path, rescale=False)

    if out_path is not None:
        assert os.path.isfile(out_path), "cannot find output tiff: {}".format(out_path)
        os.remove(out_path)


