import tempfile
import time

import pytest

from astroCAST.reduction import *
from astroCAST.helper import DummyGenerator

DG_equal = DummyGenerator()
DG_ragged = DummyGenerator(ragged=True)

class Test_FeatureExtraction:

    @pytest.mark.parametrize("ragged", [True, False])
    def test_extraction(self, ragged):

        DG = DummyGenerator(ragged=ragged)
        events = DG.get_events()

        FE = FeatureExtraction(events)
        features = FE.get_features()

        assert len(events) == len(features)

    def test_add_columns(self):

        DG = DummyGenerator(ragged=False)
        events = DG.get_events()

        FE = FeatureExtraction(events)
        features = FE.get_features(additional_columns=["dz", "dy"])

        assert len(events) == len(features)

    def test_local_caching(self):

        DG = DummyGenerator(ragged=False)
        events = DG.get_events()

        with tempfile.TemporaryDirectory() as dir:
            tmp_path = Path(dir)
            assert tmp_path.is_dir()

            FE = FeatureExtraction(events, cache_path=tmp_path)
            t0 = time.time()
            features_1 = FE.get_features(additional_columns=["dz", "dy"])
            d1 = time.time() - t0

            FE = FeatureExtraction(events, cache_path=tmp_path)
            t0 = time.time()
            features_2 = FE.get_features(additional_columns=["dz", "dy"])
            d2 = time.time() - t0

            assert d2 < d1, f"caching is taking too long: {d2} > {d1}"
            assert features_1.equals(features_2)

class Test_CNN:

    def test_training(self):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        cnn.train(data, epochs=2)

    def test_training_modified(self):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        cnn.train(data, epochs=2, dropout=0.1, regularize_latent=0.01)

    def test_embeding(self):
        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        _, X_test, _, _ = cnn.train(data, epochs=2)

        Y_test = cnn.embed(X_test)

    def test_plotting(self):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        hist, X_test, Y_test, MSE = cnn.train(data, epochs=1)

        cnn.plot_history()
        cnn.plot_examples(X_test, Y_test)

    def test_save_load(self, tmp_path):

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        hist, X_test, Y_test, MSE = cnn.train(data, epochs=1)
        cnn.save_model(tmp_path)

        cnn_naive = CNN()
        cnn_naive.load_model(tmp_path)
        cnn_naive.embed(X_test)

class Test_UMAP:

    def test_training(self):

        data = np.random.random(size=(12, 25))

        um = UMAP()
        embedded = um.train(data)

    def test_plotting(self):

        data = np.random.random(size=(12, 25))

        um = UMAP()
        embedded = um.train(data)

        # vanilla
        um.plot(use_napari=False)

        # custom axis
        fig, ax = plt.subplots(1, 1)
        um.plot(ax=ax, use_napari=False)

        # napari
        um.plot(data=embedded)

        # napari
        labels = np.random.randint(0, 5, size=len(data))
        um.plot(data=embedded, labels=labels)

    def test_save_load(self, tmp_path):

        data = np.random.random(size=(12, 25))

        um = UMAP()
        embedded = um.train(data)

        um.save(tmp_path)

        um = UMAP()
        um.load(tmp_path)
        embedded = um.embed(data)



