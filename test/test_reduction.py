import pytest

from astroCAST.reduction import *
from astroCAST.helper import DummyGenerator

DG_equal = DummyGenerator()
DG_ragged = DummyGenerator(ragged=True)

class Test_FeatureExtraction:

    @pytest.mark.parametrize(("typ", "feature_only"), [("use_dataframe", False), ("use_list", True), ("use_array", True)])
    @pytest.mark.parametrize("ragged", [True, False])
    def test_input_type(self, typ, feature_only, ragged):

        DG = DG_ragged if ragged else DG_equal

        if typ == "use_dataframe":
            data = DG.get_dataframe()

        elif typ == "use_list":
            data = DG.get_list()

        elif typ == "use_array":
            data = DG.get_array()

        else:
            raise TypeError

        FE = FeatureExtraction()
        FE.get_features(data=data, feature_only=feature_only)

    @pytest.mark.parametrize("normalize", [None, "min_max"])
    def test_normalization(self, normalize):

        data = DG_ragged.get_dataframe()

        FE = FeatureExtraction()
        FE.get_features(data=data, normalize=normalize)

    @pytest.mark.parametrize("padding", [None, "edge"])
    def test_padding(self, padding):

        data = DG_ragged.get_dataframe()

        FE = FeatureExtraction()
        FE.get_features(data=data, normalize=padding)

    def test_local_caching(self):
        raise NotImplementedError

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



