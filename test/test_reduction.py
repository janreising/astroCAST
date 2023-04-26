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

    def test_save_load(self):

        path = ""

        DG = DummyGenerator(num_rows=11, trace_length=16, ragged=False)
        data = DG.get_array()

        cnn = CNN()
        hist, X_test, Y_test, MSE = cnn.train(data, epochs=1)
        cnn.save_model(path)

        cnn_naive = CNN()
        cnn_naive.load_model(path)
        cnn_naive.predict(X_test)
