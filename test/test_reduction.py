import tempfile
import time
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage

from astrocast.autoencoders import CNN_Autoencoder
from astrocast.helper import DummyGenerator
from astrocast.reduction import FeatureExtraction, UMAP, ClusterTree

matplotlib.use('Agg')  # Use the Agg backend

DG_equal = DummyGenerator()
DG_ragged = DummyGenerator(ragged=True)


class TestFeatureExtraction:

    @pytest.mark.parametrize("ragged", [True, False])
    def test_extraction(self, ragged):
        DG = DummyGenerator(ragged=ragged)
        events = DG.get_events()

        FE = FeatureExtraction(events)
        features = FE.all_features()

        assert len(events) == len(features)

    def test_local_caching(self):
        DG = DummyGenerator(ragged=False)
        events = DG.get_events()

        with tempfile.TemporaryDirectory() as dir:
            tmp_path = Path(dir)
            assert tmp_path.is_dir()

            FE = FeatureExtraction(events, cache_path=tmp_path)
            t0 = time.time()
            features_1 = FE.all_features()
            d1 = time.time() - t0

            FE = FeatureExtraction(events, cache_path=tmp_path)
            t0 = time.time()
            features_2 = FE.all_features()
            d2 = time.time() - t0

            assert d2 < d1, f"caching is taking too long: {d2} > {d1}"
            assert features_1.equals(features_2)


class TestCNN:

    @pytest.mark.tensorflow
    def test_training(self):
        trace_length = 16
        DG = DummyGenerator(num_rows=32, trace_length=trace_length, ragged=False)
        data = DG.get_array()

        cnn = CNN_Autoencoder(target_length=trace_length)
        train_dataset, val_dataset, test_dataset = cnn.split_dataset(data)
        losses = cnn.train_autoencoder(
            X_train=train_dataset, X_val=val_dataset, X_test=test_dataset, epochs=2, batch_size=4
        )

    @pytest.mark.tensorflow
    def test_embeding(self):
        trace_length = 16
        DG = DummyGenerator(num_rows=32, trace_length=trace_length, ragged=False)
        data = DG.get_array()

        cnn = CNN_Autoencoder(target_length=trace_length)
        train_dataset, val_dataset, test_dataset = cnn.split_dataset(data)
        losses = cnn.train_autoencoder(
            X_train=train_dataset, X_val=val_dataset, X_test=test_dataset, epochs=2, batch_size=4
        )

        Y_test = cnn.embed(data)

    @pytest.mark.tensorflow
    def test_plotting(self):
        trace_length = 16
        DG = DummyGenerator(num_rows=32, trace_length=trace_length, ragged=False)
        data = DG.get_array()

        cnn = CNN_Autoencoder(target_length=trace_length)
        train_dataset, val_dataset, test_dataset = cnn.split_dataset(data)
        losses = cnn.train_autoencoder(
            X_train=train_dataset, X_val=val_dataset, X_test=test_dataset, epochs=2, batch_size=4
        )

        cnn.plot_examples_pytorch(test_dataset)

    @pytest.mark.tensorflow
    def test_save_load(self, tmp_path, target_length=18):
        with tempfile.TemporaryDirectory() as temp:
            tempdir = Path(temp)
            save_path = tempdir.joinpath("model_paramters.pth")

            autoencoder = CNN_Autoencoder(target_length=target_length)
            autoencoder.save(save_path)

            loaded_model = CNN_Autoencoder.load(save_path, target_length=target_length)


class TestUMAP:

    @classmethod
    def setup_class(cls):
        pytest.importorskip("umap")

    def test_training(self):

        data = np.random.random(size=(12, 25))

        um = UMAP()
        embedded = um.train(data)

    @pytest.mark.skip(reason="Fails on automatic tests")
    @pytest.mark.vis
    @pytest.mark.parametrize("use_napari", [True, False])
    @pytest.mark.parametrize("use_data", [True, False])
    @pytest.mark.parametrize("custom_axis", [True, False])
    @pytest.mark.parametrize("custom_labels", [True, False])
    def test_plotting_extensive(self, use_napari, custom_axis, custom_labels, use_data):

        data = np.random.random(size=(12, 8))

        um = UMAP()
        embedded = um.train(data)

        ax = None
        data_emb = None
        labels = None

        if custom_axis:
            _, ax = plt.subplots(1, 1)

        if use_data:
            data_emb = embedded

        if custom_labels:
            labels = np.random.randint(0, 5, size=len(data))

        if use_napari and not use_data:
            with pytest.raises(ValueError):
                um.plot(use_napari=use_napari, ax=ax, data=data_emb, labels=labels)

        else:
            um.plot(use_napari=use_napari, ax=ax, data=data_emb, labels=labels)

    def test_save_load(self, tmp_path):

        data = np.random.random(size=(12, 25))

        um = UMAP()
        embedded_1 = um.train(data)

        um.save(tmp_path)

        um = UMAP()
        um.load(tmp_path)
        embedded_2 = um.embed(data)

        assert np.allclose(embedded_1, embedded_2)


class TestClusterTree:

    def test_creation(self):
        X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
        X = np.array(X)

        Z = linkage(X, 'ward')

        _ = ClusterTree(Z)

    def test_functions(self):
        X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
        X = np.array(X)

        Z = linkage(X, 'ward')

        ct = ClusterTree(Z)
        root = ct.tree

        _ = ct.get_node(0)
        _ = ct.get_leaves(root)
        _ = ct.get_count(root)
        _ = ct.search(root, 0)
