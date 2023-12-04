from astrocast.autoencoders import CNN_Autoencoder, TimeSeriesRnnAE, PaddedDataLoader
from astrocast.helper import DummyGenerator


class TestAutoEncoders:

    def setup_class(self):
        pass

    def teardown_class(self):
        pass

    def test_cnn(self, trace_length=16, num_rows=128):
        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=False)
        data = DG.get_array()

        cnn = CNN_Autoencoder(target_length=trace_length)

        train_dataset, val_dataset, test_dataset = cnn.split_dataset(data)
        cnn.train_autoencoder(train_dataset, val_dataset, test_dataset, patience=1, epochs=1)

        latent = cnn.embed(data[:4])
        assert latent is not None

    def test_rnn(self, trace_length=16, num_rows=128):
        DG = DummyGenerator(num_rows=num_rows, trace_length=trace_length, ragged=True)
        data = DG.get_list()

        pdl = PaddedDataLoader(data)
        X_train, X_val, X_test = pdl.get_datasets(batch_size=16)

        rnn = TimeSeriesRnnAE()
        rnn.train_epochs(dataloader_train=X_train, dataloader_val=X_val, num_epochs=2)

        latent = rnn.embedd(X_test)
        assert latent is not None
