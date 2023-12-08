from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm


# Convolutional Neural Network Autoencoder
# ----------------------------------------
class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: Union[int, float] = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, loss):

        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss - self.min_loss <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x


class CustomUpsample(nn.Module):
    def __init__(self, target_length):
        super(CustomUpsample, self).__init__()
        self.target_length = target_length

    def forward(self, x):
        current_length = x.shape[-1]
        diff = self.target_length - current_length
        zeros = torch.zeros((x.shape[0], x.shape[1], diff)).to(x.device)
        return torch.cat([x, zeros], dim=-1)


class CNN_Autoencoder(nn.Module):
    """
    Convolutional Neural Network Autoencoder.

    This class defines a convolutional autoencoder model using PyTorch. An autoencoder is a neural network architecture
    designed for tasks such as dimensionality reduction, feature learning, and data denoising.

    Parameters:
        target_length (int): trace lenght.
    
    Example:
        autoencoder = CNN_Autoencoder(target_length=18, dropout=0.2, latent_size=128, add_noise=0.1)
    """

    def __init__(
            self, target_length: int, dropout: float = 0.15, l1_reg: float = 0.0001, latent_size=64 * 6, add_noise=None
    ):
        super(CNN_Autoencoder, self).__init__()

        self.losses = None
        self.l1_reg = l1_reg
        self.latent_size = latent_size

        self.encoder, self.decoder = self.define_layers(
            dropout=dropout, add_noise=add_noise, target_length=target_length
        )

        # Manually defining a linear layer to serve as the dense layer for the encoder output
        self.latent_size = latent_size
        self.dense_layer = None
        self.dense_layer_out = None
        self.add_noise = add_noise

    @staticmethod
    def define_layers(dropout=None, add_noise=None, target_length=18):

        encoder_layers = []
        if dropout is not None:
            encoder_layers += [nn.Dropout(dropout)]
        if add_noise is not None:
            encoder_layers += [GaussianNoise(std=add_noise)]
        encoder_layers += [nn.Conv1d(1, 128, 3, padding=1), ]
        encoder_layers += [nn.ReLU(), ]
        encoder_layers += [nn.MaxPool1d(2), ]
        encoder_layers += [nn.Conv1d(128, 64, 3, padding=1), ]
        encoder_layers += [nn.ReLU(), ]
        encoder_layers += [nn.MaxPool1d(2), ]
        encoder = nn.Sequential(*encoder_layers)

        decoder = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2), nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(), nn.Upsample(scale_factor=2), CustomUpsample(target_length=target_length),
            nn.Conv1d(128, 1, 3, padding=1), nn.Sigmoid()
        )

        return encoder, decoder

    def forward(self, x):
        x = self.encoder(x)

        shape_before_flatten = x.shape[1:]
        flattened_dim = torch.prod(torch.tensor(shape_before_flatten))

        if self.dense_layer is None:
            self.dense_layer = nn.Linear(flattened_dim.item(), self.latent_size).to(x.device)
            self.dense_layer_out = nn.Linear(self.latent_size, flattened_dim.item()).to(x.device)

        x = x.view(-1, flattened_dim)

        # Apply the dense layer
        x = self.dense_layer(x)

        # Make the output binary
        x = torch.sigmoid(x)
        x = torch.round(x)

        # Save the encoder output for later use
        encoder_output = x

        # go back to initial size
        x = self.dense_layer_out(x)

        # Add L1 regularization to the encoder output
        l1_loss = self.l1_reg * torch.norm(x, 1)

        x = x.view(-1, *shape_before_flatten)
        x = self.decoder(x)

        return x, l1_loss, encoder_output

    @staticmethod
    def split_dataset(data, val_split=0.1, train_split=0.8, seed=None):

        from torch.utils.data import random_split

        # Assuming you have a PyTorch Dataset object `full_dataset`
        # This could be an instance of a custom dataset class, or one of the built-in classes like `torchvision.datasets.MNIST`

        if seed is not None:
            torch.manual_seed(seed)

        # ensure equal length input
        unique_length = np.unique([len(data[i, :]) for i in range(len(data))])
        if len(unique_length) > 1:
            raise ValueError(f"input contains unequal length sequences: {unique_length}! aborting.")

        # Define the proportions
        total_size = len(data)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset

    def train_autoencoder(
            self, X_train, X_val=None, X_test=None, patience=5, min_delta=0.0005, epochs=100, learning_rate=0.001,
            batch_size=32
    ):

        # ensure equal length input
        for X in [X_train, X_val, X_test]:
            if X is not None:
                unique_length = np.unique([len(X_train[i]) for i in range(len(X))])
                if len(unique_length) > 1:
                    raise ValueError(f"input contains unequal length sequences: {unique_length}! aborting.")

        # Create DataLoader
        train_data = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(train_data, train_data)  # autoencoders use same data for input and output
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_data = torch.FloatTensor(X_val)
            val_dataset = TensorDataset(val_data, val_data)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        if X_test is not None:
            test_data = torch.FloatTensor(X_test)
            test_dataset = TensorDataset(test_data, test_data)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None

        # Initialize model
        model = self
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop

        losses = []
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_data, _ in train_loader:  # autoencoders don't use labels
                batch_data = batch_data.unsqueeze(1)  # add channel dimension

                # Forward pass
                outputs, l1_loss, encoded = model(batch_data)

                # Compute loss
                reconstruction_loss = criterion(outputs, batch_data)
                loss = reconstruction_loss + l1_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data, _ in val_loader:
                        batch_data = batch_data.unsqueeze(1)  # add channel dimension
                        outputs, l1_loss, encoded = model(batch_data)

                        reconstruction_loss = criterion(outputs, batch_data)
                        loss = reconstruction_loss + l1_loss
                        val_loss += loss.item()

                model.train()
            else:
                val_loss = None

            losses.append((train_loss, val_loss))

            # Early stopping logic
            if early_stopper(val_loss):
                print("Early stopping!")
                break

            pbar.set_description(
                f"train_Loss:{train_loss / len(train_loader):.4f}, val_loss:{val_loss / len(val_loader):.4f}"
            )
            pbar.update()
        pbar.close()

        # Evaluation
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for batch_data, _ in test_loader:
                    batch_data = batch_data.unsqueeze(1)  # add channel dimension
                    outputs, l1_loss, encoded = model(batch_data)

                    reconstruction_loss = criterion(outputs, batch_data)
                    loss = reconstruction_loss + l1_loss
                    test_loss += loss.item()

                print(f"Test Loss: {test_loss / len(test_loader):.4f}")
            model.train()

        self.losses = losses
        return losses

    def embed(self, data, batch_size=64):

        # Create DataLoader
        train_data = torch.FloatTensor(data)
        train_dataset = TensorDataset(train_data, train_data)  # autoencoders use same data for input and output
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = self
        encoded_traces = []
        for batch_data, _ in train_loader:
            batch_data = batch_data.unsqueeze(1)
            _, _, encoded = model(batch_data)
            encoded = encoded.detach().numpy()

            encoded_traces.append(encoded)

        encoded_traces = np.vstack(encoded_traces)

        return encoded_traces

    @staticmethod
    def reshape_to_square_matrix(vector):
        """
        Reshapes a 1D vector to a square-ish 2D matrix.
        """

        import math

        length = len(vector)
        # Find the closest integer square root of the length
        sqrt_length = int(math.sqrt(length))

        # Find the dimensions of the reshaped matrix
        rows = sqrt_length
        while length % rows != 0:
            rows -= 1
        cols = length // rows

        # Reshape the vector
        reshaped_matrix = np.reshape(vector, (rows, cols))

        return reshaped_matrix

    def plot_examples_pytorch(
            self, X_test, Y_test=None, show_diff=False, num_samples=9, figsize=(10, 6)
    ):

        model = self

        # Convert PyTorch tensor to numpy
        X_test = np.array(list(X_test))

        # Make predictions if Y_test is not provided
        if Y_test is None:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # add channel dimension
                output_tensor, l1_loss, encoder_output = model(input_tensor)
            Y_test = output_tensor.numpy().squeeze()
            encoder_output = encoder_output.numpy().squeeze()

        else:
            encoder_output = Y_test

        if show_diff and Y_test is not None:
            consensus = np.mean(encoder_output, axis=0)
            encoder_output = encoder_output - consensus
        else:
            if not isinstance(Y_test, np.ndarray):
                Y_test = Y_test.numpy().squeeze()
            else:
                Y_test = np.squeeze(Y_test)

        num_rounds = 1
        if not isinstance(num_samples, int):
            num_rounds, num_samples = num_samples

        fig = None
        for nr in range(num_rounds):
            fig, axx = plt.subplots(3, num_samples, figsize=figsize)

            for i, idx in enumerate(np.random.randint(0, len(X_test) - 1, size=num_samples)):
                inp = X_test[idx, :]
                out = Y_test[idx, :]

                inp = np.trim_zeros(inp, trim="b")
                out = out[0:len(inp)]

                axx[0, i].plot(inp, alpha=0.75, color="black")
                axx[0, i].plot(out, alpha=0.75, linestyle="--", color="darkgreen")
                axx[1, i].plot(inp - out)

                axx[0, i].sharey(axx[1, i])

                latent_output = encoder_output[idx, :]
                latent_output = self.reshape_to_square_matrix(latent_output)

                cmap = 'binary' if not show_diff else 'bwr'
                min_value = 0 if not show_diff else -1
                axx[2, i].imshow(
                    latent_output, cmap=cmap, interpolation='nearest', aspect='auto', vmin=min_value, vmax=1
                )
                axx[2, i].get_xaxis().set_visible(False)
                axx[2, i].get_yaxis().set_visible(False)

                axx[0, i].get_xaxis().set_visible(False)
                axx[1, i].get_xaxis().set_visible(False)

                if i != 0:
                    axx[0, i].get_yaxis().set_visible(False)
                    axx[1, i].get_yaxis().set_visible(False)

            axx[0, 0].set_ylabel("IN/OUT", fontweight=600)
            axx[1, 0].set_ylabel("error", fontweight=600)

        return fig

    def save(self, filepath: Union[str, Path]):
        """
        Save the model parameters to a file.

        Parameters:
        - filepath (str): The location where the model parameters should be saved.

        Example usage:
            model = CNN_Autoencoder(target_length=18)
            model.save("path/to/save/model.pth")
        """

        if isinstance(filepath, Path):
            filepath = filepath.as_posix()

        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], *args, **kwargs):
        """
        Load the model parameters from a file and return an instance of the model.

        Parameters:
        - filepath (str): The location from where the model parameters should be loaded.

        Returns:
        - CNN_Autoencoder: An instance of the CNN_Autoencoder model with loaded parameters.

        Example usage:
            loaded_model = CNN_Autoencoder.load("path/to/save/model.pth", target_length=18)
        """

        if isinstance(filepath, Path):
            filepath = filepath.as_posix()

        model = cls(*args, **kwargs)  # Create a new instance of the model
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set the model to evaluation mode
        return model


# Recurrent Neural Network Autoencoder
# ------------------------------------

class PaddedSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lengths = [len(seq) for seq in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.lengths[index]


class PaddedDataLoader:

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, data, batch_size, shuffle):

        ds = [torch.tensor(x, dtype=torch.float32) for x in data]
        ds = PaddedSequenceDataset(ds)

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return dl

    def get_datasets(
            self, batch_size=(32, "auto", "auto"), val_size=0.15, test_size=0.15, shuffle=(True, False, False)
    ):

        # First, split into training and temp sets
        train_data, temp_data = train_test_split(self.data, test_size=(val_size + test_size))
        # Then split the temp_data into validation and test sets
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)))

        if isinstance(batch_size, int):
            batch_size = (batch_size, batch_size, batch_size)
        elif isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 3:
                raise ValueError(f"please provide batch_size as int or list of length 3.")

        datasets = []
        for i, ds in enumerate([train_data, val_data, test_data]):

            bs = batch_size[i]
            if bs == "auto":
                bs = len(ds)

            dl = self.get_dataloader(ds, batch_size=bs, shuffle=shuffle[i])

            datasets.append(dl)

        return datasets

    @staticmethod
    def collate_fn(batch):
        # Sort sequences by length in descending order
        batch.sort(key=lambda x: x[1], reverse=True)  # x[1] is the length

        # Separate sequence lengths and sequences
        sequences = [x[0] for x in batch]  # x[0] is the data
        lengths = [x[1] for x in batch]  # x[1] is the length

        # Pad sequences
        sequences = pad_sequence(sequences, batch_first=True)

        return sequences, lengths


class TimeSeriesRnnAE:

    def __init__(
            self, num_features=1, rnn_hidden_dim=32, num_layers=2, encoder_lr=0.001, decoder_lr=0.001, clip=0.5,
            dropout=0, initialize_repeat=True, num_directions=1, use_cuda=False
    ):
        """
        Initialize the TimeSeriesRnnAE model.

        Parameters:
        - num_features (int): Dimensionality of the input time-series data.
        - rnn_hidden_dim (int): Number of hidden units in the LSTM layers.
        - num_layers (int): Number of LSTM layers.
        - encoder_lr (float): Learning rate for the encoder model.
        - decoder_lr (float): Learning rate for the decoder model.
        - clip (float): Value used for gradient clipping during training.
        - dropout (float): Dropout rate for LSTM layers.
        - initialize_repeat (bool): Whether to initialize the repeat vector in the decoder.
        - num_directions (int): Number of directions for the LSTM layers.
        - use_cuda (bool): Whether to use GPU acceleration if available.

        """
        super(TimeSeriesRnnAE, self).__init__()

        # set device
        if torch.cuda.is_available() and use_cuda:
            device = torch.device("cuda:0")
        else:
            device = "cpu"

        self.device = device
        self.criterion = nn.MSELoss()
        self.clip = clip

        # Create encoder rnn and decoder rnn module
        self.encoder = Encoder(
            device=device, num_features=num_features, rnn_hidden_dim=rnn_hidden_dim, num_layers=num_layers,
            dropout=dropout, num_directions=num_directions
        )
        self.decoder = Decoder(
            device=self.device, criterion=self.criterion, initialize_repeat=initialize_repeat,
            rnn_hidden_dim=rnn_hidden_dim, num_layers=num_layers, num_features=num_features,
            num_directions=num_directions, dropout=dropout
        )
        self.encoder.to(device)
        self.decoder.to(device)
        # Create optimizers for encoder and decoder
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=decoder_lr)

        # initialize historical variables
        self.learning_rates = None
        self.val_losses = None
        self.train_losses = None

    def update_learning_rates(self, encoder_factor, decoder_factor):
        self.encoder_lr = self.encoder_lr * encoder_factor
        self.decoder_lr = self.decoder_lr * decoder_factor
        self.set_learning_rates(self.encoder_lr, self.decoder_lr)

    def set_learning_rates(self, encoder_lr, decoder_lr):
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = encoder_lr
        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = decoder_lr

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    @staticmethod
    def are_equal_tensors(a, b):
        if torch.all(torch.eq(a, b)).data.cpu().numpy() == 0:
            return False
        return True

    def train_epochs(
            self, dataloader_train, dataloader_val=None, num_epochs=10, diminish_learning_rate=0.99, patience=5,
            min_delta=0.001, smooth_loss_len=3, safe_after_epoch=None, show_mode=None
    ):
        """
        Train the TimeSeriesRnnAE model for multiple epochs.

        Parameters:
        - dataloader_train (DataLoader): DataLoader object for the training data.
        - dataloader_val (DataLoader): DataLoader object for the validation data (optional).
        - num_epochs (int): Maximum number of epochs to train the model.
        - diminish_learning_rate (float): Factor by which to diminish the learning rate after each epoch.
        - patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
        - min_delta (float): Minimum change in validation loss to be considered as improvement for early stopping.
        - smooth_loss_len (int): Number of previous losses to consider for smoothing the loss curve.
        - safe_after_epoch (str, Path): Path to save the final encoder and decoder models to.
        - show_mode (str): Mode for displaying training progress ('progress', 'notebook', or None).

        Returns:
        - epoch_loss (float): The total loss for the last epoch.

        """

        self.train()

        patience_counter = 0
        losses = None

        if show_mode == "progress":
            iterator = tqdm(range(num_epochs), total=num_epochs)
        elif show_mode == "notebook":
            from IPython.display import clear_output
            from IPython.core.display_functions import display
            iterator = range(num_epochs)
        else:
            iterator = range(num_epochs)

        train_losses = []
        val_losses = []
        learning_rates = []
        for epoch in iterator:

            batch_losses = []
            for batch_data, batch_lengths in dataloader_train:
                batch_data = batch_data.unsqueeze(-1)
                batch_data = batch_data.to(dtype=torch.float32).to(self.device)  # Move to device and ensure it's float
                batch_lengths = torch.tensor(batch_lengths, dtype=torch.float32, device=self.device)

                # Pack the batch
                packed_batch_data = pack_padded_sequence(
                    batch_data, batch_lengths.cpu().numpy(), batch_first=True
                )  # .to(self.device)

                # Your existing code for training on a single batch
                batch_loss = self.train_batch(packed_batch_data, batch_lengths)
                batch_losses.append(batch_loss)

            epoch_loss = np.mean(batch_losses)
            train_losses.append(epoch_loss)

            if diminish_learning_rate is not None:
                self.update_learning_rates(diminish_learning_rate, diminish_learning_rate)
            learning_rates.append([self.encoder_lr, self.decoder_lr])

            # model saving
            if safe_after_epoch is not None:

                if not isinstance(safe_after_epoch, (str, Path)):
                    raise ValueError(f"please provide 'safe_after_epoch' as string or pathlib.Path")

                if isinstance(safe_after_epoch, str):
                    safe_after_epoch = Path(safe_after_epoch)

                encoder_file_name = safe_after_epoch.joinpath("_encoder.model")
                decoder_file_name = safe_after_epoch.joinpath("_decoder.model")

                self.save_models(encoder_file_name, decoder_file_name)

            if dataloader_val is not None:
                val_loss = self.evaluate_batch(dataloader_val)
                val_losses.append(val_loss)

            # progress
            if show_mode == "progress":

                descr = f"tloss>{epoch_loss:.2E} "

                if dataloader_val is not None:
                    descr += f"vloss>{val_losses[-1]:.2E} "

                if self.encoder_lr == self.decoder_lr:
                    descr += f"lr>{self.encoder_lr:.2E} "
                else:
                    descr += f"lr>({self.encoder_lr:.2E}, {self.decoder_lr:.2E}) "

                descr += f"P{patience_counter}"

                iterator.set_description(descr)
                iterator.update()

            elif show_mode == "notebook":

                from IPython.core.display_functions import clear_output, display

                plt.clf()
                clear_output(wait=True)

                fig, axx = plt.subplots(1, 2, figsize=(9, 4))

                axx[0].plot(train_losses, color="black", label="training")
                if dataloader_val is not None:
                    axx[0].plot(np.array(val_losses).flatten(), color="green", label="validation")

                axx[0].set_title(f"losses")
                axx[0].set_yscale("log")
                axx[0].legend()

                lrates = np.array(learning_rates)
                axx[1].plot(lrates[:, 0], color="green", label="encoder")
                axx[1].plot(lrates[:, 1], color="red", label="decoder")
                axx[1].set_title(f"learning rates")
                axx[1].legend()

                fig.suptitle(f"Epoch {epoch}/{num_epochs}; Patience {patience_counter}/{patience}")

                plt.tight_layout()

                display(fig)

            # Early stopping
            if dataloader_val is not None:
                losses = val_losses
            else:
                losses = train_losses

            smoothed_loss = np.median(np.array(losses[-smooth_loss_len - 1:-1]))
            if smoothed_loss - losses[-1] > min_delta:
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        self.eval()

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.learning_rates = learning_rates

        return losses

    def train_batch(self, packed_inputs, lengths):
        """
        Train a single batch for the TimeSeriesRnnAE model.

        Parameters:
        - packed_inputs (PackedSequence): The packed input time-series data for this batch.

        Returns:
        - loss (float): The normalized loss for this batch.
        """
        # Get batch size and number of time steps
        batch_size = packed_inputs.batch_sizes[0]  # The first element contains the batch size
        num_steps = packed_inputs.data.size(0)  # Total number of time steps across all sequences

        # Zero the gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Forward pass through the encoder and decoder
        initial_hidden = self.encoder.init_hidden(batch_size)

        z, new_hidden = self.encoder(packed_inputs, initial_hidden)

        loss = self.decoder(packed_inputs, z, lengths)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Update parameters
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / num_steps  # Normalized loss

    def evaluate_batch(self, dataloader, return_outputs=False):

        losses = []
        outputs = []
        for batch_data, batch_lengths in dataloader:

            batch_data = batch_data.unsqueeze(-1)
            batch_data = batch_data.to(dtype=torch.float32).to(self.device)  # Move to device and ensure it's float
            batch_lengths = torch.tensor(batch_lengths, dtype=torch.float32, device=self.device)

            # Pack the batch
            packed_batch_data = pack_padded_sequence(
                batch_data, batch_lengths.cpu().numpy(), batch_first=True
            )  # .to(self.device)

            batch_size = packed_batch_data.batch_sizes[0]  # The first element contains the batch size

            # encode
            initial_hidden = self.encoder.init_hidden(batch_size)
            z, new_hidden = self.encoder(packed_batch_data, initial_hidden)

            # decode
            res = self.decoder(packed_batch_data, z, batch_lengths, return_outputs=return_outputs)

            if return_outputs:
                loss, output = res
                outputs.append(output)
            else:
                loss = res
            losses.append(loss.detach().numpy())

        losses = np.mean(np.array(losses))

        if return_outputs:
            return losses, outputs
        else:
            return losses

    def save_models(self, encoder_file_name, decoder_file_name):
        torch.save(self.encoder.state_dict(), encoder_file_name)
        torch.save(self.decoder.state_dict(), decoder_file_name)

    def load_models(self, encoder_file_name, decoder_file_name):
        self.encoder.load_state_dict(torch.load(encoder_file_name))
        self.decoder.load_state_dict(torch.load(decoder_file_name))

    def embedd(self, dataloader: DataLoader):
        """
            Embeds the data using the trained encoder-decoder model.

            Args:
                dataloader (torch.utils.data.DataLoader): The data loader for the input data.

            Returns:
                Tuple: A tuple containing the embedded input data, the decoded output data, the latent representation, and the losses.

        """
        self.eval()

        x_val = []
        y_val = []
        latent = []
        losses = []
        for batch_data, batch_lengths in tqdm(dataloader):
            batch_data = batch_data.unsqueeze(-1)
            batch_data = batch_data.to(
                dtype=torch.float32
            )  # .to(self.device)  # Move to device and ensure it is a float
            batch_lengths = torch.tensor(batch_lengths, dtype=torch.float32)  # , device=self.device)

            # Pack the batch
            packed_batch_data = pack_padded_sequence(
                batch_data, batch_lengths.cpu().numpy(), batch_first=True
            )  # .to(self.device)
            batch_size = packed_batch_data.batch_sizes[0]  # The first element contains the batch size

            # encode
            encoder = self.encoder
            initial_hidden = encoder.init_hidden(batch_size)
            encoded, _ = encoder(packed_batch_data, initial_hidden)

            # decode
            decoder = self.decoder
            loss, decoded = decoder(packed_batch_data, encoded, batch_lengths, return_outputs=True)

            # Convert batch_data and decoded data to numpy
            x_np = batch_data.cpu().numpy()
            y_np = decoded.cpu().detach().numpy()
            encoded_np = encoded.cpu().detach().numpy()

            # Remove zero padding based on batch_lengths
            x_np = [x_np[i, :int(batch_lengths[i]), :] for i in range(len(batch_lengths))]
            y_np = [y_np[i, :int(batch_lengths[i]), :] for i in range(len(batch_lengths))]

            x_val.extend(x_np)
            y_val.extend(y_np)
            latent.extend(encoded_np)
            losses.append(loss.item())

        return x_val, y_val, latent, losses

    def plot_traces(self, dataloader, figsize=(10, 10), n_samples=16, sharex=False):

        self.eval()

        x_val, y_val, latent, losses = self.embedd(dataloader)

        n_samples = min(len(x_val), n_samples)
        fig, axx = plt.subplots(n_samples, 2, figsize=figsize, sharex=sharex)

        for i, idx in enumerate(np.random.randint(0, len(x_val), size=n_samples)):
            x = np.squeeze(x_val[idx])
            y = np.squeeze(y_val[idx])

            axx[i, 0].plot(x, color="gray", linestyle="--")
            axx[i, 0].plot(y, color="red", linestyle="-")
            axx[i, 0].set_ylabel(f"idx: {idx}")

            if i != 0:
                axx[i, 0].sharex(axx[0, 0])

            latent_vector = latent[i]
            latent_vector = np.reshape(latent_vector, (len(latent_vector), 1)).transpose()
            axx[i, 1].imshow(latent_vector, aspect="auto", cmap="viridis")
            axx[i, 1].axis("off")

        plt.tight_layout()

        return fig, x_val, y_val, latent, losses


class Encoder(nn.Module):

    def __init__(self, device="cpu", num_features=1, rnn_hidden_dim=32, num_layers=2, dropout=0, num_directions=1):
        """ Initializes the Encoder object.
        Parameters:
            device (str): The device to use for computations, default is “cpu”.
            num_features (int): The number of input features, default is 1.
            rnn_hidden_dim (int): The number of hidden units in the RNN, default is 32.
            num_layers (int): The number of layers in the RNN, default is 2.
            dropout (float): The dropout rate, default is 0.
        """
        super(Encoder, self).__init__()

        self.device = device
        self.num_hidden_states = 2
        rnn = nn.LSTM

        self.num_features = num_features
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_directions = num_directions

        self.rnn = rnn(
            num_features, rnn_hidden_dim, num_layers=num_layers, bidirectional=False, dropout=dropout, batch_first=True
        )

        # Initialize hidden state
        self.hidden = None
        self._init_weights()

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.rnn_hidden_dim).to(
            self.device
        ), torch.zeros(self.num_layers * self.num_directions, batch_size, self.rnn_hidden_dim).to(
            self.device
        ))

    def forward(self, packed_inputs, initial_hidden=None):

        # Initialize hidden state based on the current batch size
        batch_size = packed_inputs.batch_sizes[0]

        if initial_hidden is None:
            initial_hidden = self.init_hidden(batch_size)

        # Forward pass through RNN
        _, new_hidden = self.rnn(packed_inputs, initial_hidden)

        # Flatten the hidden state
        last_embedding_layer = self._flatten_hidden(new_hidden)

        return last_embedding_layer, new_hidden  # Return the new hidden state

    @staticmethod
    def _flatten_hidden(h):

        if h is None:
            return None

        return h[0][-1]  # Take the last hidden state from the last layer

    @staticmethod
    def _flatten(h, batch_size):
        # (num_layers*num_directions, batch_size, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers*hidden_dim)
        return h.transpose(0, 1).contiguous().view(batch_size, -1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Decoder(nn.Module):

    def __init__(
            self, criterion=nn.MSELoss(), device="cpu", initialize_repeat=True, rnn_hidden_dim=32, num_layers=2,
            num_features=1, num_directions=1, dropout=0,

    ):
        """ Initializes the Decoder object.
            Parameters:
                criterion (nn.Module): The loss function to use, default is nn.MSELoss().
                device (str): The device to use for computations, default is “cpu”.
                initialize_repeat (bool): Whether to initialize the repeat, default is True.
                rnn_hidden_dim (int): The number of hidden units in the RNN, default is 32.
                num_layers (int): The number of layers in the RNN, default is 2.
                num_features (int): The number of input features, default is 1.
                num_directions (int): The number of directions in the RNN, default is 1.
                dropout (float): The dropout rate, default is 0.

        """
        super(Decoder, self).__init__()
        self.device = device
        self.criterion = criterion

        self.initialize_repeat = initialize_repeat
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_directions = num_directions
        self.dropout = dropout

        if not self.initialize_repeat:
            self.transformation_layer = nn.Linear(
                self.rnn_hidden_dim, self.rnn_hidden_dim * self.num_layers
            )

        # RNN layer
        self.num_hidden_states = 2
        rnn = nn.LSTM

        self.rnn = rnn(
            self.num_features, self.rnn_hidden_dim * self.num_directions, num_layers=self.num_layers,
            dropout=self.dropout, batch_first=True
        )

        self.out = nn.Linear(self.rnn_hidden_dim * self.num_directions, self.num_features)

        self._init_weights()

    def forward(self, sequence, z, lengths, return_outputs=False):
        # Unpack the sequence
        padded_sequence, _ = pad_packed_sequence(sequence, batch_first=True)

        # Now padded_sequence is a tensor, and you can get its shape
        batch_size, num_steps = padded_sequence.shape[0], padded_sequence.shape[1]

        # Initialize with the embedding
        hidden = (z.repeat(self.num_layers, 1, 1), z.repeat(self.num_layers, 1, 1))

        # Initialize recovered_sequence with zeros
        recovered_sequence = torch.zeros(padded_sequence.shape, dtype=torch.float32).to(self.device)

        # Initialize prediction with zeros
        prediction = torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)

        # Loop through each time step
        for i in range(num_steps):
            prediction, hidden = self._step(prediction, hidden)
            recovered_sequence[:, i] = prediction.squeeze() if batch_size == 1 else prediction

        # Compute loss
        loss = self.criterion(padded_sequence, recovered_sequence)

        if return_outputs:
            return loss, recovered_sequence
        else:
            return loss

    def _step(self, input_, hidden):

        # Ensure the input is 3D: [batch_size, 1, input_dim]
        if len(input_.shape) == 2:
            input_ = input_.unsqueeze(1)

        # Push input through RNN layer with current hidden state
        prediction, hidden = self.rnn(input_, hidden)

        # print("hidden.shape: ", hidden[0].shape, hidden[1].shape)

        prediction = self.out(prediction)[:, :, 0]  # .squeeze(0)

        return prediction, hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
