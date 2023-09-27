from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class RnnType:
    GRU = 1
    LSTM = 2


class Parameters:

    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec(f"self.{k}={v}")


class TimeSeriesRnnAE:

    def __init__(self, params, use_cuda=False):
        """
        Initialize the TimeSeriesRnnAE model.

        Parameters:
        - input_dim (int): Dimensionality of the input time-series data.
        - hidden_dim (int): Number of hidden units in the LSTM layers.
        - num_layers (int): Number of LSTM layers.
        - dropout (float): Dropout rate for LSTM layers.
        - l1_reg (float): L1 regularization term for the latent representation.

        """
        super(TimeSeriesRnnAE, self).__init__()

        # set device
        if torch.cuda.is_available() and use_cuda:
            device = torch.device("cuda:0")
        else:
            device = "cpu"

        self.device = device
        self.params = params
        self.criterion = nn.MSELoss()

        # Create encoder rnn and decoder rnn module
        self.encoder = Encoder(device, params)
        self.decoder = Decoder(device, params, self.criterion)
        self.encoder.to(device)
        self.decoder.to(device)
        # Create optimizers for encoder and decoder
        self.encoder_lr = self.params.encoder_lr
        self.decoder_lr = self.params.decoder_lr
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.decoder_lr)

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
        Train one epoch of the TimeSeriesRnnAE model.

        Parameters:
        - epoch (int): The current epoch number.
        - X_iter (DataLoader): DataLoader object for the training data.
        - verbatim (bool): Whether to print detailed logs.

        Returns:
        - epoch_loss (float): The total loss for this epoch.
        """

        self.train()

        patience_counter = 0
        best_loss = float('inf')

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
                self.update_learning_rates(0.99, 0.99)
            learning_rates.append([self.encoder_lr, self.decoder_lr])

            if show_mode == "progress":
                iterator.set_description(
                    f"loss: {epoch_loss:.4f} "
                    f"lr: ({self.encoder_lr:.5f}, {self.decoder_lr:.5f}) "
                    f"P:{patience_counter}"
                    )
                iterator.update(1)

            elif show_mode == "notebook":

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

            # Early stopping
            if dataloader_val is not None:
                losses = val_losses
            else:
                losses = train_losses

            smoothed_loss = np.sum(np.array(losses[-smooth_loss_len:])) / min(
                len(losses), smooth_loss_len
                )  # last 5 epochs
            if best_loss - smoothed_loss > min_delta:
                best_loss = smoothed_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        self.eval()

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
        num_steps = packed_inputs.data.size(0)  # Total number of timesteps across all sequences

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
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.clip)

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

    def plot_traces(self, dataloader, figsize=(10, 10), n_samples=16, sharex=False):

        self.eval()

        x_val = []
        y_val = []
        latent = []
        losses = []
        for batch_data, batch_lengths in tqdm(dataloader):
            batch_data = batch_data.unsqueeze(-1)
            batch_data = batch_data.to(dtype=torch.float32)  # .to(self.device)  # Move to device and ensure it's float
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

        n_samples = min(len(x_val), n_samples)

        fig, axx = plt.subplots(n_samples, 2, figsize=figsize, sharey=False, sharex=sharex)

        for i, idx in enumerate(np.random.randint(0, len(x_val), size=(n_samples))):
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

        return x_val, y_val, latent, losses


class Encoder(nn.Module):

    def __init__(self, device, params):
        super(Encoder, self).__init__()
        self.device = device
        self.params = params
        # Check if valid value for RNN type
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception(
                "Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType]))
                )

        # RNN layer
        # self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        self.num_directions = 1
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        else:
            raise ValueError

        self.rnn = rnn(
            self.params.num_features, self.params.rnn_hidden_dim, num_layers=self.params.num_layers,
            bidirectional=self.params.bidirectional_encoder, dropout=self.params.dropout, batch_first=True
            )

        # Initialize hidden state
        self.hidden = None
        # Define linear layers
        self.linear_dims = params.linear_dims
        self.linear_dims = [
                               self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states] + self.linear_dims

        self._init_weights()

    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(
                self.device
                )
        elif self.params.rnn_type == RnnType.LSTM:
            return (
            torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(
                self.device
                ), torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(
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
        last_embedding_layer = self._flatten_hidden(new_hidden, batch_size)

        return last_embedding_layer, new_hidden  # Return the new hidden state

    def _flatten_hidden(self, h, batch_size):

        if h is None:
            return None
        elif isinstance(h, tuple):  # LSTM
            h_last = h[0][-1]  # Take the last hidden state from the last layer
        else:  # GRU
            h_last = h[-1]  # Take the last hidden state from the last layer
        return h_last

    def _flatten(self, h, batch_size):
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

    def __init__(self, device, params, criterion):
        super(Decoder, self).__init__()
        self.device = device
        self.params = params
        self.criterion = criterion
        # Check if a valid parameter for RNN type is given
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception(
                "Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType]))
            )

        if not self.params.initialize_repeat:
            self.transformation_layer = nn.Linear(
                self.params.rnn_hidden_dim, self.params.rnn_hidden_dim * self.params.num_layers
                )

        # RNN layer
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1

        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM

        self.rnn = rnn(
            self.params.num_features, self.params.rnn_hidden_dim * self.num_directions,
            num_layers=self.params.num_layers, dropout=self.params.dropout, batch_first=True
            )

        # self.linear_dims = self.params.linear_dims + [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states]
        #
        # print("rnn_hidden_dim: ", self.params.rnn_hidden_dim * self.num_directions)
        self.out = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.num_features)

        self._init_weights()

    def forward(self, sequence, z, lengths, return_outputs=False):
        # Unpack the sequence
        padded_sequence, _ = pad_packed_sequence(sequence, batch_first=True)

        # Now padded_sequence is a tensor, and you can get its shape
        batch_size, num_steps = padded_sequence.shape[0], padded_sequence.shape[1]

        # Initialize with the embedding
        hidden = (z.repeat(self.params.num_layers, 1, 1), z.repeat(self.params.num_layers, 1, 1))

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

    def _step(self, input, hidden):

        # Ensure the input is 3D: [batch_size, 1, input_dim]
        if len(input.shape) == 2:
            input = input.unsqueeze(1)

        # Push input through RNN layer with current hidden state
        prediction, hidden = self.rnn(input, hidden)

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


class PaddedSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lengths = [len(seq) for seq in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.lengths[index]


class PaddedDataLoader():

    def __init__(self, data):
        self.data = data

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
            ds = [torch.tensor(x, dtype=torch.float32) for x in ds]
            ds = PaddedSequenceDataset(ds)

            bs = batch_size[i]
            if bs == "auto":
                bs = len(ds)

            ds = DataLoader(ds, batch_size=bs, shuffle=shuffle[i], collate_fn=self.collate_fn)
            datasets.append(ds)

        return datasets

    def collate_fn(self, batch):
        # Sort sequences by length in descending order
        batch.sort(key=lambda x: x[1], reverse=True)  # x[1] is the length

        # Separate sequence lengths and sequences
        sequences = [x[0] for x in batch]  # x[0] is the data
        lengths = [x[1] for x in batch]  # x[1] is the length

        # Pad sequences
        sequences = pad_sequence(sequences, batch_first=True)

        return sequences, lengths
