import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import timeit
import random
import datetime

from sklearn.metrics import mean_squared_error


class RnnType:
    GRU = 1
    LSTM = 2


class ActivationFunction:
    RELU = 1
    TANH = 2
    SIGMOID = 3


class Token:
    PAD = 0
    UKN = 1
    SOS = 2
    EOS = 3


class Parameters:

    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec(f"self.{k}={v}")

class TimeSeries():

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, batch_idx):
        return self.data[batch_idx]

class TimeSeriesRnnAE:

    def __init__(self, device, params, criterion):
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

        self.device = device
        self.params = params
        self.criterion = criterion

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

    def train_epoch(self, epoch, X_iter, verbatim=False):
        """
        Train one epoch of the TimeSeriesRnnAE model.

        Parameters:
        - epoch (int): The current epoch number.
        - X_iter (DataLoader): DataLoader object for the training data.
        - verbatim (bool): Whether to print detailed logs.

        Returns:
        - epoch_loss (float): The total loss for this epoch.
        """
        start = timeit.default_timer()
        epoch_loss = 0.0
        num_batches = len(X_iter)  # Assuming X_iter is a DataLoader

        for idx, inputs in enumerate(X_iter):
            # Get size of batch (can differ between batches due to bucketing)
            batch_size = inputs.shape[0]

            # Convert to tensors and move to device
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
            inputs = inputs.float().to(self.device)

            # Train batch and get batch loss
            batch_loss = self.train_batch(inputs)

            # Update epoch loss
            epoch_loss += batch_loss

            if verbatim:
                print('[{}] Epoch: {} #batches {}/{}, loss: {:.8f}, learning rates: {:.6f}/{:.6f}'.format(
                    datetime.timedelta(seconds=int(timeit.default_timer() - start)), epoch + 1, idx + 1, num_batches,
                    (batch_loss / ((idx + 1) * batch_size)), self.encoder_lr, self.decoder_lr), end='\r')

        if verbatim:
            print()

        return epoch_loss

    def train_batch(self, inputs):
        """
        Train a single batch for the TimeSeriesRnnAE model.

        Parameters:
        - inputs (Tensor): The input time-series data for this batch.

        Returns:
        - loss (float): The normalized loss for this batch.
        """
        # Get batch size and number of time steps
        batch_size, num_steps = inputs.shape

        # Initialize hidden state for the encoder
        self.encoder.hidden = self.encoder.init_hidden(batch_size)

        # Zero the gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Forward pass through the encoder and decoder
        z, new_hidden = self.encoder(inputs)

        loss = self.decoder(inputs, z)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.clip)

        # Update parameters
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / num_steps  # Normalized loss

    def evaluate(self, input, max_steps=100):
        """
        Evaluate the TimeSeriesRnnAE model on a given input.

        Parameters:
        - input (Tensor): The input time-series data.
        - max_steps (int): Maximum number of steps for the decoded sequence.
        - use_mean (bool): Whether to use the mean of the latent space for decoding.

        Returns:
        - decoded_sequence (Tensor): The decoded time-series sequence.
        - z (Tensor): The latent representation.
        - mse (float): Mean Squared Error between the input and decoded sequence.
        """
        batch_size, _ = input.shape

        # Initialize hidden state for the encoder
        self.encoder.hidden = self.encoder.init_hidden(batch_size)

        # Forward pass through the encoder to get latent representation
        z, hidden_state= self.encoder(input)

        # Forward pass through the decoder to get the decoded sequence
        decoded_sequence = self.decoder.generate(z, max_steps=max_steps)

        # Compute Mean Squared Error (MSE) between input and decoded sequence
        # mse = mean_squared_error(input.cpu().detach().numpy(), decoded_sequence.cpu().detach().numpy())

        return decoded_sequence, z, None #, mse

    def save_models(self, encoder_file_name, decoder_file_name):
        torch.save(self.encoder.state_dict(), encoder_file_name)
        torch.save(self.decoder.state_dict(), decoder_file_name)

    def load_models(self, encoder_file_name, decoder_file_name):
        self.encoder.load_state_dict(torch.load(encoder_file_name))
        self.decoder.load_state_dict(torch.load(decoder_file_name))


class Encoder(nn.Module):

    def __init__(self, device, params):
        super(Encoder, self).__init__()
        self.device = device
        self.params = params
        # Check if valid value for RNN type
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception("Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType])))

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

        self.rnn = rnn(self.params.num_features,
                       self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bidirectional=self.params.bidirectional_encoder,
                       dropout=self.params.dropout,
                       batch_first=True)

        # Initialize hidden state
        self.hidden = None
        # Define linear layers
        self.linear_dims = params.linear_dims
        self.linear_dims = [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states] + self.linear_dims

        self._init_weights()

    def init_hidden(self, batch_size):
        if self.params.rnn_type == RnnType.GRU:
            return torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif self.params.rnn_type == RnnType.LSTM:
            return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))

    def forward(self, inputs, init_hidden_=None):
        batch_size, _ = inputs.shape

        # Reshape the inputs to [batch_size, sequence_length, 1]
        inputs = inputs.unsqueeze(-1)

        # Initialize hidden state if it's None
        if self.hidden is None:
            self.hidden = self.init_hidden(batch_size)
        elif init_hidden_ is not None:
            self.hidden = init_hidden_

        # Forward pass
        _, new_hidden = self.rnn(inputs, self.hidden)
        X = self._flatten_hidden(new_hidden, batch_size)
        return X, new_hidden  # Return the new hidden state

    def _flatten_hidden(self, h, batch_size):
        # if h is None:
        #     return None
        # elif isinstance(h, tuple): # LSTM
        #     X = torch.cat([self._flatten(h[0], batch_size), self._flatten(h[1], batch_size)], 1)
        # else: # GRU
        #     X = self._flatten(h, batch_size)
        # return X

        if h is None:
            return None
        elif isinstance(h, tuple):  # LSTM
            h_last = h[0][-1]  # Take the last hidden state from the last layer
            c_last = h[1][-1]  # Take the last cell state from the last layer
            X = torch.cat([h_last, c_last], dim=1)  # Concatenate along feature dimension
        else:  # GRU
            h_last = h[-1]  # Take the last hidden state from the last layer
            X = h_last
        # return X
        return h_last

    def _flatten(self, h, batch_size):
        # (num_layers*num_directions, batch_size, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers*hidden_dim)
        return h.transpose(0,1).contiguous().view(batch_size, -1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def _sample(self, mean, logv):
        std = torch.exp(0.5 * logv)
        # torch.randn_like() creates a tensor with values samples from N(0,1) and std.shape
        eps = torch.randn_like(std)
        # Sampling from Z~N(μ, σ^2) = Sampling from μ + σX, X~N(0,1)
        z = mean + std * eps
        return z


class Decoder(nn.Module):

    def __init__(self, device, params, criterion):
        super(Decoder, self).__init__()
        self.device = device
        self.params = params
        self.criterion = criterion
        # Check if a valid parameter for RNN type is given
        if self.params.rnn_type not in [RnnType.GRU, RnnType.LSTM]:
            raise Exception(
                "Unknown RNN type for encoder. Valid options: {}".format(', '.join([str(t) for t in RnnType])))

        if not self.params.initialize_repeat:
            self.transformation_layer = nn.Linear(self.params.rnn_hidden_dim, self.params.rnn_hidden_dim * self.params.num_layers)

        # RNN layer
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        print("num_directions: ", self.num_directions)

        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(self.params.num_features,
                       self.params.rnn_hidden_dim*self.num_directions,
                       num_layers=self.params.num_layers,
                       dropout=self.params.dropout,
                       batch_first=True)
        self.linear_dims = self.params.linear_dims + [self.params.rnn_hidden_dim * self.num_directions * self.params.num_layers * self.num_hidden_states]

        print("rnn_hidden_dim: ", self.params.rnn_hidden_dim * self.num_directions)
        self.out = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.num_features)
        self._init_weights()

    def forward(self, inputs, z, return_outputs=False):
        batch_size, num_steps = inputs.shape

        # Unflatten hidden state for GRU or LSTM
        hidden = self._unflatten_hidden(z, batch_size)
        # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
        hidden = self._init_hidden_state(hidden)
        # Create SOS token tensor as first input for decoder # THIS IS PROBABLY NOT WHAT WE WANT
        # input = torch.LongTensor([[Token.SOS]] * batch_size).to(self.device)
        input = torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)  # Example using zeros

        # Decide whether to do teacher forcing or not
        use_teacher_forcing = random.random() < self.params.teacher_forcing_prob
        # Initiliaze loss
        loss = 0
        outputs = torch.zeros((batch_size, num_steps), dtype=torch.float32).to(self.device)
        # if use_teacher_forcing:
        #     for i in range(num_steps):
        #         output, hidden = self._step(input, hidden)
        #         topv, topi = output.topk(1)
        #         outputs[:,i] = topi.detach().squeeze()
        #         #print(output[0], inputs[:, i][0])
        #         loss += self.criterion(output, inputs[:, i])
        #         input = inputs[:, i].unsqueeze(dim=1)
        # else:

        for i in range(num_steps):

            output, hidden = self._step(input, hidden)

            max_value, max_idx = output.topk(1) # get max_value, max_idx
            # input = max_idx.detach()
            # outputs[:, i] = max_idx.detach().squeeze()
            input = max_value.detach()
            outputs[:, i] = max_value.detach().squeeze()

            loss += self.criterion(max_value, inputs[:, i])
            print(max_value, inputs[:, i], loss)
        #     print(output, inputs[:, i], loss)
        # print("")

        # Return loss
        if return_outputs == True:
            return loss, outputs
        else:
            return loss

    def _step(self, input, hidden):

        # Ensure the input is 3D: [batch_size, 1, input_dim]
        if len(input.shape) == 2:
            input = input.unsqueeze(1)

        # Type casting, if necessary
        input = input.to(torch.float32)
        hidden = (hidden[0].to(torch.float32), hidden[1].to(torch.float32))

        # Push input through RNN layer with current hidden state
        output, hidden = self.rnn(input, hidden)

        # Remove the sequence dimension
        output = output.squeeze(1)

        # Push output through linear layer to get to num_features
        output = self.out(output)

        return output, hidden

    def _unflatten_hidden(self, X, batch_size):
        if self.params.rnn_type == RnnType.LSTM:  # LSTM

            if self.params.initialize_repeat:
                # Repeat the last hidden state for each layer
                # h = (self._unflatten(X, batch_size).repeat(self.params.num_layers, 1, 1),
                #      self._unflatten(X, batch_size).repeat(self.params.num_layers, 1, 1))
                h = (X.repeat(self.params.num_layers, 1, 1),
                     X.repeat(self.params.num_layers, 1, 1))
            else:
                # Learn a transformation (assuming you have a transformation_layer)
                transformed = self.transformation_layer(X)
                h = (self._unflatten(transformed, batch_size),
                     self._unflatten(transformed, batch_size))
        else:  # GRU
            if self.params.initialize_repeat:
                h = self._unflatten(X, batch_size).repeat(self.params.num_layers, 1, 1)
            else:
                transformed = self.transformation_layer(X)
                h = self._unflatten(transformed, batch_size)
        return h

    def _unflatten(self, X, batch_size):
        return X.view(self.params.num_layers, batch_size, self.params.rnn_hidden_dim).contiguous()

    def _init_hidden_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        elif isinstance(encoder_hidden, tuple): # LSTM
            return tuple([self._concat_directions(h) for h in encoder_hidden])
        else: # GRU
            return self._concat_directions(encoder_hidden)

    def _concat_directions(self, hidden):
            # hidden.shape = (num_layers * num_directions, batch_size, hidden_dim)
            #print(hidden.shape, hidden[0:hidden.size(0):2].shape)
            if self.params.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
                # Alternative approach (same output but easier to understand)
                #h = hidden.view(self.params.num_layers, self.num_directions, hidden.size(1), self.params.rnn_hidden_dim)
                #h_fwd = h[:, 0, :, :]
                #h_bwd = h[:, 1, :, :]
                #hidden = torch.cat([h_fwd, h_bwd], 2)
            return hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def generate(self, z, max_steps):
            decoded_sequence = []
            # "Expand" z vector
            #X = self.z_to_hidden(z)
            X = z
            # Unflatten hidden state for GRU or LSTM
            hidden = self._unflatten_hidden(X, 1)
            # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
            hidden = self._init_hidden_state(hidden)
            # Create SOS token tensor as first input for decoder
            input = torch.LongTensor([[Token.SOS]]).to(self.device)
            # Generate words step by step
            for i in range(max_steps):
                output, hidden = self._step(input, hidden)
                topv, topi = output.data.topk(1)
                #print(topi.shape, topi[0])
                if topi.item() == Token.EOS:
                    break
                else:
                    decoded_sequence.append(topi.item())
                    input = topi.detach()
            # Return final decoded sequence (sequence of of indices)
            return decoded_sequence


def plot_traces(timeseries_rnn_ae, X, batch_size=1, n_samples=16):

    from tqdm import tqdm
    from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
    import torch.nn as nn
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    timeseries_rnn_ae.eval()

    X_val = TimeSeries(X)
    sampler_val = BatchSampler(SequentialSampler(X_val), batch_size=batch_size, drop_last=False)
    X_val_iter = DataLoader(dataset=X_val, batch_sampler=sampler_val, num_workers=8)

    x_val = []
    y_val = []
    latent = []
    loss = []
    for iter in tqdm(X_val_iter):

        _, z, _ = timeseries_rnn_ae.evaluate(iter, max_steps=iter.shape[1])

        if len(z[0]) < 10:
            print(z)
        else:
            print(np.mean(z.detach().numpy()), z.shape)

        decoder = timeseries_rnn_ae.decoder
        l, decoded = decoder(iter, z, return_outputs=True)

        x_val.append(iter)
        y_val.append(decoded)
        latent.append(z)
        loss.append(l)

    n_samples = min(len(x_val), n_samples)

    fig, axx = plt.subplots(n_samples, 1, figsize=(10, n_samples*2), sharey=True)

    for i in range(n_samples):

        x = np.squeeze(x_val[i].numpy())
        y = np.squeeze(y_val[i].numpy())


        axx[i].plot(x, color="gray", linestyle="--")
        axx[i].plot(y, color="red", linestyle="-")
        axx[i].set_title(f"loss: {loss[i]:.2f}")

    plt.tight_layout()

