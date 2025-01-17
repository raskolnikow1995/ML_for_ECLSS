import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Takes 1 time series value in this example, to hidden size
        self.rnn = nn.RNN(1, hidden_size)
        
    def forward(self, encoder_inputs):
        # NOTE: encoder_inputs looks like: 
        # [..., X_t-4, X_t-3, X_t-2, X_t-1, X_t]
        
        # We let Pytorch handle the rollout behind-the-scenes, 
        # so just feed in the whole encoder sequence.
        # And all we need is the final hidden vector as Z
        outputs, hidden = self.rnn(encoder_inputs)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Also takes 1 time series value
        self.rnn = nn.RNN(1, hidden_size)
        # The output layer transforms the latent representation 
        # back to a single prediction
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, initial_input, encoder_outputs, hidden, targets, 
                teacher_force_probability):
        # NOTE:
        # initial_input is X_t
        # hidden is Z
        # targets looks like: [X_t+1, X_t+2, X_t+3, ...]
        # encoder_outputs are not used, but will be for attention later
        
        decoder_sequence_length = len(targets)

        # Store decoder outputs
        outputs = [None for _ in range(decoder_sequence_length)]
        
        input_at_t = initial_input
        
        # Here we have to roll out the decoder sequence ourselves because of 
        # sometimes teacher forcing
        for t in range(decoder_sequence_length):            
            output, hidden = self.rnn(input_at_t, hidden)
            outputs[t] = self.out(output)
            
            # Set-up input for next timestep
            teacher_force = random() < teacher_force_probability
            # The next timestep's input will either be this timestep's 
            # target or output
            input_at_t = targets[t] if teacher_force else outputs[t]

        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        # The best loss function to use depends on the problem.
        # We will see a different loss function later for probabilistic
        # forecasting
        self.loss_function = nn.L1Loss()
    
    def forward(self, encoder_inputs, targets, teacher_force_probability):
        encoder_outputs, hidden = self.encoder(encoder_inputs)
        outputs = self.decoder(encoder_inputs[-1], encoder_outputs,
                               hidden, targets, teacher_force_probability)
        return outputs

    def compute_loss(self, outputs, targets):
        loss = self.loss_function(outputs, targets)
        return loss
    
    def optimize(self, outputs, targets):
        self.optimizer.zero_grad()
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
