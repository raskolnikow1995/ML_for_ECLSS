import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
    
class Sequence_Dataset(Dataset):
    def __init__(self, Features, Labels):
        self.Features = torch.tensor(Features, dtype=torch.float32)
        self.Labels = torch.tensor(Labels, dtype=torch.float32)

    def __len__(self):
        return len(self.Features)

    def __getitem__(self, idx):
        return self.Features[idx], self.Labels[idx]


