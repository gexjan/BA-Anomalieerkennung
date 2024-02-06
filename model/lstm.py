import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, input):
#         h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
#         c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
#         out, _ = self.lstm(input, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_p=0.2):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # Hinzufügen von Dropout zu LSTM
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
#         self.dropout = nn.Dropout(dropout_p)  # Dropout-Schicht nach LSTM
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, input):
#         h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
#         c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
#         out, _ = self.lstm(input, (h0, c0))
#         out = self.dropout(out)  # Anwenden von Dropout
#         out = self.fc(out[:, -1, :])
#         return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_p=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Hinzufügen von Dropout zu LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_p)  # Dropout-Schicht nach LSTM
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.dropout(out)  # Anwenden von Dropout
        out = self.fc(out[:, -1, :])
        return out
