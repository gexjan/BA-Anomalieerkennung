import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(num_classes, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        # h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        # c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        # out, _ = self.lstm(input, (h0, c0))
        out, _ = self.lstm(input)
        print("Hidden: ", out)
        out = self.fc(out[:, -1, :])
        return out


