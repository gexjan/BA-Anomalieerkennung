import os
import torch
from model.lstm import LSTM

def save_model(model, input_size, hidden_size, num_layers, num_classes, model_dir):
        model_info = {
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes
        }
        torch.save(model_info, os.path.join(model_dir, 'lstm_model.pth'))

def load_model(model_dir, device):
    model_info = torch.load(os.path.join(model_dir, 'lstm_model.pth'), map_location=device)
    model = LSTM(
        model_info['input_size'],
        model_info['hidden_size'],
        model_info['num_layers'],
        model_info['num_classes']
    )
    model.load_state_dict(model_info['model_state_dict'])
    model.to(device)
    # model.eval()
    return model