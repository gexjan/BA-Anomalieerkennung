import os
import torch
from model.lstm import LSTM

# Speichern des Modells lstm_model.pth im Ordner model/
def save_model(model, input_size, hidden_size, num_layers, num_classes, model_dir, model_file, logger):
    logger.info("Saving model")
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes
    }
    torch.save(model_info, os.path.join(model_dir, model_file))

# Laden des Modells lstm_model.pth aus dem Ordner model/
def load_model(model_dir, device, model_file, logger):
    logger.info("Loading model")
    model_info = torch.load(os.path.join(model_dir, model_file), map_location=device)
    model = LSTM(
        model_info['input_size'],
        model_info['hidden_size'],
        model_info['num_layers'],
        model_info['num_classes']
    )
    model.load_state_dict(model_info['model_state_dict'])
    model.to(device)
    model.eval()
    return model