import torch
from model.lstm import LSTM
from util.modelmanager import load_model
import random
import numpy as np

# Seed-Wert festlegen
seed_value = 42

# PyTorch Seed setzen
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # für multi-GPU

# NumPy Seed setzen
np.random.seed(seed_value)

# Python Random Seed setzen
random.seed(seed_value)

# Zusätzliche Konfigurationen für PyTorch, um weitere Zufälligkeiten zu minimieren
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class Predictor:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(args.model_dir, self.device)
        self.args = args

    def predict_next(self, window):
        window_tensor = torch.tensor([window], dtype=torch.float).to(self.device)
        window_tensor = window_tensor.view(-1, len(window), self.args.input_size)  # Reshape to match model input format

        self.model.eval()
        with torch.no_grad():
            output = self.model(window_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, top_idxs = torch.topk(probabilities, self.args.candidates)

            # Returning the top 3 predictions as a simple vector (list)
            candidate_predictions = top_idxs[0].tolist()

        return candidate_predictions
