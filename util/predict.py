import torch
from model.lstm import LSTM
from util.modelmanager import load_model

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
