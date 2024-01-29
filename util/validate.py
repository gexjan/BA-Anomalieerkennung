import pandas as pd
# from util.predict import Predictor
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from model.lstm import LSTM
from util.modelmanager import load_model

class Validator:
    def __init__(self, args, logger):
        # self.predictor = Predictor(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.args.model_dir, self.device)
        self.logger = logger


    def validate(self, data):
        TP, TN, FP, FN = 0,0,0,0

        self.logger.info("Validating Log-Entries")
        print("Candidate: ", self.args.candidates)

        with torch.no_grad():
            for index, row in data.iterrows():
                sequence = row['EventSequence']
                anomaly_detected = False
                actual_label = True if row['Label'] == 'Anomaly' else False # True (Anomaly), False (normal)

                for i in range(len(sequence) - self.args.window_size):

                    window = sequence[i : i + self.args.window_size]
                    next_value = sequence[i + self.args.window_size]
                    # print("WIndow: ", window)

                    # Bereiten Sie das Fenster für die Vorhersage vor
                    window_tensor = torch.tensor([window], dtype=torch.float).to(self.device)
                    window_tensor = window_tensor.view(-1, len(window), self.args.input_size)

                    # Modellvorhersage
                    output = self.model(window_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    top, top_idxs = torch.topk(probabilities, self.args.candidates)
                    # print("Werte: ", top, top_idxs)
                    candidate_predictions = top_idxs[0].tolist()
                    # print("Predictions: ", candidate_predictions)
                    # print("Next-Label: ", next_value)

                    next_in_predictions = True if next_value in candidate_predictions else False
                    if not next_in_predictions:
                        anomaly_detected = True
                        if actual_label:
                            TP += 1
                        else:
                            FP += 1
                        break

                if not anomaly_detected:
                    if actual_label:
                        FN += 1
                    else:
                        TN += 1



        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0

        return f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}"

