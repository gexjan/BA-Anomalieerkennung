import pandas as pd
from util.predict import Predictor
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from model.lstm import LSTM
from util.modelmanager import load_model

class Validator:
    def __init__(self, args, logger):
        self.predictor = Predictor(args)
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
                    window_count += 1
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


                #     # Bestimmen Sie, ob die Vorhersage korrekt ist
                #     predicted_label = 0 if next_value in candidate_predictions else 1
                #     print("Predicted_label: ", predicted_label)
                    

                #     if predicted_label == 1:
                #         anomaly_detected = True
                #         if actual_label == 1:
                #             TP += 1
                #         else:
                #             FP += 1
                #         break  # Brechen Sie ab, wenn eine Anomalie gefunden wurde

                # if not anomaly_detected:
                #     if actual_label == 0:
                #         TN += 1
                #     else:
                #         FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0

        print("WIndow-Counts: ", window_counts)
        return f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}"

    # def validate(self, data):
    #     TP, TN, FP, FN = 0,0,0,0

    #     self.logger.info("Validating Log-Entries")

    #     with torch.no_grad():
    #         for index, row in data.iterrows():
    #             # print("Index:", index)
    #             # print("Row: ", row)

    #             sequence = row['EventSequence']
    #             for i in range(len(sequence) - self.args.window_size):
    #                 window = sequence[i : i + self.args.window_size]
    #                 next_value = sequence[i + self.args.window_size]

    #                 # Bereiten Sie das Fenster für die Vorhersage vor
    #                 window_tensor = torch.tensor([window], dtype=torch.float).to(self.device)
    #                 window_tensor = window_tensor.view(-1, len(window), self.args.input_size)

    #                 # Modellvorhersage
    #                 output = self.model(window_tensor)
    #                 probabilities = torch.softmax(output, dim=1)
    #                 _, top_idxs = torch.topk(probabilities, self.args.candidates)
    #                 candidate_predictions = top_idxs[0].tolist()

    #                 # Bestimmen Sie, ob die Vorhersage korrekt ist
    #                 predicted_label = 0 if next_value in candidate_predictions else 1
    #                 actual_label = 1 if row['Label'] == 'Anomaly' else 0
    #                 # print("Actual-Label ", actual_label)

    #                 # Aktualisieren der TP, TN, FP, FN Werte
    #                 if predicted_label == 1 and actual_label == 1:
    #                     TP += 1
    #                 elif predicted_label == 0 and actual_label == 0:
    #                     TN += 1
    #                 elif predicted_label == 1 and actual_label == 0:
    #                     FP += 1
    #                 elif predicted_label == 0 and actual_label == 1:
    #                     FN += 1

    #                 # Abbrechen, wenn eine Anomalie gefunden wurde
    #                 if predicted_label == 1:
    #                     break

    #         precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    #         recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    #         # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    #         f1 = (2 * TP) / (2*TP + FP + FN)

    #         return f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}"













        # for index, row in data.iterrows():

        #     sequence = row['EventSequence']
        #     # print("Sequence", sequence)
        #     actual_label = 1 if row['Label'] == 'Anomaly' else 0
        #     # print("Actual label: ", actual_label)

        #     for i in range(len(sequence) - self.args.window_size):
        #         window = sequence[i:i + self.args.window_size]
        #         # print(f"Window {i}: ", window)
        #         next_value = sequence[i + self.args.window_size]
        #         # print("Next-Value: ", next_value)
        #         predictions = self.predictor.predict_next(window)
        #         # print("Predictions: ", predictions)

        #         predicted_label = 0 if next_value in predictions else 1
        #         actual_labels.append(actual_label)
        #         predicted_labels.append(predicted_label)

        #         # Optional: Brechen Sie nur bei einer erkannten Anomalie ab
        #         if predicted_label == 1:
        #             break

        # precision = precision_score(actual_labels, predicted_labels)
        # recall = recall_score(actual_labels, predicted_labels)
        # f1 = f1_score(actual_labels, predicted_labels)

        # return {'Precision': precision, 'Recall': recall, 'F1': f1}


    # def validate(self, data):
    #     actual_labels = []
    #     predicted_labels = []

    #     for index, row in data.iterrows():
    #         sequence = row['EventSequence']
    #         actual_label = 1 if row['Label'] == 'Anomaly' else 0

    #         for i in range(len(sequence) - self.args.window_size):
    #             window = sequence[i:i + self.args.window_size]
    #             next_value = sequence[i + self.args.window_size]
    #             predictions = self.predictor.predict_next(window)

    #             predicted_label = 0 if next_value in predictions else 1
    #             actual_labels.append(actual_label)
    #             predicted_labels.append(predicted_label)
    #             break  # Brechen Sie nach der ersten Anomalie ab

    #     precision = precision_score(actual_labels, predicted_labels)
    #     recall = recall_score(actual_labels, predicted_labels)
    #     f1 = f1_score(actual_labels, predicted_labels)

    #     return {'Precision': precision, 'Recall': recall, 'F1': f1}

    # def validate(self, data):
    #     actual_labels = []
    #     predicted_labels = []

    #     for index, row in data.iterrows():
    #         # print("-------------------")
    #         # print("Index: ", index)
    #         # print("Row: ", row)
    #         sequence = row['EventSequence']
    #         # print("Sequence", sequence)
    #         actual_label = 1 if row['Label'] == 'Anomaly' else 0
    #         # print("Actual label: ", actual_label)

    #         for i in range(len(sequence) - self.args.window_size):
    #             window = sequence[i:i + self.args.window_size]
    #             # print(f"Window {i}: ", window)
    #             next_value = sequence[i + self.args.window_size]
    #             # print("Next-Value: ", next_value)
    #             predictions = self.predictor.predict_next(window)
    #             # print("Predictions: ", predictions)

    #             predicted_label = 0 if next_value in predictions else 1
    #             actual_labels.append(actual_label)
    #             predicted_labels.append(predicted_label)

    #             # Optional: Brechen Sie nur bei einer erkannten Anomalie ab
    #             if predicted_label == 1:
    #                 break

    #     precision = precision_score(actual_labels, predicted_labels)
    #     recall = recall_score(actual_labels, predicted_labels)
    #     f1 = f1_score(actual_labels, predicted_labels)

    #     return {'Precision': precision, 'Recall': recall, 'F1': f1}