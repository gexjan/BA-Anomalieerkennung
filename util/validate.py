import pandas as pd
from util.predict import Predictor
from sklearn.metrics import f1_score, precision_score, recall_score

class Validator:
    def __init__(self, args):
        self.predictor = Predictor(args)
        self.args = args

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

    def validate(self, data):
        actual_labels = []
        predicted_labels = []

        for index, row in data.iterrows():
            # print("-------------------")
            # print("Index: ", index)
            # print("Row: ", row)
            sequence = row['EventSequence']
            # print("Sequence", sequence)
            actual_label = 1 if row['Label'] == 'Anomaly' else 0
            # print("Actual label: ", actual_label)

            for i in range(len(sequence) - self.args.window_size):
                window = sequence[i:i + self.args.window_size]
                # print(f"Window {i}: ", window)
                next_value = sequence[i + self.args.window_size]
                # print("Next-Value: ", next_value)
                predictions = self.predictor.predict_next(window)
                # print("Predictions: ", predictions)

                predicted_label = 0 if next_value in predictions else 1
                actual_labels.append(actual_label)
                predicted_labels.append(predicted_label)

                # Optional: Brechen Sie nur bei einer erkannten Anomalie ab
                if predicted_label == 1:
                    break

        precision = precision_score(actual_labels, predicted_labels)
        recall = recall_score(actual_labels, predicted_labels)
        f1 = f1_score(actual_labels, predicted_labels)

        return {'Precision': precision, 'Recall': recall, 'F1': f1}
