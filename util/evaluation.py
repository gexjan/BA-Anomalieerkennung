import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F

class EvaluationSequenceDataset(Dataset):
    def __init__(self, x, y):
        # Sicherstellen, dass 'x' und 'y' als DataFrames vorliegen
        assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame), "x und y müssen Pandas DataFrames sein"
        y = y.drop(columns=['SeqID'])

        # Zusammenführen der DataFrames 'x' und 'y' basierend auf einem gemeinsamen Index oder Schlüssel
        # Hier nehmen wir an, dass 'x' und 'y' die gleiche Länge haben und in der gleichen Reihenfolge sind
        # Wenn es einen gemeinsamen Schlüssel gibt, verwenden Sie stattdessen pd.merge()
        combined = pd.concat([x, y], axis=1)

        # Konvertierung der 'label' Spalte zu einem Booleschen Wert, der True ist, wenn das Label 'Anomaly' ist
        combined['label'] = combined['label'] == 'Anomaly'
        combined.to_csv('combined.csv')

        # Erstellen der 'data'-Liste durch Umwandlung des DataFrame in eine Liste von Tupeln
        self.data = list(combined[['SeqID', 'window', 'next', 'label']].itertuples(index=False, name=None))
        print(self.data[:20])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, window, next_value, label = self.data[idx]
        return index, torch.tensor(window, dtype=torch.float), next_value, label
class Evaluator:
    def __init__(self, args, x, y, device, kwargs, logger, data_percentage, grouping=None):
        self.args = args
        self.x = x[:int(len(x) * data_percentage)]
        self.y = y[:int(len(y) * data_percentage)]
        self.grouping = grouping
        self.logger = logger
        self.device = device
        self.kwargs = kwargs



    def get_eval_df(self, model, use_tqdm, candidates, num_classes):
        dataset = EvaluationSequenceDataset(self.x, self.y)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)
        model.eval()
        results = []
        with torch.no_grad():
            loop = tqdm(dataloader, desc="Predicting", leave=False) if use_tqdm else dataloader
            for batch in loop:
                index, windows, next_values, labels = batch
                _, window_size = windows.shape
                # seq = torch.tensor(windows, dtype=torch.long).view(-1, window_size).to(self.device)
                seq = windows.clone().detach().to(dtype=torch.long, device=self.device).view(-1, window_size)
                seq_one_hot = F.one_hot(seq, num_classes=num_classes).float()
                outputs = model(seq_one_hot)
                probabilities = torch.softmax(outputs, dim=1)
                top_vals, top_indices = torch.topk(probabilities, candidates)
                # print(f"TOp Indices: {top_indices}, top_vals: {top_vals}")

                for batch_idx, (index, window, next_value, label) in enumerate(
                        zip(index, windows, next_values, labels)):
                    predicted = top_indices[batch_idx].tolist()
                    results.append({
                        'Index': index.item(),
                        'Sequence': window.tolist(),
                        'Next': next_value.item(),
                        'TopVals': top_vals[batch_idx].tolist(),
                        'TopIndices': predicted,
                        'Label': label.item()
                    })

        return pd.DataFrame(results)

    def evaluate(self, model, candidates, num_classes, use_tqdm=True):
        prediction_df = self.get_eval_df(model, use_tqdm, candidates, num_classes)
        prediction_df.to_csv('predictions.csv', index=False)  # Speichern ohne Index
        self.logger.info("Evaluating")

        results = []

        if self.grouping == 'session':
            grouped = list(prediction_df.groupby('Index'))  # Konvertiere in eine Liste für tqdm
            loop = tqdm(grouped, total=len(grouped), desc="Evaluating", leave=False) if use_tqdm else grouped
            for index, group_df in loop:
                # index, group_df = group
                label = group_df['Label'].iloc[0]  # Angenommen, das Label ist für den ganzen Index gleich

                # Angenommen, 'predicted_values' ist eine Spalte, die Listen oder Sets enthält,
                # und 'Next' ist eine Spalte mit den zu überprüfenden Werten.

                # Prüfen, ob 'Next' in der Liste/Set von 'predicted_values' für jede Zeile enthalten ist.
                # Dies erzeugt eine Serie von Booleschen Werten.
                anomalies_detected = group_df.apply(lambda row: row['Next'] not in row['Next-Predicted'], axis=1)

                # Überprüfen, ob mindestens eine Anomalie erkannt wurde
                anomaly_detected = anomalies_detected.any()

                if anomaly_detected:
                    if label:  # True Positive
                        results.append((1, 0, 0, 0))  # TP, TN, FP, FN
                    else:  # False Positive
                        results.append((0, 0, 1, 0))
                else:
                    if label:  # False Negative
                        results.append((0, 0, 0, 1))
                    else:  # True Negative
                        results.append((0, 1, 0, 0))

        elif self.grouping == 'time':
            loop = tqdm(prediction_df.iterrows(), total=prediction_df.shape[0], desc="Evaluating",
                        leave=False) if use_tqdm else prediction_df.iterrows()
            for _, row in loop:
                predicted = row['Next-Predicted']
                label = row['Label']
                anomaly_detected = row['Next'] not in predicted

                if anomaly_detected:
                    if label:  # True Positive
                        results.append((1, 0, 0, 0))  # TP, TN, FP, FN
                    else:  # False Positive
                        results.append((0, 0, 1, 0))
                else:
                    if label:  # False Negative
                        results.append((0, 0, 0, 1))
                    else:  # True Negative
                        results.append((0, 1, 0, 0))

        # Summiere die Ergebnisse
        self.TP, self.TN, self.FP, self.FN = map(sum, zip(*results))
        return self.calculate_f1(self.TP, self.TN, self.FP, self.FN)


    def calculate_f1(self, TP, TN, FP, FN):
        self.precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        self.recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        self.f1= (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        return self.f1

    def print_summary(self):
        self.logger.info(
            f"Evaluation results - TP: {self.TP}, TN: {self.TN}, FP: {self.FP}, FN: {self.FN}, Precision: {self.precision}, recall: {self.recall}, f1: {self.f1}")
