import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

class EvaluationSequenceDataset(Dataset):
    def __init__(self, x, y):
        # self.data = []
        # for idx in range(len(x)):
        #     window = x.iloc[idx]['window']
        #     next_value = y.iloc[idx]['next']
        #     label = x.iloc[idx]['label'] == 'Anomaly'
        #     index = x.iloc[idx]['SeqID']
        #     self.data.append((index, window, next_value, label))

        # Sicherstellen, dass 'x' und 'y' als DataFrames vorliegen
        assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame), "x und y müssen Pandas DataFrames sein"
        # Entfernen der 'SeqID' Spalte aus 'y', um Duplikate zu vermeiden
        y = y.drop(columns=['SeqID'])

        # Zusammenführen der DataFrames 'x' und 'y' basierend auf einem gemeinsamen Index oder Schlüssel
        # Hier nehmen wir an, dass 'x' und 'y' die gleiche Länge haben und in der gleichen Reihenfolge sind
        # Wenn es einen gemeinsamen Schlüssel gibt, verwenden Sie stattdessen pd.merge()
        combined = pd.concat([x, y], axis=1)

        # Konvertierung der 'label' Spalte zu einem Booleschen Wert, der True ist, wenn das Label 'Anomaly' ist
        combined['label'] = combined['label'] == 'Anomaly'

        # Erstellen der 'data'-Liste durch Umwandlung des DataFrame in eine Liste von Tupeln
        self.data = list(combined[['SeqID', 'window', 'next', 'label']].itertuples(index=False, name=None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, window, next_value, label = self.data[idx]
        return index, torch.tensor(window, dtype=torch.float), next_value, label
class Evaluator:
    def __init__(self, args, x, y, device, kwargs, logger):
        self.args = args
        self.x, self.y = x, y
        self.logger = logger
        self.device = device
        self.kwargs = kwargs


    def get_eval_df(self, model):
        self.logger.info("Create EvaluationSequenceDataset")
        dataset = EvaluationSequenceDataset(self.x, self.y)
        self.logger.info("Creating dataloader")
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True)

        self.logger.info("Predicting values")
        model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                index, windows, next_values, labels = batch
                _, window_size = windows.shape
                windows = windows.to(self.device).view(-1, window_size, self.args.input_size)
                outputs = model(windows)
                probabilities = torch.softmax(outputs, dim=1)
                top_vals, top_indices = torch.topk(probabilities, self.args.candidates)

                for batch_idx, (index, window, next_value, label) in enumerate(
                        zip(index, windows, next_values, labels)):
                    predicted = top_indices[batch_idx].tolist()
                    results.append({
                        'Index': index.item(),
                        'Next': next_value.item(),
                        'Next-Predicted': predicted,
                        'Label': label.item()
                    })

        return pd.DataFrame(results)

    def evaluate(self, model):
        prediction_df = self.get_eval_df(model)
        self.logger.info("Evaluating")
        grouped = list(prediction_df.groupby('Index'))  # Konvertiere in eine Liste für tqdm
        self.logger.info("Threading")

        results = []
        for index, group_df in tqdm(grouped, total=len(grouped), desc="Evaluating", leave=False):
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

        self.logger.info("Summieren")

        # Summiere die Ergebnisse
        TP, TN, FP, FN = map(sum, zip(*results))
        return TP, TN, FP, FN


def calculate_f1(TP, TN, FP, FN, logger):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    logger.info(
        f"Evaluation results - TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}")
    return f1
