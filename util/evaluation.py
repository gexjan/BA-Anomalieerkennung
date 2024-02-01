import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed



class EvaluationSequenceDataset(Dataset):
    def __init__(self, x, y):
        self.data = []
        for idx in range(len(x)):
            window = x.iloc[idx]['window']
            next_value = y.iloc[idx]['next']
            label = x.iloc[idx]['label'] == 'Anomaly'
            index = x.iloc[idx]['SeqID']
            self.data.append((index, window, next_value, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, window, next_value, label = self.data[idx]
        return index, torch.tensor(window, dtype=torch.float), next_value, label

def get_eval_df(x, y, model, device, candidates, input_size, logger):
    logger.info("Predicting values")
    dataset = EvaluationSequenceDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            index, windows, next_values, labels = batch
            _, window_size = windows.shape
            windows = windows.to(device).view(-1, window_size, input_size)
            outputs = model(windows)
            probabilities = torch.softmax(outputs, dim=1)
            top_vals, top_indices = torch.topk(probabilities, candidates)

            for batch_idx, (index, window, next_value, label) in enumerate(zip(index, windows, next_values, labels)):
                predicted = top_indices[batch_idx].tolist()
                results.append({
                    'Index': index.item(),
                    'Next': next_value.item(),
                    'Next-Predicted': predicted,
                    'Label': label.item()
                })

    return pd.DataFrame(results)


def evaluate_group(group):
    index, group_df = group
    label = group_df['Label'].iloc[0]  # Angenommen, das Label ist für den ganzen Index gleich
    anomaly_detected = False

    for _, row in group_df.iterrows():
        next_value = row['Next']
        predicted_values = row['Next-Predicted']
        if next_value not in predicted_values:
            anomaly_detected = True
            break

    if anomaly_detected:
        if label:  # True Positive
            return (1, 0, 0, 0)  # TP, TN, FP, FN
        else:  # False Positive
            return (0, 0, 1, 0)
    else:
        if label:  # False Negative
            return (0, 0, 0, 1)
        else:  # True Negative
            return (0, 1, 0, 0)

def evaluate(x, y, model, device, candidates, input_size, logger):
    prediction_df = get_eval_df(x, y, model, device, candidates, input_size, logger)
    logger.info("Evaluating")
    grouped = list(prediction_df.groupby('Index'))  # Konvertiere in eine Liste für tqdm
    # print(prediction_df)

    with ProcessPoolExecutor() as executor:
        # Erstelle ein Future-Objekt für jede Gruppe
        futures = {executor.submit(evaluate_group, group): group for group in grouped}

        results = []
        for future in tqdm(as_completed(futures), total=len(grouped), desc="Evaluating", leave=False):
            result = future.result()
            results.append(result)

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
