import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# class SequenceDataset(Dataset):
#     def __init__(self, dataframe, window_size, input_size):
#         self.dataframe = dataframe
#         self.window_size = window_size
#         self.input_size = input_size
#         self.sequences = self._create_sequences()
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         sequence, label = self.sequences[idx]
#         window = sequence[:self.window_size] # Nur das Fenster, ohne Next-Value
#         next_value = sequence[-1] # Der letzte Wert ist der Next-Value
#         window_tensor = torch.tensor(window, dtype=torch.float).view(self.window_size, self.input_size)
#         return window_tensor, next_value, label
#
#
#     def _create_sequences(self):
#         sequences = []
#         for _, row in self.dataframe.iterrows():
#             sequence = row['EventSequence']
#             label = row['Label'] == 'Anomaly'
#             seqlen = len(sequence)
#             if seqlen < self.window_size + 1:
#                 sequence += [1] * (self.window_size + 1 - seqlen)
#
#             for i in range(len(sequence) - self.window_size):
#                 window = sequence[i:i + self.window_size]
#                 next_value = sequence[i + self.window_size]
#                 sequences.append((window + [next_value], label))
#
#         return sequences
#
# def evaluate(evaluation_df, model, device, candidates, window_size, input_size, logger):
#     # Rest des Codes bleibt gleich
#     # ...
#     logger.info(f"Begin evaluation with {candidates} candidates")
#     TP, TN, FP, FN = 0, 0, 0, 0
#     model.eval()
#
#     dataset = SequenceDataset(evaluation_df, window_size, input_size)
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     with torch.no_grad():
#         for window_tensor, next_value, label in tqdm(data_loader, desc="Evaluating", leave=False):
#             window_tensor = window_tensor.to(device).unsqueeze(0)  # Fügt eine zusätzliche Dimension für den Batch hinzu
#             output = model(window_tensor)
#             probabilities = torch.softmax(output, dim=1)
#             top_vals, top_indices = torch.topk(probabilities, candidates)
#
#             candidate_predictions = top_indices[0].tolist()
#             next_in_candidate_predictions = next_value in candidate_predictions
#
#             if not next_in_candidate_predictions:
#                 if label:
#                     TP += 1
#                 else:
#                     FP += 1
#             else:
#                 if label:
#                     FN += 1
#                 else:
#                     TN += 1
#
#     return TP, TN, FP, FN

def evaluate(evaluation_df, model, device, candidates, window_size, input_size, logger):
    logger.info(f"Begin evaluation with {candidates} candidates")
    TP, TN, FP, FN = 0, 0, 0, 0

    model.eval()

    with torch.no_grad():
        # for index, row in evaluation_df.iterrows():
        for index, row in enumerate(tqdm(evaluation_df.iterrows(), total=evaluation_df.shape[0], desc="Evaluating", leave=False)):

            sequence = row[1]['EventSequence']
            label = row[1]['Label'] == 'Anomaly'

            anomaly_detected = False

            seqlen = len(sequence)
            if seqlen < window_size + 1:
                # Auffüllen mit 1, der ID für #PAD
                sequence += [1] * (window_size + 1 - seqlen)

            for i in range(len(sequence) - window_size):
                window = sequence[i:i + window_size]
                next_value = sequence[i + window_size]

                # Fenster in Tensoren umwandeln und umformen für die Modelleingabe
                window_tensor = torch.tensor([window], dtype=torch.float).to(device)
                window_tensor = window_tensor.view(-1, len(window), input_size)

                # Modellausgabe
                output = model(window_tensor)
                propabilities = torch.softmax(output, dim=1)
                top_vals, top_indices = torch.topk(propabilities, candidates)

                candidate_predictions = top_indices[0].tolist()
                next_in_candidate_predictions = True if next_value in candidate_predictions else False
                if not next_in_candidate_predictions:
                    anomaly_detected = True
                    if label:
                        TP += 1
                    else:
                        FP += 1
                    break

            if not anomaly_detected:
                if label:
                    FN += 1
                else:
                    TN += 1

    return TP, TN, FP, FN


def calculate_f1(TP, TN, FP, FN, logger):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    logger.info(f" Evaluation results - TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}")
    return f1
