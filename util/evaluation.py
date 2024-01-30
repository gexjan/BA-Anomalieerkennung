import torch


def evaluate(evaluation_df, model, device, candidates, window_size, input_size):
    TP, TN, FP, FN = 0, 0, 0, 0
    model.eval()

    with torch.no_grad():
        for _, row in evaluation_df.iterrows():
            sequence = row['EventSequence']
            label = row['Label'] == 'Anomaly'

            anomaly_detected = False

            seqlen = len(sequence)
            if seqlen < window_size + 1:
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


def calculate_f1(TP, TN, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Precision: {precision}, recall: {recall}, f1: {f1}")
    return f1
