import pandas as pd
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optiom
from torch.utils.tensorboard import SummaryWriter
import time
import sys
from tqdm import tqdm


def load_data(data_dir, log_file):
    train_x = pd.read_pickle(os.path.join(data_dir, "{}_x.pkl".format((log_file).replace('.log', ''))))
    train_y = pd.read_pickle(os.path.join(data_dir, "{}_y.pkl".format((log_file).replace('.log', ''))))
    with open(os.path.join(data_dir, 'label_mapping.pkl'), 'rb') as f:
        label_mapping = pickle.load(f)

    return train_x, train_y, label_mapping


def get_dataloader(train_x, train_y, batch_size, kwargs):
    # X ist die Sequenz, Y der nächste Eintrag nach der Sequenz
    X = torch.tensor(train_x['window'].tolist(), dtype=torch.float)
    Y = torch.tensor(train_y['next'].values)

    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


def train(model, train_loader, learning_rate, epochs, window_size, logger, log, device, input_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optiom.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting DeepLog training with lr={learning_rate}, epochs={epochs}, layers={model.num_layers}, hidden_size={model.hidden_size}")
    # writer = SummaryWriter(log_dir='log/' + log)

    total_step = len(train_loader)
    print("Steps: ", total_step)


    for epoch in range(epochs):
        # Setzt das Modell in den Trainingsmodus. Dies ist wichtig, da einige Modelle sich im Trainings- und
        # Evaluierungsmodus unterschiedlich verhalten (z.B. Dropout, Batch Normalization...)
        model.train()

        # Initialisierung des Trainingsverlusts für die aktuelle Epoche
        train_loss = 0

        # Aufzeichnung der Startzeit zur Berechnung der Epochendauer
        start_time = time.time()

        # Initialisieren des tqdm Fortschrittsbalkens
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        # Iteration über alle Bachtes im Dataloader
        # for step, (seq, label) in enumerate(train_loader):
        for step, (seq, label) in enumerate(tqdm_loader):
            ### Forward pass: Berechnung der Modellvorhersagen
            # Die Eingabesequenz wird zunächst geklont, von früheren Berechnungen losgelöst, in die richtige Form gebracht
            # und dann auf das richtige Gerät (CPU oder GPU) verschoben
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)

            # Berechnung des Verlustes zwischen den Vorhersagen des Modells und den tatsächlichen Labels
            loss = criterion(output, label.to(device))

            ### Backward pass: Berechnung des Gradienten des Verlustes bezüglich der Modellparameter
            # Setzen aller Gradienten auf Null, um die Akkumulation aus früheren Schritten zu verhindern
            optimizer.zero_grad()

            # Berechnung des Gradienten des Verlustes
            loss.backward()

            # Summieren des Verlustes zur späteren Ausgabe
            train_loss += loss.item()

            # Anpassung der Modellparameter basierend auf den berechneten Gradienten
            # Aktualisierung der Modellparameter
            optimizer.step()

            # graph zu Tensorboard hinzufügen.
            # writer.add_graph(model, seq)

            # Aktualisieren des tqdm Fortschrittsbalkens
            batch_speed = step / (time.time() - start_time)
            tqdm_loader.set_postfix_str(f"batch/s={batch_speed:.2f}, loss={loss.item():.4f}")

        # Berechnung und ausgabe diverser Metriken
        end_time = time.time()
        epoch_duration = end_time - start_time
        # writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
        # logger.debug('Epoch [{}/{}], train_loss: {:.4f}, time: {}'.format(epoch + 1, epochs, train_loss / total_step,epoch_duration))
    logger.info(f"Finished Deeplog training. Last Loss: {train_loss / total_step}")

    # writer.close()
    return model