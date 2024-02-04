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
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from plotly.subplots import make_subplots
from util import evaluation


def get_dataloader(train_x, train_y, batch_size, kwargs):
    # X ist die Sequenz, Y der nächste Eintrag nach der Sequenz
    X = torch.tensor(train_x['window'].tolist(), dtype=torch.float)
    Y = torch.tensor(train_y['next'].values)

    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


def save_metrics_to_csv(epoch_losses, f1_scores, loss_file='./data/loss_per_epoch.csv', f1_file='./data/f1_per_epoch.csv'):
    loss_df = pd.DataFrame(epoch_losses, columns=['Train Loss'])
    loss_df.to_csv(loss_file, index_label='Epoch')

    if f1_scores:
        f1_df = pd.DataFrame(f1_scores, columns=['F1 Score'])
        f1_df.to_csv(f1_file, index_label='Epoch')

def plot_loss_and_f1(epoch_losses, f1_scores):
    # Loss-Wert plotten
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(y=epoch_losses, mode='lines', name='Train Loss'))
    loss_fig.update_layout(title='Training Loss per Epoch',
                      xaxis_title='Epoch',
                      yaxis_title='Loss')
    loss_fig.write_image('./data/training_loss.png')

    if f1_scores:
        # F1-Wert plotten
        f_fig = go.Figure()
        f_fig.add_trace(go.Scatter(y=f1_scores, mode='lines', name='F1 Score'))
        f_fig.update_layout(title='F1-Score per Epoch',
                            xaxis_title='Epoch',
                            yaxis_title='F1')
        f_fig.write_image('./data/f1_epoch.png')

        # Plot mit beiden Metriken erstellen
        combined_fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Train Loss hinzufügen
        combined_fig.add_trace(go.Scatter(y=epoch_losses, mode='lines', name='Train Loss'), secondary_y=False)

        # F1 Score hinzufügen
        combined_fig.add_trace(go.Scatter(y=f1_scores, mode='lines', name='F1 Score'), secondary_y=True)

        # Achsenbezeichnungen hinzufügen
        combined_fig.update_layout(
            title='Train Loss and F1 Score per Epoch',
            xaxis_title='Epoch'
        )

        # Y-Achsenbezeichnungen für beide Y-Achsen hinzufügen
        combined_fig.update_yaxes(title_text='Loss', secondary_y=False)
        combined_fig.update_yaxes(title_text='F1 Score', secondary_y=True)

        # Bild speichern
        combined_fig.write_image('./data/combined_loss_f1.png')




def train(model, train_loader, learning_rate, epochs, window_size, logger, device, input_size, evaluator=None, calculate_f = False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optiom.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []
    f1_scores = []

    logger.info(f"Starting DeepLog training with lr={learning_rate}, epochs={epochs}, layers={model.num_layers}, hidden_size={model.hidden_size}, window_size={window_size}")
    # writer = SummaryWriter(log_dir='log/' + log)

    total_step = len(train_loader)

    try:
        for epoch in range(epochs):
            # Setzt das Modell in den Trainingsmodus. Dies ist wichtig, da einige Modelle sich im Trainings- und
            # Evaluierungsmodus unterschiedlich verhalten (z.B. Dropout, Batch Normalization...)
            model.train()

            # Initialisierung des Trainingsverlusts für die aktuelle Epoche
            train_loss = 0

            # Aufzeichnung der Startzeit zur Berechnung der Epochendauer
            start_time = time.time()


            # Iteration über alle Bachtes im Dataloader
            # for step, (seq, label) in enumerate(train_loader):
            for step, (seq, label) in enumerate(train_loader):
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

                # Aktualisieren des tqdm Fortschrittsbalkens
                batch_speed = step / (time.time() - start_time)

            # Berechnung und ausgabe diverser Metriken
            end_time = time.time()
            epoch_duration = end_time - start_time

            epoch_loss = train_loss / total_step
            epoch_losses.append(epoch_loss)
            logger.info('Epoch [{}/{}], train_loss: {:.4f}, time: {}'.format(epoch + 1, epochs, train_loss / total_step,epoch_duration))

            if calculate_f:
                f1 = evaluator.evaluate(model, use_tqdm=True)
                evaluator.print_summary()
                f1_scores.append(f1)
                save_metrics_to_csv(epoch_losses, f1_scores)
                plot_loss_and_f1(epoch_losses, f1_scores)
    finally:
        plot_loss_and_f1(epoch_losses, f1_scores)
    logger.info(f"Finished Deeplog training. Last Loss: {train_loss / total_step}")

    return model