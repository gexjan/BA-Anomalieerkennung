import argparse
from util.preprocessing import group_entries, slice_windows, create_label_mapping, transform_event_ids
import os
import logging
import pandas as pd
import pickle
import torch
from model.lstm import LSTM
from util.modelmanager import save_model, load_model
import random
import numpy as np
from util import training
from util import evaluation
import optuna
import optuna.visualization as vis
import plotly.io as pio
import plotly.graph_objects as go
from util.datahandler import DataHandler
import sys
from util.evaluation import Evaluator
from collections import Counter


import plotly.express as px


from util.training import get_dataloader
from datetime import timedelta

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# Seed-Wert festlegen
seed_value = 42

# PyTorch Seed setzen
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)

# NumPy Seed setzen
np.random.seed(seed_value)

# Python Random Seed setzen
random.seed(seed_value)

# Zusätzliche Konfigurationen für PyTorch, um weitere Zufälligkeiten zu minimieren
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # Initialer Parser, um den Dataset-Typ zu ermitteln
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, choices=['hdfs', 'postgres'], default='hdfs')
    pre_args, remaining_argv = pre_parser.parse_known_args()

    if pre_args.dataset == 'postgres':
        log_file_default = 'postgres_train.log'
        validation_file_default = 'postgres_validation.log'
        evaluation_file_default = 'postgres_test.log'
        log_dir_default = './logs/postgres'
    else:
        log_file_default = 'hdfs_train.log'
        validation_file_default = 'hdfs_validation.log'
        evaluation_file_default = 'hdfs_test.log'
        log_dir_default = './logs/HDFS'


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default=pre_args.dataset, choices=['hdfs', 'postgres'],
                        help='Choose the Dataset')
    parser.add_argument('--model', type=str, default='deeplog', choices=['deeplog', 'autoencoder'],
                        help='Choose the model')
    parser.add_argument('--model-dir', type=str, default='./model/',
                        help='The place where to store the model parameter')
    parser.add_argument('--data-dir', type=str, default='./data/', help='The place where to store the data')
    parser.add_argument('--log-dir', type=str, default=log_dir_default, help='The folder with the log-files')
    parser.add_argument('--log-file', type=str, default=log_file_default, help='The log used for training')
    parser.add_argument('--model-file', type=str, help='The name of the trained model')
    parser.add_argument('--grouping', type=str, default='session', choices=['session', 'time'],
                        help='Grouping of the Log-Entries. With HDFS is only session possible')


    parser.add_argument('-prepare', action='store_true', help='Pre-Process the Logs')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('-evaluate', action='store_true', help='Evaluate Model')
    parser.add_argument('-hptune', action='store_true', help='Hyperparameter tuning')
    parser.add_argument('-seqlen', action='store_true', help='Compare Sequence-Length')

    parser.add_argument('--noparse', action='store_false', help='Skip parsing')

    parser.add_argument('--parser-type', type=str, default='spell', choices=['spell', 'drain'],
                        help='Choose the parser')
    parser.add_argument('--window-size', type=int, default='10', help='Size of the windows')

    ## Training
    parser.add_argument('--batch-size', type=int, default='256', help='Input batch size for training')
    parser.add_argument('--input-size', type=int, default='1', help='Model input size')
    parser.add_argument('--num-layers', type=int, default='2', help='Number of hidden layers')
    parser.add_argument('--hidden-size', type=int, default='64', help='Size of the hidden layers')
    parser.add_argument('--epochs', type=int, default='30', help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default='0.01', help='Learning rate')
    parser.add_argument('--calculate-f', action='store_true', help='Pre-Process the Logs')
    parser.add_argument('--hptrials', type=int, default='10', help='Hyperparameter-tuning trials')
    parser.add_argument('--validation-file', type=str, default=validation_file_default,
                        help='File to validate the model')

    ## Evaluation
    parser.add_argument('--evaluation-file', type=str, default=evaluation_file_default,
                        help='File to Evaluate the model. Must contain normal and anormal entries.')
    parser.add_argument('--anomaly-file', type=str, default='anomaly_label.csv',
                        help='Contains the labels for the validation file')
    parser.add_argument('--candidates', type=int, default=9, help=("Number of prediction candidates"))
    args = parser.parse_args()

    if not args.model_file:
        args.model_file = f"{args.dataset}_bs{args.batch_size}_layers{args.num_layers}_hidden{args.hidden_size}_epochs{args.epochs}_lr{args.learning_rate}.pth"

    data_handler_file = os.path.join(args.data_dir, f"datahandler_{args.dataset}_{args.log_file}_{args.evaluation_file}_{args.parser_type}_{args.window_size}.pkl")

    if args.prepare:

        data_handler = DataHandler.create(args, logger, args.window_size)

        # Parsen der Log-Dateien
        if args.noparse:
            data_handler.parse()
        data_handler.read_structured_files()

        # Einlesen der Label-Datei. Diese enthält die Labels zu den Sequenzen
        anomaly_file = data_handler.read_anomaly_file()
        # anomaly_file = None

        # Gruppieren der Einträge nach der Block-ID
        # grouped_hdfs enthält die Spalten BlockID, EventSequence und Label
        # EventSequence ist eine Liste von EventIDs
        # Wichtig: Bei den Trainingsdaten müssen alle Anomalien entfernt werden
        # print("Davor: ", data_handler.get_structured_data('train'))
        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('train'),
                          anomaly_file,
                          logger,
                          True,
                          args.grouping),
            'train')
        logger.info(f"{len(data_handler.get_grouped_data('train'))} sequences in train-dataset")

        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('validation'),
                          anomaly_file,
                          logger,
                          True,
                          args.grouping),
            'validation')
        logger.info(f"{len(data_handler.get_grouped_data('validation'))} sequences in validation-dataset")

        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('eval'),
                          anomaly_file,
                          logger,
                          False,
                          args.grouping),
            'eval')
        logger.info(f"{len(data_handler.get_grouped_data('eval'))} sequences in evaluation-dataset")

        # Löschen der Variablen der eingelesenen Dateien
        data_handler.del_structured_files()

        # Umwandeln von EventIDs zu numerischen IDs
        # label_mapping enthält die Zuordnung der EventIDs zu den numerischen IDs
        # Das Wörterbuch soll nur die Werte aus dem Trainingsdatensatz enthalten
        data_handler.set_label_mapping(
            create_label_mapping(data_handler.get_grouped_data('train'), data_handler.get_grouped_data('validation'), logger)
        )

        data_handler.set_transformed_data(
            transform_event_ids(
                data_handler.get_grouped_data('train'),
                data_handler.get_label_mapping(),
                logger
            ),
            'train'
        )


        eventsequence = data_handler.get_transformed_data('train').iloc[0]['EventSequence']

        # Zählen der Häufigkeit jedes Ereignisses in der Sequenz
        event_counts = Counter(eventsequence)

        # Umwandeln des Counters in ein DataFrame für eine einfachere Handhabung
        df = pd.DataFrame(list(event_counts.items()), columns=['EventID', 'Count']).sort_values(by='EventID')

        # Erstellen eines Bar-Plots mit Plotly
        fig = go.Figure(data=[go.Bar(x=df['EventID'], y=df['Count'], marker_color='blue')])

        # Hinzufügen von Titel und Achsenbeschriftungen
        fig.update_layout(title='Event Frequency in Sequence',
                          xaxis_title='Event ID',
                          yaxis_title='Frequency',
                          template='plotly_white')

        # Anzeigen des Plots
        fig.show()

        # Zähle, wie oft jede EventSequence vorkommt
        # sequence_counts = data_handler.get_transformed_data('train')['EventSequence'].apply(lambda x: str(x)).value_counts()
        #
        # # Erstelle das Balkendiagramm mit Plotly
        # fig = px.bar(
        #     x=sequence_counts.index,
        #     y=sequence_counts.values,
        #     labels={'x': 'EventSequenz Nummer', 'y': 'Häufigkeit'},
        #     title='Verteilung der EventSequenzen'
        # )
        # fig.update_layout(xaxis_title="EventSequenz Nummer", yaxis_title="Häufigkeit")
        # fig.show()

        print("Mapping: ", data_handler.get_label_mapping())

        data_handler.set_transformed_data(
            transform_event_ids(
                data_handler.get_grouped_data('validation'),
                data_handler.get_label_mapping(),
                logger
            ),
            'validation'
        )

        data_handler.set_transformed_data(
            transform_event_ids(
                data_handler.get_grouped_data('eval'),
                data_handler.get_label_mapping(),
                logger
            ),
            'eval'
        )

        # Bilden von Fenstern der Größe window_size innerhalb der EventSequenze von grouped_hdfs
        # Die Fenster stehen in train_x. train_y enthält jeweils den nächsten Eintrag nach dem Fenster von train_x
        x_train, y_train = slice_windows(data_handler.get_transformed_data('train'), args.window_size, logger, use_padding=False)
        data_handler.set_prepared_data(x_train, y_train, 'train')

        # print(x_train)

        # window_counts = x_train['window'].apply(lambda x: str(x)).value_counts()
        # print("Counts: ", window_counts)
        #
        # # Erstelle ein DataFrame für das Plotting
        # plot_data = pd.DataFrame({
        #     'Window': window_counts.index,
        #     'Frequency': window_counts.values
        # })
        #
        # # Erstelle das Balkendiagramm mit Plotly
        # fig = px.bar(
        #     plot_data,
        #     x='Window',
        #     y='Frequency',
        #     labels={'Window': 'Fenster', 'Frequency': 'Häufigkeit'},
        #     title='Verteilung der Fensterhäufigkeiten'
        # )
        #
        # # Aktualisiere Layout für eine bessere Darstellung
        # fig.update_layout(xaxis_title="Fenster", yaxis_title="Häufigkeit", xaxis={'categoryorder': 'total descending'})
        #
        # # Zeige das Diagramm an
        # fig.show()

        x_valid, y_valid = slice_windows(data_handler.get_transformed_data('validation'), args.window_size, logger, use_padding=False)
        data_handler.set_prepared_data(x_valid, y_valid, 'validation')

        x_eval, y_eval = slice_windows(data_handler.get_transformed_data('eval'), args.window_size, logger, use_padding=True, time_grouping=args.grouping == 'time')
        data_handler.set_prepared_data(x_eval, y_eval, 'eval')


        # Speichern des Datahandler-Objekts in einer Datei
        with open(data_handler_file, 'wb') as f:
            pickle.dump(data_handler, f)

    if args.train:
        if not os.path.exists(data_handler_file):
            logger.error("No datahandler file. Rerun with argument -prepare")
            sys.exit(1)

        try:
            data_handler
        except NameError:
            logger.info("Loading data")
            with open(data_handler_file, 'rb') as f:
                    data_handler = pickle.load(f)

        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        train_x, train_y = data_handler.get_prepared_data('train')
        train_loader = get_dataloader(train_x, train_y, args.batch_size, kwargs)


        valid_x, valid_y = data_handler.get_prepared_data('validation')
        valid_loader = get_dataloader(valid_x, valid_y, args.batch_size, kwargs)

        if args.calculate_f:
            eval_x, eval_y = data_handler.get_prepared_data('eval')
            evaluator = Evaluator(args, eval_x, eval_y, device, kwargs, logger, 0.01)
        else:
            evaluator = None

        return_val_loss = False
        print(data_handler.get_label_mapping())
        num_classes = len(data_handler.get_label_mapping())
        print(num_classes)
        # num_classes = 30
        if args.model == 'deeplog':
            input_size = args.input_size
            hidden_size = args.hidden_size
            num_layers = args.num_layers
            learning_rate = args.learning_rate
            epochs = args.epochs
            window_size = args.window_size
            batch_size = args.batch_size

            model = LSTM(hidden_size, num_layers, num_classes).to(device)
            trained_model = training.train(
                model,
                train_loader,
                learning_rate,
                epochs,
                window_size,
                logger,
                device,
                num_classes,
                args.candidates,
                valid_loader,
                return_val_loss,
                evaluator,
                args.calculate_f)

            save_model(trained_model, input_size, hidden_size, num_layers, num_classes, args.data_dir, args.model_file, logger)


    if args.evaluate:
        if not os.path.exists(data_handler_file):
            logger.error("No datahandler file. Rerun with argument -prepare")
            sys.exit(1)

        if not os.path.exists(os.path.join(args.data_dir, args.model_file)):
            logger.error("No model trained")
            sys.exit(1)

        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        try:
            data_handler
        except NameError:
            logger.info("Loading data")
            with open(data_handler_file, 'rb') as f:
                data_handler = pickle.load(f)

        try:
            model = trained_model
        except NameError:
            model = load_model(args.data_dir, device, args.model_file, logger)

        num_classes = len(data_handler.get_label_mapping())
        # num_classes = 17
        # num_classes = 30

        eval_x, eval_y = data_handler.get_prepared_data('eval')

        evaluator = Evaluator(args, eval_x, eval_y, device, kwargs, logger, 1.0, args.grouping)
        f1_values = []
        for i in range(1,16):
            print("Candidates: ", i)
            #
            f1 = evaluator.evaluate(model, i, num_classes)
            evaluator.print_summary()
            f1_values.append(f1)
        #
        # f1 = evaluator.evaluate(model, args.candidates, num_classes)
        # evaluator.print_summary()

        fig = go.Figure(data=go.Scatter(y=f1_values, mode='lines'))
        fig.show()

    if args.hptune:
        if not os.path.exists(data_handler_file):
            logger.error("No datahandler file. Rerun with argument -prepare")
            sys.exit(1)

        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        try:
            data_handler
        except NameError:
            logger.info("Loading data")
            with open(data_handler_file, 'rb') as f:
                data_handler = pickle.load(f)

        def objective(trial, device, train_loader, valid_loader, logger, change_window_size, data_handler, num_classes):
            hidden_size = trial.suggest_int('hidden_size', 10, 300)
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
            # Es kann nicht mehr Kandidaten als Klassen geben
            # candidates = trial.suggest_int('candidates', 3, min(num_classes, 15))
            num_layers = trial.suggest_int('layers', 1, 5)
            batch_size = trial.suggest_int('batch_size', 32, 2048)

            input_size = args.input_size
            epochs = args.epochs

            # num_layers = args.num_layers

            # Verändern der Window-Size. Das erfordert die neuberechnung des Evaluations- und Trainingsdatensatzes
            # Zeitaufwendig
            if change_window_size:
                window_size = trial.suggest_int('window_size', 2, 10)
                data_handler.update_window_size(window_size, logger)

                train_x, train_y = data_handler.get_prepared_data('train')
                train_loader = get_dataloader(train_x, train_y, batch_size, kwargs)

                valid_x, valid_y = data_handler.get_prepared_data('validation')
                valid_loader = get_dataloader(valid_x, valid_y, batch_size, kwargs)

            else:
                window_size = args.window_size



            model = LSTM(hidden_size, num_layers, num_classes).to(device)
            trained_model, valid_loss = training.train(
                model,
                train_loader,
                learning_rate,
                epochs,
                window_size,
                logger,
                device,
                num_classes,
                valid_loader=valid_loader,
                return_val_loss=True
            )

            # Löschen des Modells und Freigeben des Speichers
            del model
            del trained_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            return valid_loss


        num_classes = len(data_handler.get_label_mapping())


        study = optuna.create_study(direction='minimize')
        change_window_size = False
        if change_window_size:
            train_loader = None,
            valid_loader = None
        else:
            train_x, train_y = data_handler.get_prepared_data('train')
            train_loader = get_dataloader(train_x, train_y, args.batch_size, kwargs)

            valid_x, valid_y = data_handler.get_prepared_data('validation')
            valid_loader = get_dataloader(valid_x, valid_y, args.batch_size, kwargs)



        study.optimize(lambda trial: objective(trial, device, train_loader, valid_loader, logger, change_window_size, data_handler, num_classes),
                       n_trials=args.hptrials, gc_after_trial=True)

        print('Beste Hyperparameter:', study.best_params)

        # Sammle die Laufzeiten aller Trials
        durations = [trial.datetime_complete - trial.datetime_start for trial in study.trials if
                     trial.datetime_complete and trial.datetime_start]

        # Berechne die Summe der Laufzeiten
        total_duration = sum(durations, timedelta())

        # Berechne die durchschnittliche Laufzeit
        average_duration = total_duration / len(durations)

        logger.info(f'Durchschnittliche Laufzeit eines Trials: {average_duration}')

        df = study.trials_dataframe()
        df.to_csv('data/study_results.csv')
        optuna.visualization.plot_param_importances(study)
        optuna.visualization.plot_optimization_history(study)

        # Optimierungsgeschichte
        fig = vis.plot_optimization_history(study)
        fig.update_layout(width=800, height=600)  # Ändern Sie die Größe der Figur
        pio.write_image(fig, 'data/optimization_history.png')  # Speichern Sie die Figur als Bild

        # Parameter-Importanz
        fig = vis.plot_param_importances(study)
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'data/param_importances.png')

        # Konturdiagramm für zwei Hyperparameter
        # fig = vis.plot_contour(study, params=['candidates', 'hidden_size'])
        # fig.update_layout(width=800, height=600)
        # pio.write_image(fig, 'data/contour_plot.png')

        # Parallelkoordinaten-Plot
        fig = vis.plot_parallel_coordinate(study)
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'data/parallel_coordinate.png')

    if args.seqlen:
        seqlen_results_file = 'data/seqlen_results.csv'

        if not os.path.exists(data_handler_file):
            logger.error("No datahandler file. Rerun with argument -prepare")
            sys.exit(1)

        try:
            data_handler
        except NameError:
            logger.info("Loading data")
            with open(data_handler_file, 'rb') as f:
                data_handler = pickle.load(f)

        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        epoch_f_results = {}

        for i in range(2,15):
            window_size = i
            data_handler.update_window_size(window_size, logger)
            train_x, train_y = data_handler.get_prepared_data('train')

            train_loader = get_dataloader(train_x, train_y, args.batch_size, kwargs)
            num_classes = len(data_handler.get_label_mapping())

            eval_x, eval_y = data_handler.get_prepared_data('eval')

            evaluator = Evaluator(args, eval_x, eval_y, device, kwargs, logger, 1.0)

            if args.model == 'deeplog':
                input_size = args.input_size
                hidden_size = args.hidden_size
                num_layers = args.num_layers
                learning_rate = args.learning_rate
                epochs = args.epochs
                batch_size = args.batch_size

                model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
                trained_model = training.train(
                    model,
                    train_loader,
                    learning_rate,
                    epochs,
                    window_size,
                    logger,
                    device,
                    input_size,
                    evaluator,
                    args.calculate_f)

                f1 = evaluator.evaluate(trained_model, args.candidates)
                evaluator.print_summary()
                epoch_f_results[i] = f1

                del model
                del trained_model

        # Konvertieren des epoch_f_results Dictionary in einen DataFrame
        df_results = pd.DataFrame(list(epoch_f_results.items()), columns=['window_size', 'f1_score'])

        # Speichern der Ergebnisse in einer CSV-Datei
        df_results.to_csv(seqlen_results_file, index=False)

        # Erstellen eines Scatter-Plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_results['window_size'], y=df_results['f1_score'],
                                 mode='lines+markers', name='F1 Score'))

        # Anpassen des Layouts
        fig.update_layout(title='Window Size vs. F1 Score',
                          xaxis_title='Window Size',
                          yaxis_title='F1 Score',
                          width=800, height=600)

        # Speichern des Bildes
        pio.write_image(fig, 'data/window_size_f1_score_plot.png')