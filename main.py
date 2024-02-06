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
    parser.add_argument('--batch-size', type=int, default='2048', help='Input batch size for training')
    parser.add_argument('--input-size', type=int, default='1', help='Model input size')
    parser.add_argument('--num-layers', type=int, default='2', help='Number of hidden layers')
    parser.add_argument('--hidden-size', type=int, default='100', help='Size of the hidden layers')
    parser.add_argument('--epochs', type=int, default='30', help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default='0.001', help='Learning rate')
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

        if args.noparse:
            data_handler.parse()
        data_handler.read_structured_files()

        # Einlesen der Label-Datei. Diese enthält die Labels zu den Sequenzen
        anomaly_file = data_handler.read_anomaly_file()

        # Gruppieren der Einträge nach der Block-ID
        # grouped_hdfs enthält die Spalten BlockID, EventSequence und Label
        # EventSequence ist eine Liste von EventIDs
        # Wichtig: Bei den Trainingsdaten müssen alle Anomalien entfernt werden
        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('train'),
                          anomaly_file,
                          logger,
                          True),
            'train')
        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('validation'),
                          anomaly_file,
                          logger,
                          True),
            'validation')
        data_handler.set_grouped_data(
            group_entries(args.dataset,
                          data_handler.get_structured_data('eval'),
                          anomaly_file,
                          logger,
                          False),
            'eval')

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

        x_valid, y_valid = slice_windows(data_handler.get_transformed_data('validation'), args.window_size, logger, use_padding=False)
        data_handler.set_prepared_data(x_valid, y_valid, 'validation')

        x_eval, y_eval = slice_windows(data_handler.get_transformed_data('eval'), args.window_size, logger, use_padding=True)
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

        eval_x, eval_y = data_handler.get_prepared_data('eval')

        return_val_loss = False

        num_classes = len(data_handler.get_label_mapping())

        if args.model == 'deeplog':
            input_size = args.input_size
            hidden_size = args.hidden_size
            num_layers = args.num_layers
            learning_rate = args.learning_rate
            epochs = args.epochs
            window_size = args.window_size
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
                valid_loader,
                return_val_loss,
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

        eval_x, eval_y = data_handler.get_prepared_data('eval')

        evaluator = Evaluator(args, eval_x, eval_y, device, kwargs, logger, 1.0)
        f1 = evaluator.evaluate(model, args.candidates)
        evaluator.print_summary()

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
            hidden_size = trial.suggest_int('hidden_size', 20, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            # Es kann nicht mehr Kandidaten als Klassen geben
            candidates = trial.suggest_int('candidates', 3, min(num_classes, 15))
            num_layers = trial.suggest_int('layers', 1, 3)
            batch_size = trial.suggest_int('batch_size', 8, 2048)

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


            model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            trained_model, valid_loss = training.train(
                model,
                train_loader,
                learning_rate,
                epochs,
                window_size,
                logger,
                device,
                input_size,
                valid_loader,
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
        fig = vis.plot_contour(study, params=['candidates', 'hidden_size'])
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'data/contour_plot.png')

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

    # Vorbereitung der Log-Dateien:
    # - Parsen der Log-Dateien
    # - Gruppieren nach gemeinsamen IDs
    # - Zuordnen der Anomaly-Labels zu den gruppierten Sequenzen
    # - Umwandeln der Event-IDs in numerische IDs
    # - Speichern des erzeugten Datensatzes und des Mappings der Event-IDs zu numerischen IDs
    # - Das speichern des erzeugten Datensatzes, sowie des Mappings der Event-IDs zu numerischen IDs
"""
    if args.prepare:

        if args.dataset == 'hdfs':

            # Parsen der Log-Dateien
            logparser = preprocessing.HDFSLogParser(args.log_dir, args.data_dir, args.parser_type, logger)
            # logparser.parse(args.log_file)

        elif args.dataset == 'postgres':
            # Umwandeln mehrzeiliger Einträge in einzeilige Einträge
            logger.info(f"Converting Log-File {args.log_file} to singleline")
            preprocessing.postgres_to_singleline(args.log_dir, args.log_file, args.data_dir, logger)

            # Parsen der Log-Dateien
            logparser = preprocessing.PostgresLogParser(args.data_dir, args.data_dir, args.parser_type, logger)
            logparser.parse(args.log_file)

        # Einlesen der beim Parsing erzeugten _structured.csv Datei
        # Speichern in der variable structured_df. Die Spalten Date und Time haben den Datentyp str
        logger.info("Reading parsed files")
        structured_file = os.path.join(args.data_dir, args.log_file + '_structured.csv')
        structured_df = pd.read_csv(structured_file, dtype={'Date': str, 'Time': str})
        print("Path structured: ", structured_file)

        if args.dataset == 'hdfs':
            # Einlesen der Label-Datei. Diese enthält die Labels zu den Sequenzen
            anomaly_file_path = os.path.join(args.log_dir, args.anomaly_file)
            anomaly_df = pd.read_csv(anomaly_file_path)

            # Gruppieren der Einträge nach der Block-ID
            # grouped_hdfs enthält die Spalten BlockID, EventSequence und Label
            # EventSequence ist eine Liste von EventIDs
            grouped_hdfs = preprocessing.group_hdfs(structured_df, anomaly_df, logger, remove_anomalies=False)

            # Bilden von Fenstern der Größe window_size innerhalb der EventSequenze von grouped_hdfs
            # Die Fenster stehen in train_x. train_y enthält jeweils den nächsten Eintrag nach dem Fenster von train_x
            # train_x, train_y = preprocessing.slice_hdfs(grouped_hdfs, args.grouping, args.window_size, logger)
            num_processes = 12
            train_x, train_y = preprocessing.slice_hdfs_parallel(grouped_hdfs, args.grouping, args.window_size,
                                                                 num_processes, logger)

            # Umwandeln von EventIDs zu numerischen IDs
            # label_mapping enthält die Zuordnung der EventIDs zu den numerischen IDs
            feature_extractor = preprocessing.Vectorizer(logger)
            train_x_transformed, train_y_transformed, label_mapping = feature_extractor.fit_transform(train_x, train_y)
            print("Label-Mapping: ", label_mapping)

            #### Validierung anderer Lösung
            all_data = feature_extractor.transform_valid(grouped_hdfs)
            with open('hdfs_train', 'w') as f:
                for sequence in all_data['EventSequence']:
                    f.write(' '.join(map(str, sequence)) + '\n')

            # Speichern der erzeugten Trainingsdaten
            logger.info("Saving traing data")
            train_x_transformed.to_pickle(
                os.path.join(args.data_dir, "{}_x.pkl".format((args.log_file).replace('.log', ''))))
            train_y_transformed.to_pickle(
                os.path.join(args.data_dir, "{}_y.pkl".format((args.log_file).replace('.log', ''))))

            # Speichern des label_mapping in einer Pickle-Datei
            label_mapping_path = os.path.join(args.data_dir, 'label_mapping.pkl')
            with open(label_mapping_path, 'wb') as f:
                pickle.dump(label_mapping, f)

        elif args.dataset == 'postgres':
            pass

    if args.train:

        # Wenn kein CUDA verfügbar ist, soll mittels CPU trainiert werden
        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        logger.info("Loading Data")
        train_x, train_y, label_mapping = training.load_data(args.data_dir, args.log_file)

        num_classes = len(label_mapping)
        print("Label-Mapping: ", label_mapping)

        logger.info("Create Dataloader")
        train_loader = training.get_dataloader(train_x, train_y, args.batch_size, kwargs)

        if args.model == 'deeplog':
            input_size = args.input_size
            hidden_size = args.hidden_size
            num_layers = args.num_layers
            learning_rate = args.learning_rate
            epochs = args.epochs
            window_size = args.window_size
            batch_size = args.batch_size

            model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            log = 'adam_batch_size={}_epoch={}_log={}_layers={}_hidden={}_winsize={}_lr={}'.format(
                str(batch_size), str(epochs), args.log_file, args.num_layers,
                hidden_size, window_size, learning_rate)
            trained_model = training.train(model, train_loader, learning_rate, epochs, window_size, logger, log, device,
                                           input_size)

            save_model(trained_model, input_size, hidden_size, num_layers, num_classes, args.model_dir)

    if args.predict:

        # Wenn kein CUDA verfügbar ist, soll mittels CPU trainiert werden
        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')

        # Für die Evaluation wird eine Validierungs-Log-Datei und eine Datei mit den Labels für die Validierungsdatei
        # benötigt
        if args.validation_file and args.anomaly_file:

            if args.dataset == 'hdfs':
                # Einlesen der Labels in anomaly_df
                anomaly_df = pd.read_csv(os.path.join(args.log_dir, args.anomaly_file))

                # Parsen der Validierungs-Datei
                logparser = preprocessing.HDFSLogParser(args.log_dir, args.data_dir, args.parser_type, logger)
                # logparser.parse(args.validation_file)

                # Einlesen der beim Parsing erzeugten _structured.csv Datei
                # Speichern in der variable structured_df. Die Spalten Date und Time haben den Datentyp str
                structured_file = os.path.join(args.data_dir, args.validation_file + '_structured.csv')
                structured_df = pd.read_csv(structured_file, dtype={'Date': str, 'Time': str})
                # structured_df = structured_df[:200000]

                # Gruppieren der Einträge nach der Block-ID
                # grouped_hdfs enthält die Spalten BlockID, EventSequence und Label
                # EventSequence ist eine Liste von EventIDs
                # grouped_hdfs enthält auch anormale Einträge
                grouped_hdfs = preprocessing.group_hdfs(structured_df, anomaly_df, logger, remove_anomalies=False)

                # Einlesen und setzen des label_mappings im feature_extraktor
                label_mapping_path = os.path.join(args.data_dir, 'label_mapping.pkl')
                with open(label_mapping_path, 'rb') as f:
                    label_mapping = pickle.load(f)
                feature_extractor = preprocessing.Vectorizer(logger)
                feature_extractor.label_mapping = label_mapping

                # Umwandeln der EventIDs des Validierungsdatensatzes in numerische IDs
                # Dazu wird das label_mapping aus dem Training verwendet
                validate_x_transformed = feature_extractor.transform_valid(grouped_hdfs)

                test_normal = grouped_hdfs[grouped_hdfs['Label'] == 'Normal']
                test_abnormal = grouped_hdfs[grouped_hdfs['Label'] == 'Anomaly']

                hdfs_test_normal = feature_extractor.transform_valid(test_normal)
                hdfs_test_abnormal = feature_extractor.transform_valid(test_abnormal)

                with open('hdfs_test_normal', 'w') as f:
                    for sequence in hdfs_test_normal['EventSequence']:
                        f.write(' '.join(map(str, sequence)) + '\n')

                with open('hdfs_test_abnormal', 'w') as f:
                    for sequence in hdfs_test_abnormal['EventSequence']:
                        f.write(' '.join(map(str, sequence)) + '\n')

                model = load_model(args.model_dir, device)
                TP, TN, FP, FN = evaluation.evaluate(validate_x_transformed, model, device, args.candidates,
                                                     args.window_size, args.input_size, logger)
                print(evaluation.calculate_f1(TP, TN, FP, FN, logger))

            elif args.dataset == 'postgres':
                pass

    if args.hptuning:
        if not torch.cuda.is_available():
            logger.warning("No CUDA available")
            kwargs = {}
            device = torch.device("cpu")
        else:
            kwargs = {'num_workers': 1, 'pin_memory': True}
            device = torch.device("cuda")
            logger.info('Using CUDA')


        def objective(trial, device, train_loader, logger, x_validate):
            # num_layers = trial.suggest_int('num_layers', 1, 2)
            hidden_size = trial.suggest_int('hidden_size', 20, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            # Es kann nicht mehr Kandidaten als Klassen geben
            candidates = trial.suggest_int('candidates', 3, min(num_classes, 15))
            # batch_size = trial.suggest_int('batch_size', 64, 4096)

            input_size = 1
            epochs = 120
            window_size = 10
            batch_size = 2048
            num_layers = 2
            model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            log = 'adam_batch_size={}_epoch={}_log={}_layers={}_hidden={}_winsize={}_lr={}'.format(
                str(batch_size), str(epochs), args.log_file, args.num_layers,
                hidden_size, window_size, learning_rate)
            trained_model = training.train(model, train_loader, learning_rate, epochs, window_size, logger, log, device,
                                           input_size)

            TP, TN, FP, FN = evaluation.evaluate(x_validate, trained_model, device, candidates, window_size, input_size,
                                                 logger)
            print(x_validate)

            # Löschen des Modells und Freigeben des Speichers
            del model
            del trained_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            return evaluation.calculate_f1(TP, TN, FP, FN, logger)


        logger.info("Loading Data")
        train_x, train_y, label_mapping = training.load_data(args.data_dir, args.log_file)

        num_classes = len(label_mapping)

        logger.info("Create Dataloader")
        train_loader = training.get_dataloader(train_x, train_y, args.batch_size, kwargs)

        anomaly_df = pd.read_csv(os.path.join(args.log_dir, args.anomaly_file))

        structured_file = os.path.join(args.data_dir, args.validation_file + '_structured.csv')
        structured_df = pd.read_csv(structured_file, dtype={'Date': str, 'Time': str})

        grouped_hdfs = preprocessing.group_hdfs(structured_df, anomaly_df, logger, remove_anomalies=False)

        # Einlesen und setzen des label_mappings im feature_extraktor
        label_mapping_path = os.path.join(args.data_dir, 'label_mapping.pkl')
        with open(label_mapping_path, 'rb') as f:
            label_mapping = pickle.load(f)
        feature_extractor = preprocessing.Vectorizer(logger)
        feature_extractor.label_mapping = label_mapping

        validate_x_transformed = feature_extractor.transform_valid(grouped_hdfs)

        # Starte Optuna Studie
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, device, train_loader, logger, validate_x_transformed),
                       n_trials=10, gc_after_trial=True)

        print('Beste Hyperparameter:', study.best_params)

        df = study.trials_dataframe()
        df.to_csv('study_results.csv')
        optuna.visualization.plot_param_importances(study)
        optuna.visualization.plot_optimization_history(study)

        # Optimierungsgeschichte
        fig = vis.plot_optimization_history(study)
        fig.update_layout(width=800, height=600)  # Ändern Sie die Größe der Figur
        pio.write_image(fig, 'optimization_history.png')  # Speichern Sie die Figur als Bild

        # Parameter-Importanz
        fig = vis.plot_param_importances(study)
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'param_importances.png')

        # Konturdiagramm für zwei Hyperparameter
        fig = vis.plot_contour(study, params=['candidates', 'hidden_size'])
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'contour_plot.png')

        # Parallelkoordinaten-Plot
        fig = vis.plot_parallel_coordinate(study)
        fig.update_layout(width=800, height=600)
        pio.write_image(fig, 'parallel_coordinate.png')
"""
