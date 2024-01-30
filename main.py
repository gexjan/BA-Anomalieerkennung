import argparse
from util import preprocessing
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
import sys


logging.basicConfig(level=logging.DEBUG,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hdfs', choices=['hdfs', 'postgres'],
                        help='Choose the Dataset')
    parser.add_argument('--model', type=str, default='deeplog', choices=['deeplog', 'autoencoder'],
                        help='Choose the model')
    parser.add_argument('--model-dir', type=str, default='./model/',
                        help='The place where to store the model parameter')
    parser.add_argument('--data-dir', type=str, default='./data/', help='The place where to store the data')
    parser.add_argument('--log-dir', type=str, default='./logs/HDFS', help='The folder with the log-files')
    parser.add_argument('--log-file', type=str, default='hdfs_train.log', help='The log used for training')

    parser.add_argument('-prepare', action='store_true', help='Pre-Process the Logs')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('-predict', action='store_true', help='Detect anomalies')
    parser.add_argument('-hptuning', action='store_true', help='Hyperparameter tuning')

    parser.add_argument('--parser-type', type=str, default='spell', choices=['spell', 'drain'],
                        help='Choose the parser')
    parser.add_argument('--window-size', type=int, default='10', help='Size of the windows')
    parser.add_argument('--grouping', type=str, default='sliding', choices=['sliding', 'session'],
                        help='Group entries by sliding window or session')

    ## Training
    parser.add_argument('--batch-size', type=int, default='2048', help='Input batch size for training')
    parser.add_argument('--input-size', type=int, default='1', help='Model input size')
    parser.add_argument('--num-layers', type=int, default='2', help='Number of hidden layers')
    parser.add_argument('--hidden-size', type=int, default='100', help='Size of the hidden layers')
    parser.add_argument('--epochs', type=int, default='10', help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='Learning rate')

    ## Evaluation
    parser.add_argument('--validation-file', type=str, default='hdfs_test.log',
                        help='File to validate the model. Must contain normal and anormal entries.')
    parser.add_argument('--anomaly-file', type=str, default='anomaly_label.csv',
                        help='Contains the labels for the validation file')
    parser.add_argument('--candidates', type=int, default=9, help=("Number of prediction candidates"))
    args = parser.parse_args()

    # Vorbereitung der Log-Dateien:
    # - Parsen der Log-Dateien
    # - Gruppieren nach gemeinsamen IDs
    # - Zuordnen der Anomaly-Labels zu den gruppierten Sequenzen
    # - Umwandeln der Event-IDs in numerische IDs
    # - Speichern des erzeugten Datensatzes und des Mappings der Event-IDs zu numerischen IDs
    # - Das speichern des erzeugten Datensatzes, sowie des Mappings der Event-IDs zu numerischen IDs
    if args.prepare:
        if args.dataset == 'hdfs':

            # Parsen der Log-Dateien
            logparser = preprocessing.HDFSLogParser(args.log_dir, args.data_dir, args.parser_type, logger)
            logparser.parse(args.log_file)

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
            train_x, train_y = preprocessing.slice_hdfs_parallel(grouped_hdfs, args.grouping, args.window_size, num_processes, logger)

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
                TP, TN, FP, FN = evaluation.evaluate(validate_x_transformed, model, device, args.candidates, args.window_size, args.input_size, logger)
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
            num_layers = trial.suggest_int('num_layers', 1, 2)
            hidden_size = trial.suggest_int('hidden_size', 20, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            # Es kann nicht mehr Kandidaten als Klassen geben
            candidates = trial.suggest_int('candidates', 3, min(num_classes, 15))

            input_size = 1
            epochs = 120
            window_size = 10
            batch_size = 2048

            model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
            log = 'adam_batch_size={}_epoch={}_log={}_layers={}_hidden={}_winsize={}_lr={}'.format(
                str(batch_size), str(epochs), args.log_file, args.num_layers,
                hidden_size, window_size, learning_rate)
            trained_model = training.train(model, train_loader, learning_rate, epochs, window_size, logger, log, device,
                                           input_size)

            TP, TN, FP, FN = evaluation.evaluate(validate_x_transformed, trained_model, device, candidates, window_size, input_size, logger)

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
        study.optimize(lambda trial: objective(trial, device, train_loader, logger, validate_x_transformed), n_trials=10, gc_after_trial=True)

        print('Beste Hyperparameter:', study.best_params)