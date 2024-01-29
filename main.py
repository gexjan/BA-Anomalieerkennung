import argparse
from util import preprocessing
import os
import logging
import pandas as pd
import pickle
from util.training import DeeplogTrainer, AutoencoderTrainer
from util.validate import Validator


logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='postgres', choices=['hdfs', 'postgres'] , help='Choose the Dataset')
    parser.add_argument('--model', type=str, default='deeplog', choices=['deeplog', 'autoencoder'], help='Choose the model')
    parser.add_argument('--model-dir', type=str, default='./model/', help='The place where to store the model parameter')
    parser.add_argument('--data-dir', type=str, default='./data/', help='The place where to store the data')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='The folder with the log-files')
    parser.add_argument('--log-file', type=str, default='postgresql-01.log', help='The log used for training')
    parser.add_argument('-prepare', action='store_true', help='Pre-Process the Logs')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('-predict', action='store_true', help='Detect anomalies')
    parser.add_argument('--parser-type', type=str, default='spell', choices=['spell', 'drain'] , help='Choose the parser')
    parser.add_argument('--window-size', type=int, default='5', help='Size of the windows')
    parser.add_argument('--grouping', type=str, default='sliding', choices=['sliding', 'session'], help='Group entries by sliding window or session')

    ## Training
    parser.add_argument('--seed', type=int, default='42', help='Seed')
    parser.add_argument('--batch-size', type=int, default='64', help='Input batch size for training')
    parser.add_argument('--input-size', type=int, default='1', help='Model input size')
    parser.add_argument('--num-layers', type=int, default='2', help='Number of hidden layers')
    parser.add_argument('--hidden-size', type=int, default='64', help='Size of the hidden layers')
    parser.add_argument('--epochs', type=int, default='10', help='Number of training epochs')

    ## Validation
    parser.add_argument('--validation-file', type=str, help='File to validate the model. Must contain normal and anormal entries.')
    parser.add_argument('--anomaly-file', type=str, default='anomaly_label.csv', help='Contains the labels for the validation file')
    parser.add_argument('--candidates', type=int, default=3, help=("Number of prediction candidates"))
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


        if args.dataset == 'hdfs':
            # Einlesen der Label-Datei. Diese enthält die Labels zu den Sequenzen
            anomaly_file_path = os.path.join(args.log_dir, args.anomaly_file)
            anomaly_df = pd.read_csv(anomaly_file_path)

            # Gruppieren der Einträge nach der Block-ID
            # grouped_hdfs enthält die Spalten BlockID, EventSequence und Label
            # EventSequence ist eine Liste von EventIDs
            grouped_hdfs = preprocessing.group_hdfs(structured_df, anomaly_df, logger)

            # Bilden von Fenstern der Größe window_size innerhalb der EventSequenze von grouped_hdfs
            # Die Fenster stehen in train_x. train_y enthält jeweils den nächsten Eintrag nach dem Fenster von train_x
            train_x, train_y = preprocessing.slice_hdfs(grouped_hdfs, args.grouping, args.window_size, logger)

            # Umwandeln von EventIDs zu numerischen IDs
            # label_mapping enthält die Zuordnung der EventIDs zu den numerischen IDs
            feature_extractor = preprocessing.Vectorizer()
            train_x_transformed, train_y_transformed, label_mapping = feature_extractor.fit_transform(train_x, train_y)

            # Speichern der erzeugten Trainingsdaten
            train_x_transformed.to_pickle(os.path.join(args.data_dir, "{}_x.pkl".format((args.log_file).replace('.log',''))))
            train_y_transformed.to_pickle(os.path.join(args.data_dir, "{}_y.pkl".format((args.log_file).replace('.log',''))))

            # Speichern des label_mapping in einer Pickle-Datei
            label_mapping_path = os.path.join(args.data_dir, 'label_mapping.pkl')
            with open(label_mapping_path, 'wb') as f:
                pickle.dump(label_mapping, f)

        elif args.dataset == 'postgres':
            pass

    if args.train:
        # Erzeugen eines Trainers
        if args.model == 'deeplog':
            trainer = DeeplogTrainer(args, logger)
        elif args.model == 'autoencoder':
            trainer = AutoencoderTrainer(args, logger)

        # Einlesen der Trainingsdaten X, Y und des label_mappings
        trainer.load_data()

        # Erstellen eines DataLoaders zur Dateneinspeisung in das Model
        data_loader = trainer.create_dataloader()

        # Trainieren des Modells
        trainer.train(data_loader)

    if args.predict:

        # Für die Evaluation wird eine Validierungs-Log-Datei und eine Datei mit den Labels für die Validierungsdatei benötigt
        if args.validation_file and args.anomaly_file:

            if args.dataset == 'hdfs':
                # Einlesen der Labels in anomaly_df
                anomaly_df = pd.read_csv(os.path.join(args.log_dir,args.anomaly_file))

                # Parsen der Validierungs-Datei
                logparser = preprocessing.HDFSLogParser(args.log_dir, args.data_dir, args.parser_type, logger)
                logparser.parse(args.validation_file)

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
                feature_extractor = preprocessing.Vectorizer()
                feature_extractor.label_mapping = label_mapping

                # Umwandeln der EventIDs des Validierungsdatensatzes in numerische IDs
                # Dazu wird das label_mapping aus dem Training verwendet
                validate_x_transformed = feature_extractor.transform_valid(grouped_hdfs)

                # Initialisieren des Validators
                # Dieser überprüft das Modell mit den Validierungsdaten und errechnet diverse Metriken
                validator = Validator(args, logger)
                results = validator.validate(validate_x_transformed)
                print(results)


            elif args.dataset == 'postgres':
                pass
                
