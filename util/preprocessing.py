import sys

from logparser.Spell import LogParser as SpellParser
from logparser.Drain import LogParser as DrainParser
import os
import re
import pandas as pd
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def postgres_to_singleline(log_files, log_dir, data_dir):
    """
    Diese Methode wandelt mehrzeilige Postgres-Logeinträge in einzeilige EInträge um
    :param log_files: Umzuwandelnde Log-Dateien
    :param log_dir: Pfad zu den Log-Dateien
    :param data_dir: Ausgabeverzeichnis
    """
    for logfile in log_files:
        log_path = os.path.join(log_dir, logfile)

        with open(log_path, 'r') as file:
            lines = file.readlines()

        output_path = os.path.join(data_dir, logfile)
        with open(output_path, 'w') as output_file:
            current_entry = ""

            for line in lines:
                # Überprüfen, ob die Zeile mit einem Zeitstempel beginnt
                if line[:4].isdigit() and line[4] == '-':
                    if current_entry:
                        output_file.write(current_entry.strip() + '\n')
                    current_entry = line.strip()
                else:
                    # Füge die Zeile dem aktuellen Eintrag hinzu, getrennt durch ein Pipe-Symbol
                    current_entry += ' | ' + line.strip()

            # Füge den letzten Eintrag hinzu
            if current_entry:
                output_file.write(current_entry.strip() + '\n')



def group_entries(dataset, df, anomaly_df, logger, train_data, grouping):
    """
    Diese Methode gruppiert alle Log-Einträge nach bestimmten IDs
    Bei Postgres kann nach PID gruppiert werden, bei HDFS nach Blockid
    :param dataset: hdfs oder postgres
    :param df: Dataframe der eingelesenen structured files
    :param anomaly_df: Enthält die Labels zu den IDs
    :param logger: logger object
    :param train_data: Sind die zu gruppierenden Daten trainingsdaten (true) oder evaluationsdaten (false)
    :param grouping: Soll nach der ID oder Zeit gruppiert werden
    :return:
    """
    logger.info("Grouping Log Files")
    if grouping == 'session':
        if dataset == 'hdfs':
            anomaly_file_col = 'BlockId'
            regex_pattern = re.compile(r'(blk_-?\d+)')

            # Hinzufügen einer neuen Spalte für die extrahierte Block-ID
            df['SeqID'] = df['Content'].apply(
                lambda x: regex_pattern.search(x).group(0) if regex_pattern.search(x) else None)

            # Gruppierung der Daten nach Block-ID und Sammeln der zugehörigen EventIDs
            grouped = df.groupby('SeqID')['EventId'].apply(list)

            grouped = grouped.reset_index()
            grouped.columns = ['SeqID', 'EventSequence']

            # Zusammenführen der DataFrames anhand der Block-ID
            merged_df = pd.merge(grouped, anomaly_df, left_on='SeqID', right_on=anomaly_file_col, how='left')
            merged_df.drop(columns=[anomaly_file_col], inplace=True)

            # Überprüfen auf Einträge in 'grouped' die nicht in 'anomaly_df' vorhanden sind
            missing_labels = set(grouped['SeqID']) - set(anomaly_df[anomaly_file_col])
            if missing_labels:
                logger.info(f"Einträge mit folgenden SeqIDs fehlen in anomaly_df: {missing_labels}")

            if train_data:
                # Entfernen der anomalen Einträge
                merged_df = merged_df[merged_df['Label'] == 'Normal']
                # Setze alle Labels auf 'None'
                merged_df['Label'] = 'None'

        if dataset == 'postgres':
            anomaly_file_col = 'pid'
            regex_pattern = re.compile(r'\[(\d+)\]')

            df['SeqID'] = df['PID'].apply(
                lambda x: int(regex_pattern.search(x).group(1)) if regex_pattern.search(x) else None)

            # Gruppierung der Daten nach Block-ID und Sammeln der zugehörigen EventIDs
            grouped = df.groupby('SeqID')['EventId'].apply(list)
            grouped = grouped.reset_index()
            grouped.columns = ['SeqID', 'EventSequence']

            if train_data:
                grouped['Label'] = None
                merged_df = grouped

            else:
            # Zusammenführen der DataFrames anhand der Block-ID
                merged_df = pd.merge(grouped, anomaly_df, left_on='SeqID', right_on=anomaly_file_col, how='left')
                merged_df.drop(columns=[anomaly_file_col], inplace=True)

                # Überprüfen auf Einträge in 'grouped' die nicht in 'anomaly_df' vorhanden sind
                missing_labels = set(grouped['SeqID']) - set(anomaly_df[anomaly_file_col])
                if missing_labels:
                    logger.info(f"Einträge mit folgenden SeqIDs fehlen in anomaly_df: {missing_labels}")

        # Durchschnittliche Gruppenlänge + Median
        avg_group_length = grouped['EventSequence'].apply(len).mean()
        median_group_length = grouped['EventSequence'].apply(len).median()

        data_type = "Training" if train_data else "Testing"
        logger.info(
            f"{data_type} data: Average group length: {avg_group_length}; Median group length: {median_group_length}")

        return merged_df

    elif grouping == 'time':
        # if dataset == 'hdfs':
        #     logger.fatal("No grouping available for HDFS")
        #     sys.exit(0)

        # print(df[:10].to_string())

        if train_data:
            anomaly_labels = [None]
        else:
            anomaly_labels = anomaly_df['Label'].tolist()

        # Erstellen eines DataFrames ohne Gruppierung
        # SeqID wird auf 'time' gesetzt, und die EventSequence enthält alle EventIDs
        all_events = df['EventId'].tolist()  # Sammeln aller EventIDs in eine Liste
        no_group_df = pd.DataFrame({'SeqID': ['time'],
                                    'EventSequence': [all_events],
                                    'Label': [anomaly_labels]})

        # Durchschnittliche Länge und Median der EventSequence setzen, in diesem Fall die Länge der gesamten EventList
        avg_group_length = len(all_events)
        median_group_length = len(all_events)  # Da es nur eine 'Gruppe' gibt, sind Durchschnitt und Median identisch

        data_type = "Training" if train_data else "Testing"
        logger.info(
            f"{data_type} data without grouping: Total number of events: {len(all_events)}; Average group length: {avg_group_length}; Median group length: {median_group_length}")

        return no_group_df



# Nachfolgende Methode 'slice_hdfs' verwendet den 'Sliding Window'-Ansatz für die Vorverarbeitung von Sequenzdaten für LSTM-Modelle.
# Der 'Sliding Window'-Ansatz ist wichtig, da er hilft, zeitliche Abhängigkeiten in den Daten zu erfassen. Er bietet dem Modell eine Sequenz
# von vorherigen Ereignissen (Inputs), was ihm ermöglicht, zeitliche Muster und Dynamiken besser zu verstehen.
# Ein 'Sliding Window' liefert mehr Kontext für jede Vorhersage und hilft, zukünftige Ereignisse basierend auf diesem Kontext vorherzusagen.
# Die Größe des Fensters ist entscheidend, da sie bestimmt, wie viele vergangene Informationen für die Vorhersage zur Verfügung stehen.
# Eine geeignete Fenstergröße hilft, das Gleichgewicht zwischen dem Erfassen relevanter Muster und dem Vermeiden irrelevanter Informationen zu finden.
# Dieser Ansatz wandelt die Daten in einen reichhaltigeren Merkmalssatz um, indem er jedes Fenster effektiv zu einem Vektor von Merkmalen macht.
def process_windowing(data, use_padding, time_grouping):
    seq_id, row, window_size = data
    sequence = row['EventSequence']
    label = row['Label']
    seqlen = len(sequence)
    windows = []

    # Bei time wird die Sequenzlänge nie kürzer als die window-size sein
    if use_padding and seqlen < window_size:
        padded_sequence = sequence + [1] * (window_size - seqlen) # 1 entspricht '#PAD'
        windows.append([seq_id, padded_sequence, 1, label]) # 1 entspricht '#PAD'
    else:
        i = 0
        while (i + window_size) < seqlen:
            window_slice = sequence[i: i + window_size]
            next_element = sequence[i + window_size] if (i + window_size) < seqlen else 1 # 1 entspricht '#PAD'
            if time_grouping:
                win_label = label[i + window_size]
            else:
                win_label = label
            windows.append([seq_id, window_slice, next_element, win_label])
            i += 1
    return windows


def slice_windows(df, window_size, logger, use_padding, time_grouping=False):
    logger.info("Slicing windows")
    num_threads = 10  # Anzahl der Threads beim Slicen
    data_splits = [(index, row, window_size) for index, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda data: process_windowing(data, use_padding, time_grouping), data_splits))

    windows = [item for sublist in results for item in sublist]
    sliced_windows = pd.DataFrame(windows, columns=['SeqID', 'window', 'next', 'label'])
    train_x = sliced_windows[['SeqID', 'window', 'label']]
    train_y = sliced_windows[['SeqID', 'next']]

    return train_x, train_y




def create_label_mapping(df1, df2, logger):
    logger.info("Create label mapping")
    label_mapping = {'#OOV': 0, '#PAD': 1}
    next_id_value = 2

    for index, row in df1.iterrows():
        for event_id in row['EventSequence']:
            if event_id not in label_mapping and event_id != '#PAD':
                label_mapping[event_id] = next_id_value
                next_id_value += 1

    # for index, row in df2.iterrows():
    #     for event_id in row['EventSequence']:
    #         if event_id not in label_mapping and event_id != '#PAD':
    #             label_mapping[event_id] = next_id_value
    #             next_id_value += 1
    return label_mapping


def transform_event_ids(dataset, mapping, logger):
    logger.info("Transforming IDs")
    dataset_transformed = dataset.copy()
    for index, row in dataset.iterrows():
        dataset_transformed.at[index, 'EventSequence'] = [mapping.get(event_id, mapping['#OOV']) for event_id in
                                                   row['EventSequence']]

    return dataset_transformed
