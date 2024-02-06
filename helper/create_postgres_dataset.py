import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from datetime import timedelta
import os


def multiline_to_singleline(input_file, output_file):
    with open(input_file, 'r') as file, open(output_file, 'w') as output_file:
        current_entry = ""
        for line in file:
            if line[:4].isdigit() and line[4] == '-':
                if current_entry:
                    output_file.write(current_entry.strip() + '\n')
                current_entry = line.strip()
            else:
                current_entry += ' | ' + line.strip()
        if current_entry:
            output_file.write(current_entry.strip() + '\n')

def singleline_to_multiline(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(' | ')
            outfile.write(parts[0])
            for part in parts[1:]:
                outfile.write('\n\t' + part)
            outfile.write('\n')

def read_and_parse_log(file_path):
    columns = ['date', 'time', 'timezone', 'pid', 'message']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' ', 4)
            # Entfernen des Newline-Zeichens am Ende jeder Zeile und Strippen der Teile
            parts = [part.strip() for part in parts]
            # Entfernt die Klammern von der PID
            parts[3] = parts[3].strip('[]')  
            data.append(parts)
    df = pd.DataFrame(data, columns=columns)
    # Konvertieren der 'pid' Spalte zu int
    df['pid'] = df['pid'].astype(int)
    return df


def generate_random_ip():
    return '.'.join(str(random.randint(0, 255)) for _ in range(4))

def generate_random_port():
    return random.randint(1024, 65535)


# Entfernt die ganze Gruppe, wenn eine der in unwanted_messages folgenden Meldungen vorkommt
def filter_groups(df):
    unwanted_messages = [
        "skipping special file",
        "FATAL could not receive data from WAL stream server closed the connection unexpectedly",
        "FATAL:  could not connect to the primary server: could not connect to server: Connection refused"
    ]
    
    # Prüfen, ob irgendeine der unerwünschten Nachrichten in den Log-Einträgen der Gruppe enthalten ist
    def is_unwanted_group(group):
        return any(unwanted_message in message for message in group['message'] for unwanted_message in unwanted_messages)
    
    removed_pids = []
    # Funktion zum Filtern und Erfassen von entfernten PIDs
    def filter_and_capture(x):
        if is_unwanted_group(x):
            removed_pids.append(x['pid'].iloc[0])
            return False
        return True
    
    filtered_df = df.groupby('pid').filter(filter_and_capture)
    
    return filtered_df, removed_pids

def find_next_free_pid_after_time(used_pids, start_time, test_df):
    # Anstatt 'used_pids_before_time' aus 'test_df' zu extrahieren, verwenden wir die externe Liste 'used_pids'
    if not used_pids:
        return 1  # Starten mit PID 1, wenn keine PIDs vorhanden sind
    
    next_pid = max(used_pids) + 1
    return next_pid

def generate_artificial_logs(template_list, test_df):
    used_pids = set(test_df['pid'].unique())
    artificial_logs = []
    start_time = test_df['datetime'].min()
    end_time = test_df['datetime'].max()
    total_duration = end_time - start_time
    eighty_percent_duration = total_duration * 0.999
    total_milliseconds_in_eighty_percent = int(eighty_percent_duration.total_seconds() * 1000)

    for item in template_list:
        count, messages, generators = item
        for _ in range(count):
            random_milliseconds_within_eighty_percent = random.randint(0, total_milliseconds_in_eighty_percent)
            random_start_point = start_time + timedelta(microseconds=random_milliseconds_within_eighty_percent * 1000)
            current_pid = find_next_free_pid_after_time(used_pids, random_start_point, test_df)
            used_pids.add(current_pid)
            current_time = random_start_point
            for message in messages:
                # Prüfe, ob Generatoren für die Nachricht vorhanden sind, und ersetze die Platzhalter
                if generators:
                    formatted_message = message.format(*[gen() for gen in generators])
                else:
                    formatted_message = message
                log_entry = [current_time.strftime('%Y-%m-%d'), current_time.strftime('%H:%M:%S.%f')[:-3], 'CET', str(current_pid), formatted_message]
                artificial_logs.append(log_entry)
                remaining_time = (end_time - current_time).total_seconds() * 1000
                random_additional_milliseconds = random.randint(0, int(remaining_time))
                current_time += timedelta(microseconds=random_additional_milliseconds * 1000)

    return pd.DataFrame(artificial_logs, columns=['date', 'time', 'timezone', 'pid', 'message'])



def write_log_entries(df, file_path):
    with open(file_path, 'w') as file:
        for index, row in df.iterrows():
            base_log_entry = f"{row['date']} {row['time']} {row['timezone']} [{row['pid']}] "

            # Prüfe, ob die Nachricht '|' enthält und teile sie entsprechend auf
            if '|' in row['message']:
                parts = row['message'].split(' | ')
                # Schreibe den ersten Teil direkt nach dem Grundteil der Log-Nachricht
                file.write(base_log_entry + parts[0])
                # Füge die weiteren Teile als zusätzliche Zeilen mit korrekter Einrückung hinzu
                for part in parts[1:]:
                    file.write('\n\t\t' + part)
                file.write('\n')  # Füge einen Zeilenumbruch am Ende der gesamten Nachricht hinzu
            else:
                # Wenn keine '|' vorhanden sind, schreibe die Nachricht wie zuvor
                file.write(base_log_entry + row['message'] + '\n')


if __name__ == '__main__':
    base_dir = '../logs/postgres'
    db_dir = 'summarized'
    log_dir = os.path.join(base_dir, db_dir)

    log_file = 'postgresql029e.log'
    multiline_path = os.path.join(log_dir, log_file)
    print(multiline_path)

    log_file_path = os.path.join(base_dir, log_file)

    test_log_file_path = os.path.join(base_dir, 'postgres_test.log')
    train_log_file_path = os.path.join(base_dir, 'postgres_train.log')
    labels_file_path = os.path.join(base_dir, 'anomaly_label.csv')


    test_size = 0.9

    multiline_to_singleline(multiline_path, log_file_path)

    log_df = read_and_parse_log(log_file_path)

    # Filtern der Daten
    filtered_log_df, removed_pids = filter_groups(log_df)
    # print(f"Entfernte PIDs aufgrund unerwünschter Nachrichten: {removed_pids}")

    unique_pids = filtered_log_df['pid'].unique()
    train_pids, test_pids = train_test_split(unique_pids, test_size=test_size, random_state=42)

    train_df = filtered_log_df[filtered_log_df['pid'].isin(train_pids)].copy()
    test_df = filtered_log_df[filtered_log_df['pid'].isin(test_pids)].copy()

    # Liste der hinzuzufügenden Anomalien
    template_list = [
        [1, ["Starting connection", "Authorizing", "No entry in .authorized"], []],
        [1, ["Starting connection", "Authorizing", "No IP"], []],
        [22, ["FATAL: could not connect to the primary server: could not connect to server: Connection refused | Is the server running on host \"{}\" and accepting | TCP/IP connections on port 5432?"], [generate_random_ip]]
    ]

    test_df['datetime'] = pd.to_datetime(test_df['date'] + ' ' + test_df['time'])
    # start_time = test_df['datetime'].min()
    # end_time = test_df['datetime'].max()

    artificial_logs_df = generate_artificial_logs(template_list, test_df)

    # Zusammenführen des ursprünglichen und neuen Datensatzes
    test_df_combined = pd.concat([test_df, artificial_logs_df]).sort_values(by=['datetime']).reset_index(drop=True)

    # Datetime-Spalte aktualisieren
    test_df_combined['datetime'] = pd.to_datetime(test_df_combined['date'] + ' ' + test_df_combined['time'])

    # Sortieren nach datetime
    test_df_combined = test_df_combined.sort_values(by=['datetime'])


    # Extrahiere PIDs, die im ursprünglichen Test-Datensatz vorhanden waren, sind "normal"
    original_pids = test_df['pid'].unique()
    print(original_pids)
    print(len(original_pids))

    # Erstelle eine Liste aller PIDs im kombinierten DataFrame
    combined_pids = test_df_combined['pid'].unique()
    print(len(combined_pids))

    # Bestimme die PIDs der künstlich hinzugefügten Einträge
    artificial_pids = [pid for pid in combined_pids if pid not in original_pids]
    print(artificial_pids)

    # Erstelle ein DataFrame für die Labels
    labels_df = pd.DataFrame({
        'pid': combined_pids,
        'Label': ['Anomaly' if pid in artificial_pids else 'Normal' for pid in combined_pids]
    })


    # In Dateien speichern
    labels_df.to_csv(labels_file_path, index=False)

    write_log_entries(test_df_combined, test_log_file_path)
    write_log_entries(train_df, train_log_file_path)

    # Laden der anomaly_label.csv
    anomaly_label_df = pd.read_csv(labels_file_path)

    # Umwandeln der 'pid' Spalte zu int, falls sie noch nicht vom Typ int ist
    anomaly_label_df['pid'] = anomaly_label_df['pid'].astype(int)

    # Finden der PIDs, die in der Log-Datei vorkommen, aber nicht in der anomaly_label.csv
    missing_pids_in_labels = set(combined_pids) - set(anomaly_label_df['pid'])

    # Überprüfen, ob es fehlende PIDs gibt und Ausgabe
    if missing_pids_in_labels:
        print(f"Fehlende PIDs in anomaly_label.csv: {missing_pids_in_labels}")
    else:
        print("Alle PIDs der Log-Datei kommen in der anomaly_label.csv vor.")
