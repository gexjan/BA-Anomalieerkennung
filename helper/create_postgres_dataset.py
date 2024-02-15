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

def read_and_parse_log(file_path):
    columns = ['date', 'time', 'timezone', 'pid', 'message']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(' ', 4)
            parts = [part.strip() for part in parts]
            parts[3] = parts[3].strip('[]')  
            data.append(parts)
    df = pd.DataFrame(data, columns=columns)
    df['pid'] = df['pid'].astype(int)
    return df


def generate_random_session_time():
    hours = 0
    minutes = 0
    seconds = random.randint(0, 9)
    milliseconds = random.randint(0, 999)

    return f"{00}:{00}:{seconds:02}.{milliseconds:03}"


def generate_random_ip():
    return '.'.join(str(random.randint(0, 255)) for _ in range(4))

def generate_random_port():
    return random.randint(1024, 65535)


# Entfernt die ganze Gruppe, wenn eine der nachfolgenden Nachrichten vorkommt
def filter_groups(df):
    unwanted_messages = [
        "no pg_hba.conf entry for host",
        "using stale statistics instead of current ones because stats collector is not responding",
        "could not send data to WAL stream SSL",
        "database system was interrupted while in recovery",
        "invalid record length at",
        "could not connect to the primary server",
        "WARNING:  skipping special file",
        "FATAL:  could not receive data from WAL stream",
        "LOG:  invalid resource manager ID",
        "FATAL:  database",
        "ERROR:  function",
        "ERROR:  cannot execute",
        "ERROR:  could not access file",
        "ERROR:  unrecognized configuration parameter",
        "FATAL:  terminating",
        "could not send end-of-streaming message",
        "could not connect to the primary server",
        "ERROR:  canceling statement due",
        "ERROR:  GROUP BY",
        "ERROR:  syntax error at or near",
        "FATAL:  connection to client lost",
        "ERROR:  deadlock detected",
        "current transaction is aborted, commands ignored until end of transaction block",
        "there is already a transaction in progress",
        "could not receive data from client: Connection reset by peer"
    ]

    def is_unwanted_group(group):
        return any(unwanted_message in message for message in group['message'] for unwanted_message in unwanted_messages)
    
    removed_pids = []
    def filter_and_capture(x):
        if is_unwanted_group(x):
            removed_pids.append(x['pid'].iloc[0])
            return False
        return True
    
    filtered_df = df.groupby('pid').filter(filter_and_capture)
    
    return filtered_df, removed_pids

def find_next_free_pid_after_time(used_pids, start_time, test_df):
    if not used_pids:
        return 1  # Starten mit PID 1, wenn keine PIDs vorhanden sind
    
    next_pid = max(used_pids) + 1
    return next_pid

def generate_artificial_logs(template_list, test_df):
    used_pids = set(test_df['pid'].unique())
    artificial_logs = []
    anomaly_timestamps = []
    start_time = test_df['datetime'].min()
    end_time = test_df['datetime'].max()
    total_duration = end_time - start_time
    eighty_percent_duration = total_duration * 0.999
    total_milliseconds_in_eighty_percent = int(eighty_percent_duration.total_seconds() * 1000)

    for item in template_list:
        count, messages, generators, anomalies = item
        for _ in range(count):
            random_milliseconds_within_eighty_percent = random.randint(0, total_milliseconds_in_eighty_percent)
            random_start_point = start_time + timedelta(microseconds=random_milliseconds_within_eighty_percent * 1000)
            current_pid = find_next_free_pid_after_time(used_pids, random_start_point, test_df)
            used_pids.add(current_pid)
            current_time = random_start_point
            for idx, message in enumerate(messages):
                if generators:
                    formatted_message = message.format(*[gen() for gen in generators])
                else:
                    formatted_message = message
                log_entry = [current_time.strftime('%Y-%m-%d'), current_time.strftime('%H:%M:%S.%f')[:-3], 'CET', str(current_pid), formatted_message]
                artificial_logs.append(log_entry)
                if idx in anomalies:
                    anomaly_timestamps.append((current_time, 'Anomaly'))
                else:
                    anomaly_timestamps.append((current_time, 'Normal'))
                remaining_time = (end_time - current_time).total_seconds() * 1000 * 0.005 # Verwenden des Faktors 0.005, um den Abstand zwischen den einzelnen Log-Meldungen zu reduzieren
                random_additional_milliseconds = random.randint(0, int(remaining_time))
                current_time += timedelta(microseconds=random_additional_milliseconds * 1000)

    artificial_logs_df = pd.DataFrame(artificial_logs, columns=['date', 'time', 'timezone', 'pid', 'message'])
    anomaly_timestamps_df = pd.DataFrame(anomaly_timestamps, columns=['datetime', 'Label'])
    return artificial_logs_df, anomaly_timestamps_df




def write_log_entries(df, file_path):
    with open(file_path, 'w') as file:
        for index, row in df.iterrows():
            base_log_entry = f"{row['date']} {row['time']} {row['timezone']} [{row['pid']}] "

            # Prüfe, ob die Nachricht '|' enthält und teile sie entsprechend auf
            if '|' in row['message']:
                parts = row['message'].split(' | ')
                file.write(base_log_entry + parts[0])
                for part in parts[1:]:
                    file.write('\n\t\t' + part)
                file.write('\n')
            else:
                file.write(base_log_entry + row['message'] + '\n')


if __name__ == '__main__':
    base_dir = '../logs/postgres'
    db_dir = 'summarized'
    log_dir = os.path.join(base_dir, db_dir)

    # 033e zu simpel, nur 5 Keys
    # 0102 hat 5 keys zu häufig vorkommen, und 5-10 weitere die jeweils max 5 mal vorkommen
    # 004 hat 5 Keys
    # 029e hat 5 Keys

    log_file = 'postgresql0102.log'
    multiline_path = os.path.join(log_dir, log_file)

    log_file_path = os.path.join(base_dir, log_file)

    test_log_file_path = os.path.join(base_dir, 'postgres_test.log')
    train_log_file_path = os.path.join(base_dir, 'postgres_train.log')
    validation_log_file_path = os.path.join(base_dir, 'postgres_validation.log')
    labels_file_path = os.path.join(base_dir, 'anomaly_label.csv')
    time_labels_file_path = os.path.join(base_dir, 'anomaly_label_time.csv')

    # 90% der Log-Einträge als Testdaten
    test_size = 0.8

    # 20% der Trainingsdaten als Validierungsdaten
    validation_size = 0.2


    multiline_to_singleline(multiline_path, log_file_path)
    log_df = read_and_parse_log(log_file_path)
    print(log_df)


    filtered_log_df, removed_pids = filter_groups(log_df)
    print(filtered_log_df)
    unique_pids = filtered_log_df['pid'].unique()
    print(unique_pids)
    # train_pids, test_pids = train_test_split(unique_pids, test_size=test_size, random_state=42)
    train_size = 1 - test_size

    # Aufteilung der Daten in Trainings- und Testdatensatz basierend auf dem berechneten train_size
    train_pids = unique_pids[:int(len(unique_pids) * train_size)]
    test_pids = unique_pids[-int(len(unique_pids) * test_size):]

    train_df = filtered_log_df[filtered_log_df['pid'].isin(train_pids)].copy()
    test_df = filtered_log_df[filtered_log_df['pid'].isin(test_pids)].copy()


    template_list = [
        [20,
         ["[unknown]@[unknown] LOG:  connection received: host={} port={}", "mydbuser@linuxbox FATAL:  no pg_hba.conf entry for host \"{}\", user \"mydbuser\", database \"mydbuser\", SSL on"],[generate_random_ip, generate_random_port], [1]],
        [30, ["FATAL: could not connect to the primary server: could not connect to server: Connection refused | Is the server running on host \"{}\" and accepting | TCP/IP connections on port 5432?"],[generate_random_ip], [0]],  # [0] zeigt an, dass die erste Nachricht anomal ist
        [30, ["[unknown]@[unknown] LOG:  connection received: host=[local]", "postgres@[unknown] LOG:  replication connection authorized: user=postgres application_name=backup", "postgres@[unknown] WARNING:  skipping special file \"./log\"", "postgres@[unknown] LOG:  disconnection: session time: 0:00:05.371 user=postgres database= host=[local]"],[generate_random_session_time], [2]],
    ]

    test_df['datetime'] = pd.to_datetime(test_df['date'] + ' ' + test_df['time'])

    artificial_logs_df, artificial_logs_timestamp_df = generate_artificial_logs(template_list, test_df)

    # Markieren aller existierenden Einträge aus test_df als 'Normal'
    existing_logs_df = test_df[['datetime']].copy()
    existing_logs_df['Label'] = 'Normal'
    combined_logs_timestamp_df = pd.concat([existing_logs_df, artificial_logs_timestamp_df]).sort_values(
        by=['datetime']).reset_index(drop=True)


    test_df_combined = pd.concat([test_df, artificial_logs_df]).sort_values(by=['datetime']).reset_index(drop=True)
    test_df_combined['datetime'] = pd.to_datetime(test_df_combined['date'] + ' ' + test_df_combined['time'])
    test_df_combined = test_df_combined.sort_values(by=['datetime'])


    # Alle PIDs aus dem ursprünglichen, bereinigten,  Datensatz sind normal
    original_pids = test_df['pid'].unique()

    combined_pids = test_df_combined['pid'].unique()
    artificial_pids = [pid for pid in combined_pids if pid not in original_pids]

    labels_df = pd.DataFrame({
        'pid': combined_pids,
        'Label': ['Anomaly' if pid in artificial_pids else 'Normal' for pid in combined_pids]
    })



    labels_df.to_csv(labels_file_path, index=False)
    write_log_entries(test_df_combined, test_log_file_path)

    # train_df, validation_df = train_test_split(train_df, test_size=validation_size, random_state=42)
    train_size = 1 - validation_size
    train_df = train_df[:int(len(train_df) * train_size)]
    validation_df = train_df[int(len(train_df) * train_size):]
    write_log_entries(train_df, train_log_file_path)
    write_log_entries(validation_df, validation_log_file_path)

    combined_logs_timestamp_df.to_csv(time_labels_file_path, index=False)