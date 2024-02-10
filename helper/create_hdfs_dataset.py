from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split


# Variablen für die Anzahl der gewünschten Sequenzen
anzahl_normale_train = 4855
anzahl_anomale_verwerfen = 1638  # Diese werden für das Training verworfen
anzahl_normale_validation = 1500

log_dir = '../logs/HDFS/'
log_file = 'HDFS.log'
label_file = 'anomaly_label.csv'
data_dir = '../data/'

log_file_path = os.path.join(log_dir, log_file)
train_file_path = os.path.join(log_dir, 'hdfs_train.log')
test_file_path = os.path.join(log_dir, 'hdfs_test.log')
validation_file_path = os.path.join(log_dir, 'hdfs_validation.log')

# Lese die Anomalie-Labels
labels_df = pd.read_csv(os.path.join(log_dir, label_file))

# Gruppierung der Block-IDs nach ihrem Label
normale_block_ids = labels_df[labels_df['Label'] == 'Normal']['BlockId'].tolist()
anomale_block_ids = labels_df[labels_df['Label'] == 'Anomaly']['BlockId'].tolist()
print(len(normale_block_ids))
print(len(anomale_block_ids))

normale_block_ids, normale_block_train_ids = train_test_split(normale_block_ids, test_size=anzahl_normale_train, random_state=42)

# normale_block_ids, normale_block_validation_ids = train_test_split(normale_block_ids, test_size=anzahl_normale_validation, random_state=42)

anomaly_block_ids, entfernte_anomaly_ids = train_test_split(anomale_block_ids, test_size=anzahl_anomale_verwerfen, random_state=42)


test_ids = normale_block_ids + anomaly_block_ids
test_ids, anzahl_normale_validation = train_test_split(test_ids, test_size=anzahl_normale_validation)

block_id_regex = re.compile(r'(blk_-?\d+)')

# Speichern der Zeilen, zugeordnet nach Block-ID
block_id_groups = {}

# Lesen der Log-Datei und Gruppierung der Zeilen nach Block-ID
with open(log_file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    match = block_id_regex.search(line)
    if match:
        block_id = match.group(1)
        if block_id not in block_id_groups:
            block_id_groups[block_id] = []
        block_id_groups[block_id].append(line)

# Markieren der Zeilen für Training, Test oder Validierung
train_lines = [line for block_id in normale_block_train_ids for line in block_id_groups.get(block_id, [])]
print("Train_lines: ", len(train_lines))
validation_lines = [line for block_id in normale_block_validation_ids for line in block_id_groups.get(block_id, [])]
print("Validation-lines: ", len(validation_lines))
test_lines = [line for block_id in test_ids for line in block_id_groups.get(block_id, [])]
print("test_lines: ", len(test_lines))

# Schreiben der Zeilen in die entsprechenden Dateien
with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file, open(validation_file_path, 'w') as validation_file:
    for line in train_lines:
        train_file.write(line)
    for line in validation_lines:
        validation_file.write(line)
    for line in test_lines:
        test_file.write(line)

print("Trainingsdaten, Validierungsdaten und Testdaten wurden erfolgreich vorbereitet.")
