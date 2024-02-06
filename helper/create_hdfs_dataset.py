from sklearn.model_selection import train_test_split
import sys
# setting path
sys.path.append('../')
import os
import re

# Parameter für den Anteil der Testdaten, die verwendet werden sollen
test_data_usage_percentage =  0.01 # z.B. 1.0 für 100% der Testdaten
train_data_usage_percentage =  0.7 # z.B. 1.0 für 100% der Trainingsdaten
validation_data_percentage = 1.0  # z.B. 0.2 für 20% der Trainingsdaten als Validierungsdaten


log_dir = '../logs/HDFS/'
log_file = 'HDFS.log'
data_dir = '../data/'

log_file_path = os.path.join(log_dir, log_file)
train_file_path = os.path.join(log_dir, 'hdfs_train.log')
test_file_path = os.path.join(log_dir, 'hdfs_test.log')
validation_file_path = os.path.join(log_dir, 'hdfs_validation.log')

block_id_regex = re.compile(r'(blk_-?\d+)')

# Speichern der Zeilen, zugeordnet nach Block-ID
block_id_groups = {}

# Lesen der Log-Datei und Gruppierung der Zeilen nach Block-ID
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Extrahieren der Block-IDs und Zuordnen der Zeilen
for line in lines:
    match = block_id_regex.search(line)
    if match:
        block_id = match.group(1)
        if block_id not in block_id_groups:
            block_id_groups[block_id] = []
        block_id_groups[block_id].append(line)

# Aufteilen der Block-IDs in Trainings- und Testdaten
block_ids = list(block_id_groups.keys())
train_ids, test_ids = train_test_split(block_ids, test_size=0.98, random_state=42)

train_ids, validation_ids = train_test_split(train_ids, test_size=0.5, random_state=42)

# Auswahl eines Prozentsatzes der Test-IDs, falls notwendig
if test_data_usage_percentage < 1.0:
    test_ids = test_ids[:int(len(test_ids) * test_data_usage_percentage)]

# Auswahl eines Prozentsatzes der Trainings-IDs, falls notwendig
if train_data_usage_percentage < 1.0:
    train_ids = train_ids[:int(len(train_ids) * train_data_usage_percentage)]

if validation_data_percentage < 1.0:
    validation_ids = validation_ids[:int(len(validation_ids) * validation_data_percentage)]

# Markieren der Zeilen für Training oder Test
train_lines = set()
test_lines = set()
validation_lines = set()

for block_id in train_ids:
    train_lines.update(block_id_groups[block_id])

for block_id in test_ids:
    test_lines.update(block_id_groups[block_id])

for block_id in validation_ids:
    validation_lines.update(block_id_groups[block_id])

# Schreiben der Zeilen in die entsprechenden Dateien unter Beibehaltung der Reihenfolge
with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file, open(validation_file_path, 'w') as validation_file:
    for line in lines:
        if line in train_lines:
            train_file.write(line)
        elif line in test_lines:
            test_file.write(line)
        elif line in validation_lines:
            validation_file.write(line)

print("Trainingsdaten und Testdaten wurden erfolgreich vorbereitet.")
