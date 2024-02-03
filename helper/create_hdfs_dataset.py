from sklearn.model_selection import train_test_split
import sys
# setting path
sys.path.append('../')
import os
import re

# Parameter für den Anteil der Testdaten, die verwendet werden sollen
test_data_usage_percentage =  0.01 # z.B. 1.0 für 100% der Testdaten
train_data_usage_percentage =  0.05 # z.B. 1.0 für 100% der Trainingsdaten

log_dir = '../logs/HDFS/'
log_file = 'HDFS.log'
data_dir = '../data/'

log_file_path = os.path.join(log_dir, log_file)
train_file_path = os.path.join(log_dir, 'hdfs_train.log')
test_file_path = os.path.join(log_dir, 'hdfs_test.log')

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
train_ids, test_ids = train_test_split(block_ids, test_size=0.99, random_state=42)

# Auswahl eines Prozentsatzes der Test-IDs, falls notwendig
if test_data_usage_percentage < 1.0:
    test_ids = test_ids[:int(len(test_ids) * test_data_usage_percentage)]

# Auswahl eines Prozentsatzes der Trainings-IDs, falls notwendig
if train_data_usage_percentage < 1.0:
    train_ids = train_ids[:int(len(train_ids) * train_data_usage_percentage)]

# Markieren der Zeilen für Training oder Test
train_lines = set()
test_lines = set()

for block_id in train_ids:
    train_lines.update(block_id_groups[block_id])

for block_id in test_ids:
    test_lines.update(block_id_groups[block_id])

# Schreiben der Zeilen in die entsprechenden Dateien unter Beibehaltung der Reihenfolge
with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file:
    for line in lines:
        if line in train_lines:
            train_file.write(line)
        elif line in test_lines:
            test_file.write(line)

print("Trainingsdaten und Testdaten wurden erfolgreich vorbereitet.")


# from sklearn.model_selection import train_test_split
# import sys
# # setting path
# sys.path.append('../')
# import os
# import re
#
#
# log_dir = '../logs/HDFS/'
# log_file = 'HDFS.log'
# data_dir = '../data/'
# anomaly_file = 'anomaly_label.csv'
#
# log_file_path = os.path.join(log_dir, log_file)
# train_file_path = os.path.join(log_dir, 'hdfs_train.log')
# test_file_path = os.path.join(log_dir, 'hdfs_test.log')
#
# block_id_regex = re.compile(r'(blk_-?\d+)')
#
# block_id_groups = {}
# with open(log_file_path, 'r') as file:
#     for line in file:
#         match = block_id_regex.search(line)
#         if match:
#             block_id = match.group(1)
#             if block_id not in block_id_groups:
#                 block_id_groups[block_id] = []
#             block_id_groups[block_id].append(line)
#
# block_ids = list(block_id_groups.keys())
# block_group_data = [block_id_groups[id] for id in block_ids]
#
#
# train_data, test_data = train_test_split(block_group_data, test_size=0.99, random_state=42)
#
# max_length_train = max(len(group) for group in train_data)
# max_length_test = max(len(group) for group in test_data)
#
# print("Maximale Länge der Gruppen in train_data:", max_length_train)
# print("Maximale Länge der Gruppen in test_data:", max_length_test)
#
#
# with open(train_file_path, 'w') as train_file:
#     for group in train_data:
#         for line in group:
#             train_file.write(line)
#
# with open(test_file_path, 'w') as test_file:
#     for group in test_data:
#         for line in group:
#             test_file.write(line)
