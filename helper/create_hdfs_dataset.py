from sklearn.model_selection import train_test_split
import sys
# setting path
sys.path.append('../')
import os
import re


log_dir = '../logs/HDFS/'
log_file = 'HDFS.log'
data_dir = '../data/'
anomaly_file = 'anomaly_label.csv'

log_file_path = os.path.join(log_dir, log_file)
train_file_path = os.path.join(log_dir, 'hdfs_train.log')
test_file_path = os.path.join(log_dir, 'hdfs_test.log')

block_id_regex = re.compile(r'(blk_-?\d+)')

block_id_groups = {}
with open(log_file_path, 'r') as file:
    for line in file:
        match = block_id_regex.search(line)
        if match:
            block_id = match.group(1)
            if block_id not in block_id_groups:
                block_id_groups[block_id] = []
            block_id_groups[block_id].append(line)

block_ids = list(block_id_groups.keys())
block_group_data = [block_id_groups[id] for id in block_ids]


train_data, test_data = train_test_split(block_group_data, test_size=0.99, random_state=42)

max_length_train = max(len(group) for group in train_data)
max_length_test = max(len(group) for group in test_data)

print("Maximale Länge der Gruppen in train_data:", max_length_train)
print("Maximale Länge der Gruppen in test_data:", max_length_test)


with open(train_file_path, 'w') as train_file:
    for group in train_data:
        for line in group:
            train_file.write(line)

with open(test_file_path, 'w') as test_file:
    for group in test_data:
        for line in group:
            test_file.write(line)