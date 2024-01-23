#!/bin/bash

python main.py -predict --log-dir ./logs/HDFS --validation-file hdfs_validation_200k.log --anomaly-file anomaly_label.csv --dataset hdfs

# -prepare -train --log-dir ./logs/HDFS/ --log-file hdfs_20k.log --dataset hdfs --window-type id --grouping sliding --batch-size 256 --input-size 1 --num-layers 2 --hidden-size 128 --epochs 20
