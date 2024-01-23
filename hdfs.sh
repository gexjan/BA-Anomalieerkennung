#!/bin/bash

python main.py -train --log-dir ./logs/HDFS/ --log-file hdfs_200k.log --dataset hdfs --window-type id --grouping sliding --batch-size 2048 --input-size 1 --num-layers 2 --hidden-size 64 -predict --validation-file hdfs_20k.log --anomaly-file anomaly_label.csv --epochs 50
