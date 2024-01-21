#!/bin/bash
python main.py -train --log-dir ./logs/HDFS/ --log-file hdfs_20k.log --dataset hdfs --window-type id --grouping sliding --batch-size 64 --input-size 1 --num-layers 2 --hidden-size 64

