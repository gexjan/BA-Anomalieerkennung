#!/bin/bash
python main.py -prepare -train --log-dir ./logs/HDFS/ --log-file hdfs_20k.log --dataset hdfs --window-type id --grouping sliding --batch-size 256 --input-size 1 --num-layers 2 --hidden-size 128 --epochs 20
