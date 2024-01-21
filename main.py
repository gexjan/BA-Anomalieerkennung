import argparse
from util import dataloader, preprocessing
import os
import logging
import sys
import pandas as pd


logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='postgres', choices=['hdfs', 'postgres'] , help='Choose the Dataset')
    parser.add_argument('--model', type=str, default='deeplog', choices=['deeplog', 'autoencoder'], help='Choose the model')
    parser.add_argument('--model-dir', type=str, default='./model/', help='The place where to store the model parameter')
    parser.add_argument('--data-dir', type=str, default='./data/', help='The place where to store the data')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='The folder with the log-files')
    parser.add_argument('--log-file', type=str, default='postgresql-01.log', help='The log used for training')
    parser.add_argument('-prepare', action='store_true', help='Pre-Process the Logs')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('-predict', action='store_true', help='Detect anomalies')
    parser.add_argument('--parser-type', type=str, default='spell', choices=['spell', 'drain'] , help='Choose the parser')
    parser.add_argument('--window-size', type=int, default='5', help='Size of the windows')
    parser.add_argument('--window-type', type=str, default='id', choices=['id','time'], help="Build windows by id or time (window_size)")
    parser.add_argument('--grouping', type=str, default='sliding', choices=['sliding', 'session'], help='Group entries by sliding window or session')


    args = parser.parse_args()

    if args.prepare:
        log_path = dataloader.load_log(args.log_dir, args.log_file)

        if args.dataset == 'hdfs':
            logparser = preprocessing.HDFSLogParser(args.log_dir, args.data_dir, args.parser_type, logger)
            logparser.parse(args.log_file)

        elif args.dataset == 'postgres':
            logger.info(f"Converting Log-File {args.log_file} to singleline")
            preprocessing.postgres_to_singleline(args.log_dir, args.log_file, args.data_dir, logger)

            # logger.info("Parsing Log-File")
            logparser = preprocessing.PostgresLogParser(args.data_dir, args.data_dir, args.parser_type, logger)
            logparser.parse(args.log_file)

        structured_file = os.path.join(args.data_dir, args.log_file + '_structured.csv')
        template_file = os.path.join(args.data_dir, args.log_file + '_templates.csv')

        structured_df = pd.read_csv(structured_file, dtype={'Date': str, 'Time': str})
        template_df = pd.read_csv(template_file)


        feature_extractor = preprocessing.Vectorizer()
        if args.dataset == 'hdfs':
            anomaly_file_path = os.path.join(args.log_dir, 'anomaly_label.csv')
            anomaly_df = pd.read_csv(anomaly_file_path)
            grouped_hdfs = preprocessing.group_hdfs(structured_df, anomaly_df, args.window_type)
            # print("Grouped: ", grouped_hdfs[:10].to_string())
            # print(grouped_hdfs.columns)
            train_x, train_y = preprocessing.slice_hdfs(grouped_hdfs, args.grouping, args.window_size)

            train_x_transformed, train_y_transformed = feature_extractor.fit_transform(train_x, train_y)
            print(train_x_transformed[:20].to_string())
            print(train_y_transformed[:20].to_string())
        elif args.dataset == 'postgres':
            pass

