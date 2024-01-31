from logparser.Spell import LogParser as SpellParser
from logparser.Drain import LogParser as DrainParser

from util.preprocessing import postgres_to_singleline
import os
import pandas as pd


class DataHandler:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def get_parse_params(self):
        # Diese Methode sollte in den Subklassen überschrieben werden
        raise NotImplementedError

    def parse(self):
        # Diese Methode sollte in den Subklassen überschrieben werden
        raise NotImplementedError

    def get_parser(self, log_format, rex, tau, st, depth, log_dir):
        if self.args.parser_type == 'spell':
            return SpellParser(indir=log_dir, outdir=self.args.data_dir, log_format=log_format, tau=tau,
                               rex=rex)
        elif self.args.parser_type == 'drain':
            return DrainParser(indir=log_dir, outdir=self.args.data_dir, log_format=log_format, st=st,
                               depth=depth, rex=rex)
        else:
            raise ValueError("Unbekannter Parser-Typ")

    @staticmethod
    def create(args, logger):
        if args.dataset == 'hdfs':
            return HDFSDataHandler(args, logger)
        elif args.dataset == 'postgres':
            return PostgresDataHandler(args, logger)
        else:
            raise ValueError("Unbekannter Datensatz")

    def read_structured_files(self):
        self.logger.info("Reading structured files")
        structured_train_file_path = os.path.join(self.args.data_dir, self.args.log_file + '_structured.csv')
        structured_eval_file_path = os.path.join(self.args.data_dir, self.args.evaluation_file + '_structured.csv')

        self.struct_train_df = pd.read_csv(structured_train_file_path)
        self.struct_eval_df = pd.read_csv(structured_eval_file_path)

    def read_anomaly_file(self):
        anomaly_file_path = os.path.join(self.args.log_dir, self.args.anomaly_file)
        return pd.read_csv(anomaly_file_path)

    def get_structured_data(self, data_type):
        return getattr(self, f"struct_{data_type}_df")

    def set_grouped_data(self, grouped_data, data_type):
        setattr(self, f"df_grouped_{data_type}", grouped_data)

    def get_grouped_data(self, data_type):
        return getattr(self, f"df_grouped_{data_type}")

    def set_sliced_windows(self, sliced_data, data_type):
        data_dict = {'x': sliced_data[0], 'y': sliced_data[1]}
        setattr(self, f"df_sliced_{data_type}", data_dict)

    def get_sliced_windows(self, data_type):
        data_dict = getattr(self, f"df_sliced_{data_type}", None)
        if data_dict is not None:
            return data_dict
        return None

    def set_label_mapping(self, label_mapping):
        self.label_mapping = label_mapping

    def get_label_mapping(self):
        return self.label_mapping

    def set_transformed_windows(self, data, var_name, data_type):
        return setattr(self, f"df_transformed_{data_type}_{var_name}", data)

    def get_transformed_windows(self, var_name, data_type):
        return getattr(self, f"df_transformed_{data_type}_{var_name}")



class HDFSDataHandler(DataHandler):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        # Spezifische Initialisierung für HDFSDataHandler

    def get_parse_params(self):
        log_format = '<Date> <Time> <Component> <Content>'
        rex = [r"blk_-?\d+",
               r"(\d+\.){3}\d+(:\d+)?"]
        tau = 0.7  # Spell
        st = 0.5  # Drain
        depth = 4  # Drain
        return log_format, rex, tau, st, depth

    def parse(self):
        files = [self.args.log_file, self.args.evaluation_file]

        log_format, rex, tau, st, depth = self.get_parse_params()
        parser = self.get_parser(log_format, rex, tau, st, depth, self.args.log_dir)
        for file in files:
            self.logger.info(f"Parsing file {file}")
            parser.parse(file)


class PostgresDataHandler(DataHandler):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        # Spezifische Initialisierung für PostgresDataHandler

    def get_parse_params(self):
        log_format = '<Date> <Time> <Timeformat> <PID> <Content>'
        rex = []
        tau = 0.5  # Spell
        st = 0.5  # Drain
        depth = 4  # Drain
        return log_format, rex, tau, st, depth

    def parse(self):
        self.logger.info("Converting Postgres-Logs zu singleline")
        files = [self.args.log_file, self.args.evaluation_file]

        # Umwandeln in einzeilige Einträge
        postgres_to_singleline(files, self.args.log_dir, self.args.data_dir)

        log_format, rex, tau, st, depth = self.get_parse_params()

        # Hier stammen die Log-Dateien nicht aus dem Log-Dir, sondern Data-dir
        parser = self.get_parser(log_format, rex, tau, st, depth, self.args.data_dir)

        for file in files:
            self.logger.info(f"Parsing file {file}")
            parser.parse(file)
