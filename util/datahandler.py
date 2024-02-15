from logparser.Spell import LogParser as SpellParser
from logparser.Drain import LogParser as DrainParser

from util.preprocessing import postgres_to_singleline, slice_windows
import os
import pandas as pd


class DataHandler:
    def __init__(self, args, logger, window_size):
        self.args = args
        self.logger = logger
        self.window_size = window_size

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
    def create(args, logger, window_size):
        if args.dataset == 'hdfs':
            return HDFSDataHandler(args, logger, window_size)
        elif args.dataset == 'postgres':
            return PostgresDataHandler(args, logger, window_size)
        else:
            raise ValueError("Unbekannter Datensatz")

    def read_structured_files(self):
        self.logger.info("Reading structured files")
        structured_train_file_path = os.path.join(self.args.data_dir, self.args.log_file + '_structured.csv')
        structured_validation_file_path = os.path.join(self.args.data_dir, self.args.validation_file + '_structured.csv')
        structured_eval_file_path = os.path.join(self.args.data_dir, self.args.evaluation_file + '_structured.csv')

        self.struct_train_df = pd.read_csv(structured_train_file_path)
        self.struct_validation_df = pd.read_csv(structured_validation_file_path)
        self.struct_eval_df = pd.read_csv(structured_eval_file_path)

    def del_structured_files(self):
        del self.struct_train_df
        del self.struct_validation_df
        del self.struct_eval_df

    def read_anomaly_file(self):
        if self.args.grouping == 'time':
            file_name = self.args.anomaly_file.split('.')[0]
            file_name += '_time.csv'
            anomaly_file_path = os.path.join(self.args.log_dir, file_name)
        else:
            anomaly_file_path = os.path.join(self.args.log_dir, self.args.anomaly_file)
        return pd.read_csv(anomaly_file_path)

    def get_structured_data(self, data_type):
        return getattr(self, f"struct_{data_type}_df")

    def set_grouped_data(self, grouped_data, data_type):
        setattr(self, f"df_grouped_{data_type}", grouped_data)

    def get_grouped_data(self, data_type):
        return getattr(self, f"df_grouped_{data_type}")

    def set_transformed_data(self, df, data_type):
        setattr(self, f"df_transformed_{data_type}", df)

    def get_transformed_data(self, data_type):
        return getattr(self, f"df_transformed_{data_type}")

    def set_prepared_data(self, x, y, data_type):
        setattr(self, f"df_prepared_x_{data_type}", x)
        setattr(self, f"df_prepared_y_{data_type}", y)

    def get_prepared_data(self, data_type):
        return getattr(self, f"df_prepared_x_{data_type}"), getattr(self, f"df_prepared_y_{data_type}")

    def set_label_mapping(self, label_mapping):
        self.label_mapping = label_mapping

    def get_label_mapping(self):
        return self.label_mapping

    def update_window_size(self, window_size, logger):
        print(window_size)
        x_train, y_train = slice_windows(self.get_transformed_data('train'), window_size, logger, False)
        x_eval, y_eval = slice_windows(self.get_transformed_data('eval'), window_size, logger, True)

        self.set_prepared_data(x_train, y_train, 'train')
        self.set_prepared_data(x_eval, y_eval, 'eval')

class HDFSDataHandler(DataHandler):
    def __init__(self, args, logger, window_size):
        super().__init__(args, logger, window_size)
        # Spezifische Initialisierung für HDFSDataHandler

    def get_parse_params(self):
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        rex = [r"blk_-?\d+",
               r"(\d+\.){3}\d+(:\d+)?"]
        tau = 0.7  # Spell
        st = 0.5  # Drain
        depth = 4  # Drain
        return log_format, rex, tau, st, depth

    def parse(self):
        files = [self.args.log_file, self.args.evaluation_file, self.args.validation_file]

        log_format, rex, tau, st, depth = self.get_parse_params()
        parser = self.get_parser(log_format, rex, tau, st, depth, self.args.log_dir)
        for file in files:
            self.logger.info(f"Parsing file {file}")
            parser.parse(file)


class PostgresDataHandler(DataHandler):
    def __init__(self, args, logger, window_size):
        super().__init__(args, logger, window_size)
        # Spezifische Initialisierung für PostgresDataHandler

    def get_parse_params(self):
        log_format = '<Date> <Time> <Timeformat> <PID> <Content>'
        rex = [

            # "DETAIL:  parameters:\s*(.*)",  # SQL-Parameters
            # "statement:\s*(.*)",  # SQL-Statements,

            # "LOG:  execute\s*(.*)",  # SQL-Statements,
            # "STATEMENT: \s*(.*)",  # SQL-Statements,
            # "CONTEXT: \s*(.*)",  # SQL-Statements,
            "LOG:  duration:\s*(.*)",  # SQL-Statements,
            "\[?\w+\]?\@\[?\w+\]?",  # Benutzer@Datenbank
            #

            "DETAIL:  \s*(.*)",  # SQL-Statements,
            # "duration:\s*(.*)" # SQL-Query duration
            # "parameters:\s*(.*)" # SQL-Query parameters
            # "LOG:\s*execute\s*(.*)", # Ausgeführte SQL-Funktionen

            # "(?<=user=)[^\s]*", # user=xxx
            # "(?<=database=)[^\s]*", # database=xxx
            # "(?<=automatic vacuum of table\s).*", # Automatic vacuum of table,
            # "\d{4}-\d{2}-\d{2}" # Datum im Format yyyy-mm-dd
               ]
        tau = 0.6  # Spell
        st = 0.5  # Drain
        depth = 4  # Drain
        return log_format, rex, tau, st, depth

    def parse(self):
        self.logger.info("Converting Postgres-Logs zu singleline")
        files = [self.args.log_file, self.args.evaluation_file, self.args.validation_file]

        # Umwandeln in einzeilige Einträge
        postgres_to_singleline(files, self.args.log_dir, self.args.data_dir)

        log_format, rex, tau, st, depth = self.get_parse_params()

        # Hier stammen die Log-Dateien nicht aus dem Log-Dir, sondern Data-dir
        parser = self.get_parser(log_format, rex, tau, st, depth, self.args.data_dir)

        for file in files:
            self.logger.info(f"Parsing file {file}")
            parser.parse(file)
