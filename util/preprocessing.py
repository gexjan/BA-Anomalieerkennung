from logparser.Spell import LogParser as SpellParser
from logparser.Drain import LogParser as DrainParser
import os
import re
import pandas as pd

class Parser:
    def __init__(self, indir, outdir, log_format, rex, parser_type, tau, st, depth, logger):
        self.indir = indir
        self.outdir = outdir
        self.log_format = log_format
        self.tau = tau
        self.rex = rex
        self.parser_type = parser_type
        self.st = st
        self.depth = depth
        self.logger = logger

    def parse(self, log_file):
        self.logger.info("Parsing log file")
        if self.parser_type == 'spell':
            parser = SpellParser(indir=self.indir, outdir=self.outdir, log_format=self.log_format, tau=self.tau, rex=self.rex)
        elif self.parser_type == 'drain':
            parser = DrainParser(indir=self.indir, outdir=self.outdir, log_format=self.log_format, st=self.st, depth=self.depth, rex=self.rex)
        parser.parse(log_file)
        self.logger.info("Log file parsed")

class PostgresLogParser(Parser):
    def __init__(self, indir, outdir, parser_type, logger):
        log_format = '<Date> <Time> <Timeformat> <PID> <Content>'
        tau = 0.5  # Angenommener Wert für Postgres
        st = 0.5
        depth = 4
        rex = []
        super().__init__(indir, outdir, log_format, rex, parser_type, tau, st, depth, logger)

class HDFSLogParser(Parser):
    def __init__(self, indir, outdir, parser_type, logger):
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        tau = 0.5  # Angenommener Wert für HDFS
        st = 0.5
        depth = 4
        rex = []
        super().__init__(indir, outdir, log_format, rex, parser_type, tau, st, depth, logger)




def postgres_to_singleline(log_dir, log_file, data_dir, logger):
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, 'r') as file:
        lines = file.readlines()

    output_path = os.path.join(data_dir, log_file)

    with open(output_path, 'w') as output_file:
        current_entry = ""

        for line in lines:
            # Überprüfen, ob die Zeile mit einem Zeitstempel beginnt
            if line[:4].isdigit() and line[4] == '-':
                if current_entry:
                    output_file.write(current_entry.strip() + '\n')
                current_entry = line.strip()
            else:
                # Füge die Zeile dem aktuellen Eintrag hinzu, getrennt durch ein Pipe-Symbol
                current_entry += ' | ' + line.strip()

        # Füge den letzten Eintrag hinzu
        if current_entry:
            output_file.write(current_entry.strip() + '\n')
    logger.info(f"Created file in {output_path}")


def group_hdfs(df, anomaly_df, window_type):
    if window_type == 'id':
        block_id_pattern = re.compile(r'(blk_-?\d+)')

        # Hinzufügen einer neuen Spalte für die extrahierte Block-ID
        df['BlockID'] = df['Content'].apply(lambda x: block_id_pattern.search(x).group(0) if block_id_pattern.search(x) else None)

        # Gruppierung der Daten nach Block-ID und Sammeln der zugehörigen EventIDs
        grouped = df.groupby('BlockID')['EventId'].apply(list)

        grouped = grouped.reset_index()
        grouped.columns = ['BlockID', 'EventSequence']
        
        # Zusammenführen der DataFrames anhand der Block-ID
        merged_df = pd.merge(grouped, anomaly_df, left_on='BlockID', right_on='BlockId', how='left')
        merged_df.drop(columns=['BlockId'], inplace=True)

        # Entfernen der anomalen Zeilen
        normal_df = merged_df[merged_df['Label'] == 'Normal']
        return normal_df

    elif window_type == 'time':
        df['Date'] = df['Date'].astype(str)
        df['Time'] = df['Time'].astype(str)

        df['DateTime'] = pd.to_datetime(df['Date'] + df['Time'], format='%y%m%d%H%M%S')

        df['Minute'] = df['DateTime'].dt.floor('T')

        # Gruppieren nach Minute und Sammeln der EventIDs
        grouped = df.groupby('Minute')['EventId'].apply(list)

        grouped.reset_index()
        grouped.columns = ['Minute', 'EventSequence']

        return grouped
    

def slice_hdfs(df, grouping_type, window_size):
    if grouping_type == 'session':
        print(df[:10])
    elif grouping_type == 'sliding':
        windows = []
        print(df[:20].to_string())
        for index, row in df.iterrows():
            sequence = row['EventSequence']
            seqlen = len(sequence)
            i = 0
            while (i + window_size) < seqlen:
                window_slice = sequence[i : i + window_size]
                next_element = sequence[i + window_size]
                i += 1
                windows.append([index, window_slice, next_element])
            else:
                window_slice = sequence[i : i + window_size]
                window_slice += ['#PAD'] * (window_size - len(window_slice))
                next_element = '#PAD'
                windows.append([index, window_slice, next_element])

        sliced_windows = pd.DataFrame(windows, columns=['id', 'window', 'next'])
        train_x = sliced_windows[['id', 'window']]
        train_y = sliced_windows[['id', 'next']]
        print(train_x[:20].to_string())
        print(train_y[:20].to_string())
    return train_x, train_y

class Vectorizer:
    def fit_transform(self, x, y):
        self.label_mapping = {'#OOV': 0, '#PAD': 1}
        next_id_value = 2

        # Extrahieren der einzigartigen IDs aus allen Fenstern
        for index, row in x.iterrows():
            for event_id in row['window']:
                if event_id not in self.label_mapping and event_id != '#PAD':
                    self.label_mapping[event_id] = next_id_value
                    next_id_value += 1

        return self.transform(x, y)

    def transform(self, x, y):
        x_transformed = x.copy()
        for index, row in x.iterrows():
            x_transformed.at[index, 'window'] = [self.label_mapping.get(event_id, self.label_mapping['#OOV']) for event_id in row['window']]

        y_transformed = y.copy()
        y_transformed['next'] = y['next'].apply(lambda event_id: self.label_mapping.get(event_id, self.label_mapping['#OOV']))

        return x_transformed, y_transformed