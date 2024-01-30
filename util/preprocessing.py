from logparser.Spell import LogParser as SpellParser
from logparser.Drain import LogParser as DrainParser
import os
import re
import pandas as pd
from multiprocessing import Pool


# Basisklasse für das Parsen von Log-Dateien
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

    # Parsen der übergebenen Log-Datei
    # Das Ergebnis ist eine Ausgabe im Ordner data mit _structured.csv und _template.csv
    # Nutzt entweder den Spell- oder Drain-Parser, abhängig vom 'parser_type'.
    def parse(self, log_file):
        self.logger.info("Parsing log file")
        if self.parser_type == 'spell':
            parser = SpellParser(indir=self.indir, outdir=self.outdir, log_format=self.log_format, tau=self.tau,
                                 rex=self.rex)
        elif self.parser_type == 'drain':
            parser = DrainParser(indir=self.indir, outdir=self.outdir, log_format=self.log_format, st=self.st,
                                 depth=self.depth, rex=self.rex)
        parser.parse(log_file)
        self.logger.info("Log file parsed")


class PostgresLogParser(Parser):
    def __init__(self, indir, outdir, parser_type, logger):
        log_format = '<Date> <Time> <Timeformat> <PID> <Content>'

        # Parameter für Spell

        tau = 0.5

        # Parameter für Drain
        st = 0.5
        depth = 4

        # Reguläre Ausdrücke für dynamische Log-Teile
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


# Methode zum Umwandeln mehrzeiliger Postgres-Logeinträge in einzeilige Einträge
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


# Nachfolgende Methode gruppiert alle Log-Einträge mit derselben Block-ID.
# Das ist notwendig, da Log-Einträge, die zur selben Block-ID gehören, typischerweise
# zusammenhängende Ereignisse darstellen. Die Gruppierung ermöglicht eine
# effektivere Analyse dieser zusammenhängenden Ereignisse und hilft bei der Identifizierung
# von Mustern oder Anomalien, die innerhalb einer bestimmten Block-ID auftreten.
def group_hdfs(df, anomaly_df, logger, remove_anomalies=True):
    logger.info("Grouping Log Files")
    block_id_pattern = re.compile(r'(blk_-?\d+)')

    # Hinzufügen einer neuen Spalte für die extrahierte Block-ID
    df['BlockID'] = df['Content'].apply(
        lambda x: block_id_pattern.search(x).group(0) if block_id_pattern.search(x) else None)

    # Gruppierung der Daten nach Block-ID und Sammeln der zugehörigen EventIDs
    grouped = df.groupby('BlockID')['EventId'].apply(list)

    grouped = grouped.reset_index()
    grouped.columns = ['BlockID', 'EventSequence']

    # Zusammenführen der DataFrames anhand der Block-ID
    merged_df = pd.merge(grouped, anomaly_df, left_on='BlockID', right_on='BlockId', how='left')
    merged_df.drop(columns=['BlockId'], inplace=True)

    # Entfernen der anomalen Zeilen
    # Beim Training ist das notwendig, um dem Modell das "normale" Verhalten beizubringen
    if remove_anomalies:
        merged_df = merged_df[merged_df['Label'] == 'Normal']
    return merged_df


# Nachfolgende Methode 'slice_hdfs' verwendet den 'Sliding Window'-Ansatz für die Vorverarbeitung von Sequenzdaten für LSTM-Modelle.
# Der 'Sliding Window'-Ansatz ist wichtig, da er hilft, zeitliche Abhängigkeiten in den Daten zu erfassen. Er bietet dem Modell eine Sequenz
# von vorherigen Ereignissen (Inputs), was ihm ermöglicht, zeitliche Muster und Dynamiken besser zu verstehen.
# Ein 'Sliding Window' liefert mehr Kontext für jede Vorhersage und hilft, zukünftige Ereignisse basierend auf diesem Kontext vorherzusagen.
# Die Größe des Fensters ist entscheidend, da sie bestimmt, wie viele vergangene Informationen für die Vorhersage zur Verfügung stehen.
# Eine geeignete Fenstergröße hilft, das Gleichgewicht zwischen dem Erfassen relevanter Muster und dem Vermeiden irrelevanter Informationen zu finden.
# Dieser Ansatz wandelt die Daten in einen reichhaltigeren Merkmalssatz um, indem er jedes Fenster effektiv zu einem Vektor von Merkmalen macht.
def slice_hdfs(df, grouping_type, window_size, logger):
    logger.info("Slicing windows")

    # Verwendung aller Log-Einträge einer Session
    if grouping_type == 'session':
        print(df[:10])

    # Verwendung von Sliding-Windows
    elif grouping_type == 'sliding':

        # Initialisierung einer leeren Liste, um die erstellten Fenster und das jeweils nächste Element zu speichern.
        windows = []
        for index, row in df.iterrows():
            sequence = row['EventSequence']

            # Bestimmen der Länge der Eventsequence
            seqlen = len(sequence)

            # Das Fenster wird so lange über die Sequenz geschoben, solange es noch einen nächsten Wert gibt
            i = 0
            while (i + window_size) < seqlen:
                window_slice = sequence[i: i + window_size]
                next_element = sequence[i + window_size]
                windows.append([index, window_slice, next_element])
                i += 1

            sliced_windows = pd.DataFrame(windows, columns=['id', 'window', 'next'])
            train_x = sliced_windows[['id', 'window']]
            train_y = sliced_windows[['id', 'next']]
    return train_x, train_y

def process_slice(data):
    index, row, window_size = data
    sequence = row['EventSequence']
    seqlen = len(sequence)
    windows = []
    i = 0
    while (i + window_size) < seqlen:
        window_slice = sequence[i: i + window_size]
        next_element = sequence[i + window_size]
        windows.append([index, window_slice, next_element])
        i += 1
    return windows

def slice_hdfs_parallel(df, grouping_type, window_size, num_processes, logger):
    logger.info("Slicing windows")
    if grouping_type == 'session':
        # Code für 'session' bleibt unverändert
        pass

    elif grouping_type == 'sliding':
        # Aufteilen der Daten für die Parallelverarbeitung
        data_splits = [(index, row, window_size) for index, row in df.iterrows()]

        # Erstellen eines Pools von Arbeitern
        with Pool(num_processes) as pool:
            results = pool.map(process_slice, data_splits)

        # Zusammenführen der Ergebnisse
        windows = [item for sublist in results for item in sublist]
        sliced_windows = pd.DataFrame(windows, columns=['id', 'window', 'next'])
        train_x = sliced_windows[['id', 'window']]
        train_y = sliced_windows[['id', 'next']]

    return train_x, train_y


# Mit dem Vectorizer werden EventIDs in numerische IDs umgewandelt
class Vectorizer:
    def __init__(self, logging):
        self.logging = logging

    # Label_Mapping mit allen einzigartig vorkommenden EventIDs und #OOV und #PAD zu erstellen
    def fit_transform(self, x, y):
        self.logging.info("Fitting vectorizer")
        self.label_mapping = {'#OOV': 0, '#PAD': 1}
        next_id_value = 2

        # Extrahieren der einzigartigen IDs aus allen Fenstern
        for index, row in x.iterrows():
            for event_id in row['window']:
                if event_id not in self.label_mapping and event_id != '#PAD':
                    self.label_mapping[event_id] = next_id_value
                    next_id_value += 1

        x_transformed, y_transformed = self.transform(x, y)
        return x_transformed, y_transformed, self.label_mapping

    # Ersetzen der EventIDs mithilfe des label_mappings
    def transform(self, x, y):
        self.logging.info("Transforming vectorizer")
        x_transformed = x.copy()
        for index, row in x.iterrows():
            x_transformed.at[index, 'window'] = [self.label_mapping.get(event_id, self.label_mapping['#OOV']) for
                                                 event_id in row['window']]

        y_transformed = y.copy()
        y_transformed['next'] = y['next'].apply(
            lambda event_id: self.label_mapping.get(event_id, self.label_mapping['#OOV']))

        return x_transformed, y_transformed

    # Bei der Validierung gibt es keine seperaten X und Y Datensätze, sondern nur einen X-Datensatz
    # In diesem werden die EventIDs ersetzt
    def transform_valid(self, x):
        self.logging.info("Transforming vectorizer")
        x_transformed = x.copy()
        for index, row in x.iterrows():
            x_transformed.at[index, 'EventSequence'] = [self.label_mapping.get(event_id, self.label_mapping['#OOV']) for
                                                        event_id in row['EventSequence']]

        return x_transformed
