import os
import re
from datetime import datetime


base_dir = '../data/04'
input_dir = base_dir + '/raw'
combined_dir = base_dir + '/combined'
singleline_dir = base_dir + '/singleline'
output_dir = base_dir + '/parsed/'
combined_log = 'combined_logs.log'


# Erstellen einer Liste aller Logdateien im Ordner
log_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Wörterbuch zum Speichern von Datum und Dateinamen
date_file_mapping = {}

# Regex für das Erfassen des Datums im Timestamp
date_regex = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Durchlaufen aller Dateien und Erfassen des Datums aus dem ersten Logeintrag
for file in log_files:
    with open(os.path.join(input_dir, file), 'r') as f:
        first_line = f.readline()
        date_match = date_regex.search(first_line)
        if date_match:
            date_str = date_match.group(1)
            date_file_mapping[date_str] = file

# Sortieren der Dateien basierend auf dem Datum
sorted_dates = sorted(date_file_mapping.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))

# Überprüfen auf fehlende Tage
missing_dates = []
for i in range(1, len(sorted_dates)):
    current_date = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
    previous_date = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d")
    if (current_date - previous_date).days > 1:
        missing_dates.append(current_date.strftime("%Y-%m-%d"))

# Kombinieren der Dateien in der richtigen Reihenfolge
with open(f"{combined_dir}/{combined_log}", 'w') as outfile:
    for date in sorted_dates:
        file = date_file_mapping[date]
        with open(os.path.join(input_dir, file), 'r') as infile:
            outfile.write(infile.read())

print(missing_dates)

# Beispiel: sort_and_combine_logs("path/to/your/log/files")
# Hinweis: Der tatsächliche Pfad zum Log-Dateiordner muss angegeben werden.