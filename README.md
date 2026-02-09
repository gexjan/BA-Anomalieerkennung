# BA Anomalieerkennung Datenbank-Logdateien mit maschinellem Lernen
Der Code in diesem Repository stellt die Implementierung der zugehörigen Bachelorarbeit dar.

## Installation
Das Skript benötigt verschiedene Pakete. Diese können mit folgendem Befehl installiert werden:
```bash
conda env xxxxx
```

## Datensätze
Das Projekt unterstützt zwei Datensätze:
1. **PostgreSQL:** Hierbei werden die Daten des Datenbanksystems verwendet. Es gibt die beiden Dateien `postgres_train.log` und `postgres_test.log`.
2. **HDFS:** Dieser Datensatz wird zur Evaluation des Projekts bzw. zum Vergleich mit anderen Lösungen genutzt. Es gibt die Dateien `hdfs_train.log` und `hdfs_test.log`. Außerdem ist es möglich, mit dem Skript `create_hdfs_dataset.py` aus dem Ordner `helper` die beiden Dateien aus der ursprünglichen Log-Datei zu erzeugen und dabei die Größe zu variieren.

## Nutzung
Es stehen verschiedene Ausführungsmodi zur Verfügung:

### Prepare
Dieser Schritt umfasst das Preprocessing. Die Log-Dateien werden eingelesen und Fenster gebildet, die für das Training verwendet werden können. Befehl:
```bash
python main.py -prepare
```
Es ist außerdem möglich, das erneute Parsen zu verhindern, sofern dieser Schritt bereits ausgeführt wurde:
```bash
python main.py -prepare --noparse
```
Zusätzlich können die Pfade der Log-Dateien und -Ordner angegeben werden. Eine vollständige Übersicht über alle Argumente kann so ausgegeben werden:
```bash
python main.py --help
```

### Train
Dieser Schritt setzt die Ausführung von `prepare` voraus.
Das Training lässt sich wie folgt ausführen:
```bash
python main.py -train
```
Die Ausgabe ist ein Model im Ordner `data`.
Weitere Argumente sind über `--help` einsehbar.

Es gibt die Möglichkeit, bereits während des Trainings den F1-Score nach jeder Epoche zu berechnen. Das erhöht jedoch die benötigte Zeit deutlich:
```bash
python main.py -train --calculate-f
```

Beim Training werden verschiedene Dateien erzeugt und im Ordner `data` ausgegeben:
- `loss_per_epoch.csv`
- `f1_per_epoch.csv` sofern `--calculate-f` verwendet wird
- `training_loss.png`
- `f1_epoch.png`
- `combined_loss_f1.png`

### Evaluate
Im Evaluate-Modus wird das trainierte Modell ausgewertet und es werden F1-Score, Precision und Recall ausgegeben:
```bash
python main.py -evaluate
```
Zur Evaluation werden die Datensätze `hdfs_test.log` bzw. `hdfs_train.log` verwendet.

### Hptune
Hptune steht für Hyperparameter-Tuning. Dieses basiert auf dem Framework `optuna`:
```bash
python main.py -hptune --hptrials xx
```
Mit `--hptrials` lässt sich die Anzahl der Durchläufe festlegen. Optuna nutzt standardmäßig bayessche Optimierung zur Wahl der Hyperparameter.

Beim Hyperparameter-Tuning werden verschiedene Dateien erzeugt und im Ordner `data` ausgegeben:
- `study_results.csv`
- `optimization_history.png`
- `param_importances.png`
- `contour_plot.png`
- `prallel_coordinate.png`

### Predict
TBA

### Zusammenfassung
Für einen vollständigen Lauf unter Nutzung der Standarddateien:
```bash
python main.py -prepare -train -predict
```

Das erzeugte Modell kann anschließend im Ordner `data` unter dem Dateinamen `lstm_model.pth` gefunden werden.
