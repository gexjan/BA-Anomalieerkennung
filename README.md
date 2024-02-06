# BA Anomalieerkennung Datenbank-Logdateien mit maschinellem Lernen

## Installation
Das Skript benötigt diverse Pakete diese können installiert werden mit 
```bash
conda env xxxxx
```

## Datensätze
Das Projekt unterstützt zwei Datensätze:
1. **PostgreSQL:** Hierbei werden die Daten der Datenbanksysteme verwendet. Es gibt die zwei Dateien `postgres_train.log` und `postgres_test.log`
2. **HDFS:** Dieser Datensatz wird zur Evaluation des Projekts bzw. zum Vergleich mit anderen Lösungen genutzt. Es gibt die Dateien `hdfs_train.log` und `hdfs_test.log`. Es ist zudem möglich mittels `create_hdfs_dataset.py` Skript aus dem ordner `helper` die zwei Datei aus der ursprünglichen Log-Datei zu erzeugen. Dann kann bei der Größe variiert werden.

## Nutzung
Es stehen diverse Modi zur Ausführung bereit:

### Prepare
Dieser Schritt umfasst das Pre-Processing. Es werden die Log-Dateien eingelesen und Fenster gebildet, die für das Training verwendet werden können. Befehl:
```bash
python main.py -prepare
```
Es ist auch möglich das erneute Parsen zu verhindern, sofern dies bereits ausgeführt wurde
```bash
python main.py -prepare --noparse
```
Weiterhin ist es möglich die Pfade der Log-Dateien und -Ordner anzugeben, eine vollständige übersicht über alle Argumente kann ausgegeben werden mit 
```bash
python main.py --help
```

### Train
Dieser Schritt setzt die Ausführung von `prepare` voraus.
Das training lässt sich ausführen mit:
```bash
python main.py -train
```
Die Ausgabe ist ein Model im Ordner `data`.
Es können weitere Argumente angegeben werden, siehe `--help`

Es gibt die Möglichkeit bereits während des Trainings den F1-Score nach jeder Epoche zu berechnen. Das erhöht jedoch sehr stark die benötigte Zeit:
```bash
python main -train --calculate-f
```

Beim Training werden diverse Dateien erzeugt und im `data` Ordner ausgegeben:
- `loss_per_epoch.csv`
- `f1_per_epoch.csv` sofern `--calculate-f` verwendet wird
- `training_loss.png`
- `f1_epoch.png`
- `combined_loss_f1.png`

### Evaluate
Im Evaluate-Modus wird das trainierte Modell evaluiert und ein F1-Score sowie Precision und Recall ausgegeben:
```bash
python main.py -evaluate
```
Zur Evaluation werden die Datensätze `hdfs_test.log` bzw. `hdfs_train.log` verwendet.

### Hptune
Hptune ist das Hyperparameter-Tuning. Dieses basiert auf dem Framework `optuna:
```bash
python main.py -hptune --hptrials xx
```
Mittels `--hptrials` lässt sich die Anzahl der Ausführungen angeben. Optuna nutzt standardmäßig zur Wahl der Hyperparameter die bays`sche Optimierung.

Beim Hyperparameter-Tuning werden diverse Dateien erzeugt und im Ordner `data` ausgegeben:
- `study_results.csv`
- `optimization_history.png`
- `param_importances.png`
- `contour_plot.png`
- `prallel_coordinate.png`

### Predict
TBA

### Zusammenfassung
Für ein vollständigen lauf unter Nutzung der standardmäßigen Dateien:
```bash
python main.py -prepare -train -predict
```

Das erzeugte Modell kann dann im Ordner data mit dem Dateinamen `lstm_model.pth` gefunden werden.