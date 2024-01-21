import os

def load_log(log_dir, log_file):
    log_path = os.path.join(log_dir, log_file)

    # Prüfen ob die Datei existiert
    if not os.path.exists(log_path):
        print(f"Die Log-Datei {log_path} existiert nicht")
        return
    
    return log_path