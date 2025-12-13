import os
import csv

DATA_DIR = "DataSet"
OUT_FILE = "labels.csv"

folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label_id", "name"])
    for idx, name in enumerate(folders):
        writer.writerow([idx, name])

print("Saved labels to", OUT_FILE)
