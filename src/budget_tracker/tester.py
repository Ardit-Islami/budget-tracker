from pathlib import Path
import csv

path = Path("/home/arditi/code/budget-tracker-new/data/raw/NationWide 01.10 to 31.10.csv")

with path.open("r", encoding="cp1252", errors="replace", newline="") as f:
    reader = csv.reader(f)
    lengths = {}
    for i, row in enumerate(reader, start=1):
        n = len(row)
        lengths[n] = lengths.get(n, 0) + 1
        # Print the first few rows that deviate from the majority width
        # once we know what the majority is.
        if i < 50:
            print(i, n, row)

print("Column-count histogram:", lengths)