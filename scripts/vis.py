import numpy as np
import pandas as pd
import argparse

file = "Data/AP01/Flow - 30-05-2024.txt"
meta = {}

with open(file, "r") as f:
    for _ in range(1, 5):
        line = f.readline()
        row = line.split(":", 1)
        meta[row[0]] = row[1]
    
    lines = f.readlines()
    table = []
    headers = ["Date", "Time", "Value"]

    for i in range(7, len(lines)):
        t = lines[i].split(" ")
        t[1] = "".join(t[1][:-1])
        t[2] = "".join(t[2][:-2])
        table.append(t)

    df = pd.DataFrame(table, columns=headers)
    print(df.iloc[0])
