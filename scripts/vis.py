import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

flow_file = "Data/AP01/Flow - 30-05-2024.txt"
sleep_file = "Data/AP01/Sleep profile - 30-05-2024.txt"

def extractor(file, save_path=None):
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
            t[1] = "".join(t[1][:-5]) # im ignoring the milliseconds
            t[2] = "".join(t[2][:-2])
            table.append(t)
        
        df = pd.DataFrame(table, columns=headers)

        if save_path is not None and os.path.exists(file) == False:
            df.to_csv(save_path) 

    return (df, meta)

def sampler(df, sample_rate):
    indices = np.arange(0, len(df) - sample_rate, sample_rate)
    return df.iloc[indices,:] # plot based on seconds

df_flow, meta_flow = extractor(flow_file, "AP01_FLOW.csv")
df_sleep, meta_sleep = extractor(sleep_file, "AP01_SLEEP.csv")

sampler(df_flow, 32)

# df_synchro = df_flow[df_flow["Time"] == df_sleep["Time"]]
# print(df_synchro.head())
# plt.plot(df_flow['Time'].head(100), df_flow['Value'].head(100))

# plt.xticks(rotation=90)
# plt.show()