import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

filename = "data/AP01/Flow - 30-05-2024.txt"

with open(filename, 'r') as f:
    lines = f.readlines()

data_lines = lines[7:]
data = []
for line in data_lines:
    parts = line.strip().split(';')
    if len(parts) == 2:
        data.append([parts[0].strip(), parts[1].strip()])

df = pd.DataFrame(data, columns=['RawTime', 'Value'])
df['Value'] = pd.to_numeric(df['Value'])

df['RawTime'] = df['RawTime'].str.replace(',', '.')
df['Timestamp'] = pd.to_datetime(df['RawTime'], format='%d.%m.%Y %H:%M:%S.%f')

start_time = df['Timestamp'].iloc[0]
end_time = start_time + pd.Timedelta(seconds=5)

df_subset = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df_subset['Timestamp'], df_subset['Value'])

ticks = pd.date_range(start=start_time, end=end_time, periods=35)
ax.set_xticks(ticks)

def format_func(x, pos):
    dt = mdates.num2date(x)
    return f"{dt.strftime('%H:%M:%S')},{dt.microsecond // 1000:03d}"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()