import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import pickle
from vis import process_file, process_events

def apply_filter(data, fs, lowcut=0.17, highcut=0.4, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, required=True)
    parser.add_argument('-out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = []

    ap_folders = [f for f in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, f))]

    for ap in ap_folders:
        ap_path = os.path.join(args.in_dir, ap)
        files = os.listdir(ap_path)
        
        flow_file = next((f for f in files if f.lower().startswith('flow') and 'events' not in f.lower()), None)
        thorac_file = next((f for f in files if f.lower().startswith('thorac')), None)
        spo2_file = next((f for f in files if f.lower().startswith('spo2')), None)
        events_file = next((f for f in files if f.lower().startswith('flow events')), None)

        if not (flow_file and thorac_file and spo2_file):
            continue

        df_flow = process_file(os.path.join(ap_path, flow_file))
        df_thorac = process_file(os.path.join(ap_path, thorac_file))
        df_spo2 = process_file(os.path.join(ap_path, spo2_file))
        
        df_events = pd.DataFrame()
        if events_file:
            df_events = process_events(os.path.join(ap_path, events_file))

        df_flow['Value'] = apply_filter(df_flow['Value'].values, fs=32)
        df_thorac['Value'] = apply_filter(df_thorac['Value'].values, fs=32)

        start_time = df_flow['Timestamp'].iloc[0]
        end_time = df_flow['Timestamp'].iloc[-1]
        
        current_time = start_time
        window_duration = pd.Timedelta(seconds=30)
        step_duration = pd.Timedelta(seconds=15)

        while current_time + window_duration <= end_time:
            window_end = current_time + window_duration
            
            flow_window = df_flow[(df_flow['Timestamp'] >= current_time) & (df_flow['Timestamp'] < window_end)]
            thorac_window = df_thorac[(df_thorac['Timestamp'] >= current_time) & (df_thorac['Timestamp'] < window_end)]
            spo2_window = df_spo2[(df_spo2['Timestamp'] >= current_time) & (df_spo2['Timestamp'] < window_end)]
            
            label = "Normal"
            if not df_events.empty:
                overlaps = []
                window_events = df_events[(df_events['End'] > current_time) & (df_events['Start'] < window_end)]
                for _, event in window_events.iterrows():
                    ev_start = max(event['Start'], current_time)
                    ev_end = min(event['End'], window_end)
                    overlap_duration = (ev_end - ev_start).total_seconds()
                    if overlap_duration > 15:
                        overlaps.append(event['Type'])
                if overlaps:
                    label = overlaps[0]
            
            dataset.append({
                'AP': ap,
                'Start': current_time,
                'End': window_end,
                'Flow': flow_window['Value'].values,
                'Thorac': thorac_window['Value'].values,
                'SpO2': spo2_window['Value'].values,
                'Label': label
            })
            
            current_time += step_duration

    output_file = os.path.join(args.out_dir, 'dataset.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    main()