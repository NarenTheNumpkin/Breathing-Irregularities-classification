import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

def process_file(filepath):
    with open(filepath, 'r') as f:
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
    return df

def process_events(filepath):
    events = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if ';' in line and '-' in line and line[0].isdigit():
            parts = line.split(';')
            if len(parts) >= 3:
                time_range = parts[0].strip()
                event_type = parts[2].strip()
                
                date_start, end_time_str = time_range.split('-')
                date_str, start_time_str = date_start.split(' ')
                
                start_dt_str = f"{date_str} {start_time_str}".replace(',', '.')
                end_dt_str = f"{date_str} {end_time_str}".replace(',', '.')
                
                start_dt = pd.to_datetime(start_dt_str, format='%d.%m.%Y %H:%M:%S.%f')
                end_dt = pd.to_datetime(end_dt_str, format='%d.%m.%Y %H:%M:%S.%f')
                
                if end_dt < start_dt:
                    end_dt += pd.Timedelta(days=1)
                    
                events.append({
                    'Start': start_dt,
                    'End': end_dt,
                    'Type': event_type
                })
    if events:
        return pd.DataFrame(events)
    return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True)
    args = parser.parse_args()

    ap_path = args.name
    ap_name = os.path.basename(os.path.normpath(ap_path))
    temp_out_path = os.path.join("Plots_Output", ap_name)
    vis_out_path = "Visualizations"

    os.makedirs(temp_out_path, exist_ok=True)
    os.makedirs(vis_out_path, exist_ok=True)

    files = os.listdir(ap_path)
    flow_file = next((f for f in files if f.lower().startswith('flow') and 'events' not in f.lower()), None)
    thorac_file = next((f for f in files if f.lower().startswith('thorac')), None)
    spo2_file = next((f for f in files if f.lower().startswith('spo2')), None)
    events_file = next((f for f in files if f.lower().startswith('flow events')), None)

    if not (flow_file and thorac_file and spo2_file):
        print("Required data files missing.")
        exit()

    df_flow = process_file(os.path.join(ap_path, flow_file))
    df_thorac = process_file(os.path.join(ap_path, thorac_file))
    df_spo2 = process_file(os.path.join(ap_path, spo2_file))

    df_events = pd.DataFrame()
    if events_file:
        df_events = process_events(os.path.join(ap_path, events_file))

    configs = [
        {"df": df_flow, "label": "Nasal Flow", "color": "tab:blue", "ylabel": "Nasal Flow (L/min)"},
        {"df": df_thorac, "label": "Thorac/Abdominal Resp.", "color": "tab:orange", "ylabel": "Resp. Amplitude"},
        {"df": df_spo2, "label": "SpO2", "color": "gray", "ylabel": "SpO2 (%)"}
    ]

    start_time = df_flow['Timestamp'].iloc[0]
    end_time = df_flow['Timestamp'].iloc[-1]
    
    current_time = start_time

    while current_time < end_time:
        window_end = current_time + pd.Timedelta(minutes=5)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        start_str = current_time.strftime('%Y-%m-%d %H:%M')
        end_str = window_end.strftime('%Y-%m-%d %H:%M')
        title = f"{ap_name} - {start_str} to {end_str}"
        fig.suptitle(title, fontsize=16)
        
        has_data = False
        
        for i, ax in enumerate(axes):
            config = configs[i]
            df = config["df"]
            df_subset = df[(df['Timestamp'] >= current_time) & (df['Timestamp'] < window_end)]
            
            if not df_subset.empty:
                has_data = True
                ax.plot(df_subset['Timestamp'], df_subset['Value'], label=config["label"], color=config["color"], linewidth=1.0)
            
            if not df_events.empty and i == 0:
                window_events = df_events[(df_events['End'] > current_time) & (df_events['Start'] < window_end)]
                for _, event in window_events.iterrows():
                    ev_start = max(event['Start'], current_time)
                    ev_end = min(event['End'], window_end)
                    
                    if 'Hypopnea' in event['Type']:
                        color = 'yellow'
                    else:
                        color = 'red'
                    
                    ax.axvspan(ev_start, ev_end, color=color, alpha=0.3)
                    mid_point = ev_start + (ev_end - ev_start) / 2
                    ax.text(mid_point, 0.98, f"{event['Type']}", transform=ax.get_xaxis_transform(),
                            rotation=0, verticalalignment='top', horizontalalignment='center', 
                            fontsize=9, color='black')

            ax.set_ylabel(config["ylabel"], fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if has_data:
            axes[-1].xaxis.set_major_locator(mdates.SecondLocator(interval=6))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            axes[-1].tick_params(axis='x', rotation=90, labelsize=8)
            axes[-1].set_xlabel("Time", fontsize=12)
            
            plt.subplots_adjust(hspace=0.1, bottom=0.18, top=0.92)
            
            filename = f"{ap_name}_{current_time.strftime('%Y%m%d_%H%M')}_to_{window_end.strftime('%H%M')}.png"
            plt.savefig(os.path.join(temp_out_path, filename))
        
        plt.close(fig)
        current_time = window_end

    image_files = sorted([f for f in os.listdir(temp_out_path) if f.endswith('.png')])
    
    if image_files:
        images = []
        for img_file in image_files:
            img_path = os.path.join(temp_out_path, img_file)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            
        pdf_path = os.path.join(vis_out_path, f"{ap_name}.pdf")
        images[0].save(pdf_path, save_all=True, append_images=images[1:])