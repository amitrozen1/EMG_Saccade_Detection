import sys
import os
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
code_dir = os.path.join(project_dir, 'emg_data_analysis')
data_dir = os.path.join(project_dir, 'data')

sys.path.append(project_dir)
sys.path.append(code_dir)
sys.stdout.reconfigure(encoding='utf-8')

from config import FREQ, FS_DOWN, CHANNELS, ENGBERT_PARAMS
from filtering import load_pkl_file, downsample_signal, lowpass_filter
from saccade_detection import (detect_saccades_engbert_2d, classify_direction_by_angle, detect_blinks,
    remove_saccades_overlapping_blinks, pick_best_heog)
from becerra_code.src.shared.differentiation import differentiate
from results_plots import make_all_figures, run_diagnostics, plot_pipeline_validation
from results_summary import print_results_table, print_saccade_sequence

data_A = load_pkl_file(os.path.join(data_dir, "participant_A_readings_EOG.pkl"))
data_B = load_pkl_file(os.path.join(data_dir, "participant_B_readings_EOG.pkl"))
all_participants = [("PA", data_A), ("PB", data_B)]

stories_original_channels = {}
stories_EOG_like_channels = {}
stories_filtered_channels = {}
downsampled_EOG_like = {}
results_engbert = {}

# Retrieve signals
for pid, data in all_participants:
    for story_id, chunk in data.items():
        full_id = f"{pid}_{story_id}"
        story_signals = {ch: chunk.data[ch] for ch in CHANNELS.values()}
        stories_original_channels[full_id] = story_signals

# Pseudo-EOG construction
for full_id, signals in stories_original_channels.items():
    v_eog = signals[CHANNELS['v_eog_up']] - signals[CHANNELS['v_eog_down']]
    h_eog_lf = signals[CHANNELS['h_eog_right']] - signals[CHANNELS['ref_l']]
    h_eog_rf = signals[CHANNELS['h_eog_right']] - signals[CHANNELS['ref_r']]

    stories_EOG_like_channels[full_id] = {"v_eog_saccade": v_eog, "v_eog_blink": v_eog, "h_eog_lf": h_eog_lf, "h_eog_rf": h_eog_rf}

# Filtering
for full_id, signals in stories_EOG_like_channels.items():
    stories_filtered_channels[full_id] = {}
    for eog_type, signal_data in signals.items():
        cutoff = 15 if "v_eog_blink" in eog_type else 25
        stories_filtered_channels[full_id][eog_type] = lowpass_filter(signal_data, fs=FREQ, cutoff=cutoff, order=4)

# Downsampling
for full_id, signals in stories_filtered_channels.items():
    downsampled_EOG_like[full_id] = {}
    for ch, signal in signals.items():
        downsampled_EOG_like[full_id][ch] = downsample_signal(signal, FREQ, FS_DOWN)

# Saccade and blink Detection
for full_id, signals in downsampled_EOG_like.items():
    results_engbert[full_id] = {}
    
    v_eog_sac = signals.get("v_eog_saccade")
    v_eog_blink = signals.get("v_eog_blink")
    h_eog_raw = signals.get("h_eog_lf")
    h_eog_alt = signals.get("h_eog_rf")
    
    if h_eog_raw is not None and h_eog_alt is not None:
        h_eog = pick_best_heog(h_eog_raw, h_eog_alt, FS_DOWN)
    else:
        h_eog = h_eog_raw
    
    results_engbert[full_id]["h_eog_used"] = h_eog
    time_step = 1.0 / FS_DOWN
    blinks = []
    
    # Detect saccades
    if v_eog_sac is not None and h_eog is not None:
        v_vel = differentiate(v_eog_sac.astype(np.float64), step=time_step, method='l11')
        h_vel = differentiate(h_eog.astype(np.float64), step=time_step, method='l11')
        
        saccades = detect_saccades_engbert_2d(h_vel, v_vel, fs=FS_DOWN, **ENGBERT_PARAMS)
        
        results_engbert[full_id].update({"saccades": saccades, "h_vel": h_vel, "v_vel": v_vel})
        
        h_saccades, v_saccades, obl_saccades = 0, 0, 0
        for onset, _, offset in saccades:
            label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
            if label == "H":
                h_saccades += 1
            elif label == "V":
                v_saccades += 1
            else:
                obl_saccades += 1
        
        results_engbert[full_id]["dir_counts"] = {"H": h_saccades, "V": v_saccades, "OBL": obl_saccades}
        print(f"{full_id} - Saccades (Engbert 2D): Total={len(saccades)}, H={h_saccades}, V={v_saccades}, OBL={obl_saccades}")
    
    # Detect blinks    
    if v_eog_blink is not None:
        v_blink_diff = differentiate(v_eog_blink.astype(np.float64), step=time_step, method='l11')
        blinks = detect_blinks(v_eog_blink.astype(float), v_blink_diff.astype(float), FS_DOWN)
        results_engbert[full_id]["blinks"] = blinks
        print(f"{full_id} - Blinks: Found {len(blinks)} events")

    # Clean overlaps
    if "saccades" in results_engbert[full_id]:
        clean_saccades = remove_saccades_overlapping_blinks(results_engbert[full_id]["saccades"], blinks, fs=FS_DOWN)
        results_engbert[full_id]["saccades_no_blinks"] = clean_saccades
        print(f"{full_id} - Clean saccades: {len(clean_saccades)}")

print("\n RESULTS TABLE\n")
#print_results_table(results_engbert, downsampled_EOG_like)
print_results_table(
    results_engbert, downsampled_EOG_like, FS_DOWN
)
print("\nANALYSIS COMPLETE \n")

#make_all_figures(results_engbert, downsampled_EOG_like, FS_DOWN)

#run_diagnostics(results_engbert, downsampled_EOG_like, FS_DOWN)
#print_saccade_sequence(results_engbert, "PA_Reading_13.png_2")
#print_saccade_sequence(results_engbert, "PB_Reading_14.png_1")