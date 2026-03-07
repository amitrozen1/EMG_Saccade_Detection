import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')
# Adds the parent directory of the current file to sys.path
# This allows importing local project modules located outside the current folder

import numpy as np

from filtering import load_pkl_file, get_emg_signal_from_chunk, bandpass_filter, downsample_signal, interp_nans, lowpass_filter
from saccade_detection import detect_saccades_engbert_2d, classify_direction_by_angle, detect_blinks_standard, remove_saccades_overlapping_blinks
from differentiation import differentiate
from results_plots import plot_saccade_distribution, make_all_figures
from results_summary import print_results_table


#story_id = "Reading_2.png_4"
ch_raw = 12
freq= 4000
fs_down = 500

stories_original_channels, stories_filtered_channels, \
    stories_EOG_like_channels, downsampled_EOG_like, results_engbert = {}, {}, {}, {}, {}

channels_of_interest = [9, 11, 12, 13, 14, 15]
# 9: below the right eye
# 11: corner of the right eye
# 12: above the right eyebrow arch (medial)
# 13: above the right eyebrow tail (lateral)
# 14: at the top of the forhead (right)
# 15: at the top of the forhead (left)    

data_A = load_pkl_file("C:/Users/moshe/Documents/Saccades_project/data/participant_A_readings_EOG.pkl")
data_B = load_pkl_file("C:/Users/moshe/Documents/Saccades_project/data/participant_B_readings_EOG.pkl")

all_participants = [("PA", data_A), ("PB", data_B)]

#########################################
# retrieving all signals for each story #
#########################################

for pid, data in all_participants:
    for story_id, chunk in data.items():
        full_id = f"{pid}_{story_id}"
        
        story_signals = {}
        for ch in channels_of_interest:
            # Extract all the signal time points for the specified channel index
            story_signals[ch]= chunk.data[ch]  
        
        stories_original_channels[full_id] = story_signals


### Channels substraction to get approximation for EOG signal ###

for full_id, signals in stories_original_channels.items():
    
    stories_EOG_like_channels[full_id] = {}
    curr_story_id = stories_EOG_like_channels[full_id]
    
    v_eog = signals[13] - signals[9]   # above eyebrow arch - below eye
#    v_eog_2 = signals[12] - signals[9]     #lateral above eye - below eye
    h_eog_lf = signals[11] - signals[15]  # right eye corner - left forehead
    h_eog_rf = signals[11] - signals[14]  # right eye corner - right forehead

    curr_story_id["v_eog_saccade"] = v_eog
    curr_story_id["v_eog_blink"] = v_eog
#    curr_story_id["v_eog_2"] = v_eog_2
    curr_story_id["h_eog_lf"] = h_eog_lf
    curr_story_id["h_eog_rf"] = h_eog_rf
    
    
### Apply bandpass filter to each signal in all_signals ###

for full_id, signals in stories_EOG_like_channels.items():
    
    stories_filtered_channels[full_id] = {}
    #curr_story_id = stories_filtered_channels[story_id]
    
    for eog_type, signal_data in signals.items():
        if "v_eog_blink" in eog_type:
            eog_cutoff = 15
        elif "v_eog_saccade" in eog_type:
            eog_cutoff = 25
        elif "h_eog" in eog_type:
            eog_cutoff = 25

        stories_filtered_channels[full_id][eog_type] = lowpass_filter(signal_data, fs=freq, cutoff=eog_cutoff, order=4)

### downsample all signals to fs_down ###

for full_id, signals in stories_filtered_channels.items():
    
    downsampled_EOG_like[full_id] = {}
    
    for ch, signal in signals.items():
        signal_ds = downsample_signal(signal, freq, fs_down)
        downsampled_EOG_like[full_id][ch] = signal_ds

### Detect saccades & blinks ###

def pick_best_heog(h1, h2):
    # choose HEOG derivation with the larger median absolute amplitude
    amp1 = np.median(np.abs(h1))
    amp2 = np.median(np.abs(h2))
    return h1 if amp1 > amp2 else h2


for full_id, signals in downsampled_EOG_like.items():
    results_engbert[full_id] = {}
    
    # parameters definition
    v_eog_sac = signals.get("v_eog_saccade", None)
    v_eog_blink = signals.get("v_eog_blink", None)
    h_eog_raw = signals.get("h_eog_lf", None)  # נסה עם h_eog_lf
    h_eog_alt = signals.get("h_eog_rf", None)  # וגם עם h_eog_rf
    
    if h_eog_raw is not None and h_eog_alt is not None and v_eog_sac is not None:
        h_eog = pick_best_heog(h_eog_raw, h_eog_alt)
    else:
        h_eog = h_eog_raw
    
    time_step = 1.0 / fs_down
    
    # 2D saccade detection for horizontal and vertical velocity pairs
    if v_eog_sac is not None and h_eog is not None:
        # Compute velocities using Lanczos-11 differentiation
        v_vel = differentiate(v_eog_sac.astype(np.float64), step=time_step, method='l11')
        h_vel = differentiate(h_eog.astype(np.float64), step=time_step, method='l11')
        
        # Detect saccades using Engbert 2D algorithm
        saccades = detect_saccades_engbert_2d(h_vel, v_vel, fs=fs_down,eta=6.0, smooth_win=5, 
            min_samples=10, max_ms=80,refractory_ms=0, use_hysteresis=False,edge_frac=0.3)
        results_engbert[full_id]["saccades"] = saccades
        results_engbert[full_id]["h_vel"] = h_vel
        results_engbert[full_id]["v_vel"] = v_vel
        
        # direction classification by angle
        h_saccades = 0
        v_saccades = 0
        obl_saccades = 0

        for onset, pk, offset in saccades:
            label = classify_direction_by_angle(h_vel, v_vel, onset, pk, offset)
            if label == "H":
                h_saccades += 1
            elif label == "V":
                v_saccades += 1
            else:
                obl_saccades += 1
        
        results_engbert[full_id]["dir_counts"] = {"H": h_saccades, "V": v_saccades, "OBL": obl_saccades}
        
        print(f"{full_id} - Saccades (Engbert 2D): "
            f"Total={len(saccades)}, Horizontal={h_saccades}, "
            f"Vertical={v_saccades}, Oblique={obl_saccades}")
        
        
    # Blink detection: vertical-dominant transients with limited horizontal deflection
    blinks = []
    if v_eog_blink is not None and h_eog is not None:
        blinks = detect_blinks_standard( v_eog_blink.astype(float),
            h_eog.astype(float), fs_down)
        results_engbert[full_id]["blinks"] = blinks
        
        print(f"{full_id} - Blinks (standard amplitude-based): "
            f"Found {len(blinks)} events")
        
    # --- remove saccades that overlap with blinks ---
    if "saccades" in results_engbert[full_id]:
        saccades = results_engbert[full_id]["saccades"]
        clean_saccades = remove_saccades_overlapping_blinks(saccades, blinks)
        results_engbert[full_id]["saccades_no_blinks"] = clean_saccades
    
    if "saccades_no_blinks" in results_engbert[full_id]:
        print(f"{full_id} - Clean saccades (no blinks): {len(results_engbert[full_id]['saccades_no_blinks'])}")


#plot_saccade_distribution(df)

print("\n=== RESULTS TABLE ===")
print_results_table(results_engbert, downsampled_EOG_like)
print("=== ANALYSIS COMPLETE ===")

make_all_figures(results_engbert, downsampled_EOG_like, fs_down)
"""
# Summary of detection results
print("\n=== DETECTION SUMMARY ===")
for story_id, events in results_engbert.items():
    saccades = events.get("saccades", [])
    blinks = events.get("blinks", [])
    h_vel = events.get("h_vel", None)
    v_vel = events.get("v_vel", None)
    
    # Classify saccades by direction using raw signal and velocity
    h_sacc = 0
    v_sacc = 0
    
    if h_vel is not None and v_vel is not None:
        # Retrieve raw signals
        signals_data = downsampled_EOG_like[story_id]
        v_eog_sac = signals_data.get("v_eog_saccade", None)
        h_eog = signals_data.get("h_eog_lf", None)
        
        for onset, pk, offset in saccades:
            # Peak velocity comparison
            h_max_vel = np.max(np.abs(h_vel[onset:offset+1]))
            v_max_vel = np.max(np.abs(v_vel[onset:offset+1]))
            
            # Amplitude change comparison
            if v_eog_sac is not None and h_eog is not None:
                h_delta = np.abs(h_eog[offset] - h_eog[onset])
                v_delta = np.abs(v_eog_sac[offset] - v_eog_sac[onset])
            else:
                h_delta = h_max_vel
                v_delta = v_max_vel
            
            # Classification decision
            if h_max_vel > v_max_vel or (h_max_vel == v_max_vel and h_delta > v_delta):
                h_sacc += 1
            else:
                v_sacc += 1
    
    n_blink = len(blinks)
    print(f"{story_id}: Saccades (H={h_sacc}, V={v_sacc}, Total={len(saccades)}), Blinks={n_blink}")

print("=== ANALYSIS COMPLETE ===\n")
"""

