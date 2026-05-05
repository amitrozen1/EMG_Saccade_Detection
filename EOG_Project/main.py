import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append("C:/Users/moshe/Documents/Saccades_project/emg_data_analysis")
sys.stdout.reconfigure(encoding='utf-8')

from filtering import load_pkl_file, downsample_signal, lowpass_filter
from saccade_detection import detect_saccades_engbert_2d, classify_direction_by_angle, detect_blinks_standard, remove_saccades_overlapping_blinks
from differentiation import differentiate
from results_plots import make_all_figures, run_diagnostics
from results_summary import print_results_table, print_saccade_sequence

# Configuration
freq = 4000
fs_down = 500
CHANNELS = {
    'VEOG_UPPER': 13,       # above the right eyebrow tail (lateral)
    'VEOG_LOWER': 9,        # below the right eye
    'HEOG_RIGHT': 11,       # corner of the right eye
    'REF_FRONTAL_L': 15,    # at the top of the forhead (left)
    'REF_FRONTAL_R': 14     # at the top of the forhead (right)
    
}

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

# EOG Approximation (Channel Subtraction)
for full_id, signals in stories_original_channels.items():
    stories_EOG_like_channels[full_id] = {
        "v_eog_saccade": signals[CHANNELS['VEOG_UPPER']] - signals[CHANNELS['VEOG_LOWER']],
        "v_eog_blink": signals[CHANNELS['VEOG_UPPER']] - signals[CHANNELS['VEOG_LOWER']],
        "h_eog_lf": signals[CHANNELS['HEOG_RIGHT']] - signals[CHANNELS['REF_FRONTAL_L']],
        "h_eog_rf": signals[CHANNELS['HEOG_RIGHT']] - signals[CHANNELS['REF_FRONTAL_R']]
    }

# Filtering
for full_id, signals in stories_EOG_like_channels.items():
    stories_filtered_channels[full_id] = {}
    for eog_type, signal_data in signals.items():
        cutoff = 15 if "v_eog_blink" in eog_type else 25
        stories_filtered_channels[full_id][eog_type] = lowpass_filter(signal_data, fs=freq, cutoff=cutoff, order=4)

# Downsampling
for full_id, signals in stories_filtered_channels.items():
    downsampled_EOG_like[full_id] = {}
    for ch, signal in signals.items():
        downsampled_EOG_like[full_id][ch] = downsample_signal(signal, freq, fs_down)

def pick_best_heog(h1, h2, fs):
    """Selects the HEOG derivation with the larger velocity variance using Lanczos differentiation.."""
    time_step = 1.0 / fs
    v1 = differentiate(h1.astype(np.float64), step=time_step, method='l11')
    v2 = differentiate(h2.astype(np.float64), step=time_step, method='l11')
    
    mad1 = np.median(np.abs(v1 - np.median(v1)))
    mad2 = np.median(np.abs(v2 - np.median(v2)))
    return h1 if mad1 > mad2 else h2

# === GRID SEARCH OPTIMIZATION ===
print("\n=== STARTING GRID SEARCH FOR SMOOTHING WINDOW ===")
window_options = [5, 7, 9, 11, 15]
grid_results = []

for full_id, signals in downsampled_EOG_like.items():
    v_eog_sac = signals.get("v_eog_saccade")
    h_eog_raw = signals.get("h_eog_lf")
    h_eog_alt = signals.get("h_eog_rf")
    
    # pick the best HEOG channel based on velocity variance to optimize saccade detection sensitivity
    h_eog = pick_best_heog(h_eog_raw, h_eog_alt, fs_down) if (h_eog_raw is not None and h_eog_alt is not None) else h_eog_raw
    time_step = 1.0 / fs_down
    
    if v_eog_sac is not None and h_eog is not None:
        v_vel = differentiate(v_eog_sac.astype(np.float64), step=time_step, method='l11')
        h_vel = differentiate(h_eog.astype(np.float64), step=time_step, method='l11')
        
        for win in window_options:
            test_saccades = detect_saccades_engbert_2d(
                h_vel, v_vel, fs=fs_down, eta_h=6, eta_v=11, 
                smooth_win=win, min_samples=10, max_ms=150, 
                refractory_ms=0, use_hysteresis=False, edge_frac=0.3
            )
            
            h_cnt = v_cnt = o_cnt = 0
            for onset, _, offset in test_saccades:
                label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
                if label == "H": h_cnt += 1
                elif label == "V": v_cnt += 1
                else: o_cnt += 1
                
            total_sac = len(test_saccades)
            # ratio calculation with handling for zero vertical count to avoid division by zero
            hv_ratio = (h_cnt / v_cnt) if v_cnt > 0 else float('inf')
            
            grid_results.append({
                "Story": full_id,
                "Window": win,
                "Horizontal": h_cnt,
                "Vertical": v_cnt,
                "Oblique": o_cnt,
                "H:V Ratio": round(hv_ratio, 2),
                "Total": total_sac
            })

df_grid = pd.DataFrame(grid_results)
print(df_grid.sort_values(by=["Story", "Window"]).to_string(index=False))
print("=================================================\n")


# Saccade & Blink Detection
for full_id, signals in downsampled_EOG_like.items():
    results_engbert[full_id] = {}
    
    v_eog_sac = signals.get("v_eog_saccade")
    v_eog_blink = signals.get("v_eog_blink")
    h_eog_raw = signals.get("h_eog_lf")
    h_eog_alt = signals.get("h_eog_rf")
    
    h_eog = pick_best_heog(h_eog_raw, h_eog_alt, fs_down) if (h_eog_raw is not None and h_eog_alt is not None) else h_eog_raw
    time_step = 1.0 / fs_down
    
    if v_eog_sac is not None and h_eog is not None:
        v_vel = differentiate(v_eog_sac.astype(np.float64), step=time_step, method='l11')
        h_vel = differentiate(h_eog.astype(np.float64), step=time_step, method='l11')
        
        saccades = detect_saccades_engbert_2d(h_vel, v_vel, fs=fs_down, eta_h=6, eta_v=11, smooth_win=15, 
            min_samples=10, max_ms=150, refractory_ms=15, use_hysteresis=False, edge_frac=0.3)
        
        results_engbert[full_id].update({"saccades": saccades, "h_vel": h_vel, "v_vel": v_vel})
        
        h_saccades = v_saccades = obl_saccades = 0
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
        
    if v_eog_blink is not None and h_eog is not None:
        blinks = detect_blinks_standard(v_eog_blink.astype(float), h_eog.astype(float), fs_down, k_vert=5.0)
        results_engbert[full_id]["blinks"] = blinks
        print(f"{full_id} - Blinks: Found {len(blinks)} events")
        
    if "saccades" in results_engbert[full_id]:
        clean_saccades = remove_saccades_overlapping_blinks(results_engbert[full_id]["saccades"], blinks)
        results_engbert[full_id]["saccades_no_blinks"] = clean_saccades
        print(f"{full_id} - Clean saccades: {len(clean_saccades)}")

# Results & Visualization
print("\n=== RESULTS TABLE ===")
print_results_table(results_engbert, downsampled_EOG_like)
print("=== ANALYSIS COMPLETE ===")

make_all_figures(results_engbert, downsampled_EOG_like, fs_down)

# Diagnostics, Validation Plots & Sequences
run_diagnostics(results_engbert, downsampled_EOG_like, fs_down)
print_saccade_sequence(results_engbert, "PA_Reading_13.png_2")
print_saccade_sequence(results_engbert, "PB_Reading_14.png_1")