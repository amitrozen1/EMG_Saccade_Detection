import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
# Adds the parent directory of the current file to sys.path
# This allows importing local project modules located outside the current folder

import numpy as np

from filtering import load_pkl_file, get_emg_signal_from_chunk, bandpass_filter, downsample_signal, interp_nans
from saccades_detection import find_saccades_on_channel, plot_velocity_with_method
from visualize import plot_signal_with_saccades, stage_stats, plot_signal_stage
from validate_signal import qc_plot_signal_and_velocity

story_id = "Reading_2.png_4"
ch_raw = 12

data_A = load_pkl_file("C:/Users/moshe/Documents/Saccades_project/data/participant_A_readings_EOG.pkl")

channels_of_interest = [9, 11, 12, 13, 14, 15]
          
# 9: below the right eye
# 11: corner of the right eye
# 12: above the right eye (medial)
# 13: above the right eye (lateral)
# 14: at the top of the forhead (right)
# 15: at the top of the forhead (left)


### retrieving all signals for each story ###

stories_original_channels, stories_filtered_channels, stories_EOG_like_channels, downsampled_EOG_like = {}, {}, {}, {}


# retrive frequency from the first chunk of data (if not specified in chunk, default to 4000 Hz)

first_chunk = next(iter(data_A.values()))
freq = getattr(first_chunk, 'freq', 4000)
#print(f"Using frequency: {freq} Hz")

for story_id, chunk in data_A.items():
    #print(f"Processing story: {story_id}")
    story_signals = {}

    for ch in channels_of_interest:
        story_signals[ch]= chunk.data[ch]  # Extract all the signal time points for the specified channel index
    
    stories_original_channels[story_id] = story_signals

#raw_sig = stories_original_channels[story_id][ch_raw]
#stage_stats(raw_sig, f"raw ch{ch_raw}")
#plot_signal_stage(raw_sig, fs=freq, title=f"RAW ch{ch_raw} (fs={freq}Hz)")

for ch, sig in stories_original_channels[story_id].items():
    n_nan = np.isnan(np.asarray(sig)).sum()
    print(f"RAW ch{ch}: NaNs={n_nan}")

### Apply bandpass filter to each signal in all_signals ###


for story_id, signals in stories_original_channels.items():
    
    stories_filtered_channels[story_id] = {}
    #curr_story_id = stories_filtered_channels[story_id]
    
    for ch, signal in signals.items():
        stories_filtered_channels[story_id][ch] = bandpass_filter(signal, fs=freq, lowcut=0.1, highcut=40.0, order=4)
        #Coarse filtering before subtraction

#filt_sig = stories_filtered_channels[story_id][ch_raw]
#stage_stats(filt_sig, f"bandpassed ch{ch_raw}")
#plot_signal_stage(filt_sig, fs=freq, title=f"FILTERED ch{ch_raw} (fs={freq}Hz)")

### Channels substraction to get approximation for EOG signal ###

for story_id, signals in stories_filtered_channels.items():
    
    stories_EOG_like_channels[story_id] = {}
    curr_story_id = stories_EOG_like_channels[story_id]
    
    v_eog_1 = signals[13] - signals[9]   # medial above eye - below eye
    v_eog_2 = signals[12] - signals[9]   # lateral above eye - below eye
    h_eog_1 = signals[11] - signals[15]  # right eye corner - left forehead
    h_eog_2 = signals[11] - signals[14]  # right eye corner - right forehead

    curr_story_id["v_eog_1"] = v_eog_1
    curr_story_id["v_eog_2"] = v_eog_2
    curr_story_id["h_eog_1"] = h_eog_1
    curr_story_id["h_eog_2"] = h_eog_2
    
    '''
    for ch, filtered_signal in signals.items():
        curr_story_id["v_eog_1"] = curr_story_id[13] - curr_story_id[9]  # v_eog_1 = above right eye (medial) - below right eye
        curr_story_id["v_eog_2"] = curr_story_id[12] - curr_story_id[9]  # v_eog_2 = above right eye (lateral) - below right eye
        curr_story_id["h_eog_1"] = curr_story_id[11] - curr_story_id[15]  # h_eog_1 = corner of right eye (medial) - left forhead
        curr_story_id["h_eog_2"] = curr_story_id[11] - curr_story_id[14]  # h_eog_2 = corner of right eye (medial) - right forhead
'''


### Detect saccades for each channel in each story ###

for story_id, signals in stories_EOG_like_channels.items():
    
    downsampled_EOG_like[story_id] = {}
    
    for ch, signal in signals.items():
        signal_ds = downsample_signal(signal, 8, zero_phase=True)  # downsample by factor of 8
        downsampled_EOG_like[story_id][ch] = signal_ds
        

### Detect saccades using velocity threshold method ###
    
fs_down = 500
results = {}
for story_id, chans in downsampled_EOG_like.items():
    results[story_id] = {}
    for name, sig in chans.items():
        results[story_id][name] = find_saccades_on_channel(
            sig, fs=fs_down, method='l11', k=3.5, low_frac=0.75, min_dur_s=0.012, robust=True, smooth_ms=5, refractory_s=0.025
        )

    
story_id = "Reading_2.png_4"
electrode = "v_eog_1"

signal = downsampled_EOG_like[story_id][electrode]
fs = 500  # דוגמה – לשים את הערך הנכון בהתאם ל-downsampling שלך

#plot_velocity_with_method(signal, fs, method='l11', threshold=20, title=f"{story_id} - {electrode}")






'''

fs_raw   = freq          # 4000 בד"כ
fs_down  = 500           # כי factor=8 ב-decimate/downsample
method   = 'l11'         # נגזרת Lanczos-11
thr_vis  = 20            # סף לצביעה בגרף (ויזואלי בלבד)
k_stats  = 3.0           # μ + k·σ
low_frac = 0.3           # היסטרזיס
min_dur  = 0.010         # 10ms

sid = "Reading_2.png_4"  # אפשר לשנות לסיפור אחר
names = ["v_eog_1","v_eog_2","h_eog_1","h_eog_2"]

print("\n=== BULK EOG-like scan ===")
for name in names:
    x = stories_EOG_like_channels[sid][name]

    # אחרי הסינון (fs_raw)
    stage_stats(x, f"{sid} :: {name} (filtered, fs={fs_raw}Hz)")
    plot_signal_stage(x, fs=fs_raw, title=f"{sid} :: {name} (filtered, fs={fs_raw}Hz)")

    # downsample בטוח (מנקה NaN בפנים)
    xd = downsample_signal(x, factor=8, zero_phase=True)
    stage_stats(xd, f"{sid} :: {name} (decimated, fs={fs_down}Hz)")
    plot_signal_stage(xd, fs=fs_down, title=f"{sid} :: {name} (decimated, fs={fs_down}Hz)")

    # מהירות + סימון סף ויזואלי
    plot_velocity_with_method(xd, fs=fs_down, method=method, threshold=thr_vis,
                              title=f"{sid} :: {name} velocity ({method}), thr={thr_vis}")

    # זיהוי סקאדות (VT μ+k·σ)
    res = find_saccades_on_channel(
        xd, fs=fs_down, method=method,
        k=k_stats, low_frac=low_frac, min_dur_s=min_dur
    )
    n_ev = int(res["onsets"].size)
    if n_ev:
        dur_ms = (res["offsets"] - res["onsets"]) * (1000.0 / fs_down)
        print(f"{sid} :: {name} -> events={n_ev}, thr_high={res['thr_high']:.3g}, "
              f"thr_low={res['thr_low']:.3g}, dur(ms) first 5={np.round(dur_ms[:5],1)}")
    else:
        print(f"{sid} :: {name} -> events=0, thr_high={res['thr_high']:.3g}, thr_low={res['thr_low']:.3g}")

print("=== DONE ===\n")
'''

fs_down = 500
sid = "Reading_2.png_4"
for name in ["v_eog_1","v_eog_2","h_eog_1","h_eog_2"]:
    sig = downsampled_EOG_like[sid][name]
    qc_plot_signal_and_velocity(
        sig, fs_down,
        method='l11', k=3.5, low_frac=0.75, min_dur_s=0.012,
        title_prefix=f"{sid} :: {name}"
    )
