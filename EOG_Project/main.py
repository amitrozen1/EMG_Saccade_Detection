import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adds the parent directory of the current file to sys.path
# This allows importing local project modules located outside the current folder

import numpy as np
from EDF import EdfAnalyzer
from scipy.signal import decimate

from filtering import load_pkl_file, get_emg_signal_from_chunk, bandpass_filter
from VT import detect_saccades_velocity_threshold



data_A = load_pkl_file("C:/Users/moshe/Documents/Saccades_project/data/participant_A_readings_EOG.pkl")

channels_of_interest = [9, 11, 12, 13, 14, 15]

# 9: below the right eye
# 11: corner of the right eye
# 12: above the right eye (medial)
# 13: above the right eye (lateral)
# 14: at the top of the forhead (right)
# 15: at the top of the forhead (left)


### retrieving all signals for each story ###

stories_original_channels = {}
stories_filtered_channels = {}
stories_EOG_like_channels = {}

# retrive frequency from the first chunk of data (if not specified in chunk, default to 500 Hz)

first_chunk = next(iter(data_A.values()))
freq = getattr(first_chunk, 'freq', 4000)  # default to 4000 if not found
#print(f"Using frequency: {freq} Hz")

for story_id, chunk in data_A.items():

    story_signals = {}

    for ch in channels_of_interest:
        story_signals[ch]= chunk.data[ch]  # Extract all the signal time points for the specified channel index
    
    stories_original_channels[story_id] = story_signals


### Apply bandpass filter to each signal in all_signals ###


for story_id, signals in stories_original_channels.items():
    
    stories_filtered_channels[story_id] = {}
    #curr_story_id = stories_filtered_channels[story_id]
    
    for ch, signal in signals.items():
        stories_filtered_channels[story_id][ch] = bandpass_filter(signal, fs=freq, lowcut=0.1, highcut=40.0, order=4)
        #Coarse filtering before subtraction

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

signal_ds = decimate(filtered_signal, downsample_factor, ftype='iir', zero_phase=True)
