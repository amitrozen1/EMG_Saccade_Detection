#import pyedflib
import mne

'''
edf = pyedflib.EdfReader("path_to_your_file.edf")

n_channels = edf.signals_in_file    # number of channels

channel_labels = edf.getSignalLabels()      # channel names

sampling_rates = [edf.getSampleFrequency(i) for i in range(n_channels)]     # sampling frequencies for each channel

for i in range(n_channels):
    print(f"Channel {channel_labels[i]}: {sampling_rates[i]} Hz")

edf.close()
'''
# load the EDF file from EMG toe a raw object using MNE 
raw_data = mne.io.read_raw_edf("C:/Users/moshe/Documents/Saccades_project/data/04122022_1545_A_SD_filtered.edf", preload=False)

channel_labels = raw_data.info['ch_names']   # extracts channel names
sampling_rate = raw_data.info['sfreq']   # extracts sampling frequency

for label in channel_labels:
    # prints the channel name and its sampling rate
    print(f"Channel {label}: {sampling_rate} Hz")
