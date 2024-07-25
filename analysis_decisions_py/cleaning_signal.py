import numpy as np

def remove_emg_artifacts(emg_signals, std_threshold=2):
    """
    Remove artifacts from EMG signals by filtering out values that deviate more than a specified number
    of standard deviations from the mean of each channel (row).
    
    Parameters:
    - emg_signals: A 2D numpy array where each row represents an EMG signal from one channel.
    - std_threshold: The number of standard deviations from the mean used to identify outliers (default is 3).
    
    Returns:
    - A 2D numpy array with artifacts (outliers) replaced by NaN, indicating these values have been removed.
    """
    # Copy the original signals to avoid modifying the input array
    filtered_signals = np.copy(emg_signals)
    
    for i in range(emg_signals.shape[0]):  # Iterate over each row (channel)
        channel_mean = np.mean(emg_signals[i, :])
        channel_std = np.std(emg_signals[i, :])
        
        lower_bound = channel_mean - std_threshold * channel_std
        upper_bound = channel_mean + std_threshold * channel_std
        # Identify indices of values that are outliers
        lower_outlayers = np.where((emg_signals[i, :] < lower_bound))
        upper_outlayers = np.where((emg_signals[i, :] > upper_bound))
        
        filtered_signals[i, lower_outlayers] = lower_bound
        filtered_signals[i, upper_outlayers] = upper_bound

    return filtered_signals


def uniform_continuity_clean(emg_data, window_size, step_size_ratio=1, std_threshold=2):
    # Length of the signal
    nrows = emg_data.shape[0]
    ncols = emg_data.shape[1]
    step_size = int(window_size*step_size_ratio)
    
    # Sliding window approach
    for r in range(nrows):
        signal = emg_data[r]
        for start in range(0, ncols - window_size + 1, step_size):
            end = start + window_size
            window = signal[start:end]
            mean = np.mean(window)
            std = np.std(window)
            lower_bound = mean - std_threshold * std
            upper_bound = mean + std_threshold * std
            
            # Check each value in the window
            for i in range(start, end):
                if signal[i] < lower_bound:
                    signal[i] = lower_bound
                elif signal[i] > upper_bound:
                    signal[i] = upper_bound
        emg_data[r] = signal
    
    return emg_data