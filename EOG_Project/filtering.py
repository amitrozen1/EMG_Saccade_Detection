import pickle
import numpy as np
from scipy.signal import butter, filtfilt, decimate

def load_pkl_file(path):
    """
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except ModuleNotFoundError as e:
            print("שגיאה - חסר מודול:", e)
            f.seek(0)  # מחזיר את הקובץ להתחלה
            import pickletools
            pickletools.dis(f)

    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_emg_signal_from_chunk(chunk, channel_index):
    signal = chunk.data[channel_index, :]   # Extract all the signal time points for the specified channel index
    freq = getattr(chunk, 'freq', 500)  # Default frequency if not specified in chunk
    return signal, freq


def bandpass_filter(signal, fs, lowcut=0.1, highcut=10.0, order=4):
    """
    Apply a bandpass filter to a signal.
    
    Args:
        signal (np.array): Input signal.
        fs (float): Sampling frequency (Hz).
        lowcut (float): Lower cutoff frequency (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        order (int): Filter order.
        
    Returns:
        np.array: Filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)

    return filtered



def downsample_signal(signal, factor, zero_phase=True):
    """
    Downsample signal using decimate (includes anti-aliasing filter).
    
    Args:
        signal (np.array): Input signal.
        factor (int): Downsampling factor.
        zero_phase (bool): If True, use filtfilt to avoid phase shift.
    
    Returns:
        np.array: Downsampled signal.
    """
    return decimate(signal, factor, ftype='iir', zero_phase=zero_phase)
