import pickle
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt  


def load_pkl_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_emg_signal_from_chunk(chunk, channel_index):
    signal = chunk.data[channel_index, :]   # Extract all the signal time points for the specified channel index
    freq = getattr(chunk, 'freq', 500)  # Default frequency if not specified in chunk
    return signal, freq


def bandpass_filter(signal, fs, lowcut=0.1, highcut=40.0, order=4):
    """    
    Args:
        signal (np.array): Input signal.
        fs (float): Sampling frequency (Hz).
        lowcut (float): Lower cutoff frequency (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        order (int): Filter order.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = interp_nans(x)  # Interpolate NaNs to avoid issues with filtering
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid band: low={lowcut}Hz, high={highcut}Hz, fs={fs}")

    sos = butter(order, [low, high], btype='bandpass', output='sos')
    y = sosfiltfilt(sos, x)
    
    return y


def downsample_signal(signal, og_fs, target_fs):
    factor = og_fs / target_fs
    if factor != int(factor):
        raise ValueError(f"Downsampling factor must be integer, got {factor}")
    return resample_poly(signal, up=1, down=int(factor))


def interp_nans(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    
    maskFinite = ~np.isnan(x) # Create a mask for finite values, True for finite, False for NaN or Inf
    if maskFinite.all(): # If all values are finite, no need to interpolate
        return x
    if maskFinite.sum() < 2: # Not enough points to interpolate
        return np.zeros_like(x, dtype=float)
    
    interp_signal = x.copy()
    interp_signal[~maskFinite] = np.interp(np.where(~maskFinite)[0], np.where(maskFinite)[0], x[maskFinite])
    return interp_signal

def lowpass_filter(signal, fs, cutoff=40.0, order=4):
    """    
    Args:
        signal (np.array): Input signal.
        fs (float): Sampling frequency (Hz).
        cutoff (float): Cutoff frequency (Hz).
        order (int): Filter order.
    """

    sos = butter(order, cutoff/(fs/2), btype='lowpass', output='sos')
    y = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    
    return y
