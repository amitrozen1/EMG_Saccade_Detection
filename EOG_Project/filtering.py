import pickle
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


class Annotation:
    def __init__(self) -> None:
        self.Time = None


class Chunk:
    def __init__(self) -> None:
        self.Start = None
        self.End = None
        self.data = None
        self.Data = None

def load_pkl_file(path):
    with open(path, 'rb') as f:
        return _ChunkAwareUnpickler(f).load()


class _ChunkAwareUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'Chunk':
            return Chunk
        if module == '__main__' and name == 'Annotation':
            return Annotation
        return super().find_class(module, name)

def get_emg_signal_from_chunk(chunk, channel_index):
    signal = chunk.data[channel_index, :] 
    freq = getattr(chunk, 'freq', 500) 
    return signal, freq

def bandpass_filter(signal, fs, lowcut=0.1, highcut=40.0, order=4):
    x = np.asarray(signal, dtype=float)
    x = interp_nans(x) 
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)

def downsample_signal(signal, og_fs, target_fs):
    factor = og_fs / target_fs
    return resample_poly(signal, up=1, down=int(factor))

def interp_nans(x):
    # Quick linear interpolation for missing data points
    x = np.asarray(x, dtype=float)
    nans = np.isnan(x)
    if not nans.any():
        return x
        
    non_nans = ~nans
    x[nans] = np.interp(np.where(nans)[0], np.where(non_nans)[0], x[non_nans])
    return x

def lowpass_filter(signal, fs, cutoff=40.0, order=4):
    sos = butter(order, cutoff/(fs/2), btype='lowpass', output='sos')
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))