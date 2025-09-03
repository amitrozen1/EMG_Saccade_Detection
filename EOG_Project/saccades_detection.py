import numpy as np
import matplotlib.pyplot as plt
from differentiation import differentiate

def find_saccades_on_channel(signal, fs, method='l11', k=3.0, low_frac=0.3, min_dur_s=0.01):
    """
    Detect saccades in a given signal using velocity threshold method.

    Args:
        signal (array-like): The input signal to analyze.
        fs (int): Sampling frequency of the signal.
        method (str, optional): Differentiation method to use. Defaults to 'l11'.
        k (float, optional): Parameter for statistical differentiatio (μ + k·σ)). 
        low_frac (float, optional): Fraction of the high threshold to define a low threshold. 
        min_dur_s (float, optional): Minimum duration of a saccade in seconds. defaults to 10ms.

    Returns:
        _type_: _description_
    """


    time_step = 1.0 / fs
    v = differentiate(signal.astype(np.float64), step=time_step, method=method)     # Casting to float64 reduces numerical instability in Numba

    speed = np.abs(v)   # we're intrested in the speed of the signal, not the direction
    
    threshold = speed.mean() + k * speed.std(ddof=1)        # μ + k·σ
    low = low_frac * threshold


    cand = speed > threshold    # candidate saccades are where speed exceeds the threshold
    idx = np.flatnonzero(cand)  # indices of those candidate saccades
    onsets, offsets = [], []
    if idx.size:
        breaks = np.where(np.diff(idx) > 1)[0]  # Find positions where there is a gap (>1) between consecutive True indices in 'idx'
        starts = np.r_[0, breaks + 1]   # Start indices (in 'idx') for each continuous True segment
        ends   = np.r_[breaks, idx.size - 1]    # End indices (in 'idx') for each continuous True segment


        for s, e in zip(idx[starts], idx[ends]):

            onset = s
            while onset > 0 and speed[onset-1] >= low:
                onset -= 1
            offset = e
            last = len(speed) - 1
            while offset < last and speed[offset+1] >= low:
                offset += 1

            if (offset - onset) * time_step >= min_dur_s:
                onsets.append(onset)
                offsets.append(offset)

    return {
        'velocity': v,
        'speed': speed,
        'thr_high': threshold,
        'thr_low': low,
        'onsets': np.array(onsets, int),
        'offsets': np.array(offsets, int),
    }



def plot_velocity_with_method(signal, fs, method='l11', threshold=None, title=None):
   
    print("signal dtype/shape:", signal.dtype, signal.shape)
    print("signal finite?  ", np.isfinite(signal).all(), " min/max:", np.nanmin(signal), np.nanmax(signal))
    # calculate velocity
    h = 1.0 / fs
    velocity = differentiate(signal, h, method)
    print("velocity finite? ", np.isfinite(velocity).all(),
          " min/max:", np.nanmin(velocity), np.nanmax(velocity))
    print("velocity any nonzero? ", np.any(np.abs(velocity) > 0))
    speed = np.abs(velocity)

    # time axis
    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(t, velocity, label=f'Velocity ({method})')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [°/s]')
    if title:
        plt.title(title)
    plt.grid(True)

    # highlight points above threshold if given
    if threshold is not None:
        idx = np.where(speed > threshold)[0]
        plt.scatter(t[idx], velocity[idx], s=12, c='red', label=f'|v| > {threshold}')

    plt.legend()
    plt.tight_layout()
    plt.show()


# plot_velocity_with_method(rec.Y, fs=rec.sampling_frequency, method='l11', threshold=20, title="Velocity with l11 differentiation")

