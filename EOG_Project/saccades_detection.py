import numpy as np
import matplotlib.pyplot as plt
from differentiation import differentiate


def _smooth_speed(speed, fs, win_ms=5):
    """Mean moving window smoothing on |v| to tame derivative noise."""
    n = max(1, int(win_ms * fs / 1000))
    if n % 2 == 0:
        n += 1
    if speed.size < n:
        return speed
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(speed, n).mean(axis=1)
    pad = n // 2
    return np.pad(w, (pad, pad), mode='edge')

def _robust_thr(speed, k):
    """Robust threshold: median + k * 1.4826 * MAD (fallback to std if MAD==0)."""
    med = np.median(speed)
    mad = np.median(np.abs(speed - med))
    sigma = 1.4826 * mad if mad > 0 else speed.std(ddof=1)
    return med + k * sigma


def find_saccades_on_channel(signal, fs, method='l11', k=3.5, low_frac=0.75, min_dur_s=0.012,
                             *,  # only keyword below (keeps old calls valid)
                              robust=True, smooth_ms=5, refractory_s=0.025,
                              veto_speed=None, veto_factor=5.0):
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
    
    #threshold = speed.mean() + k * speed.std(ddof=1)        # μ + k·σ
    # smooth |v| a little (reduces choppiness of the derivative)
    if smooth_ms and smooth_ms > 0:
        speed = _smooth_speed(speed, fs, win_ms=smooth_ms)

    # robust threshold by default; fallback to mean+std if robust=False
    if robust:
        threshold = _robust_thr(speed, k)
    else:
        threshold = speed.mean() + k * speed.std(ddof=1)
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

    # --- refractory merge (e.g., 25 ms) ---
    if onsets and offsets and refractory_s and refractory_s > 0:
        refr = int(refractory_s * fs)
        merged_on, merged_off = [], []
        for a, b in zip(onsets, offsets):
            if not merged_on:
                merged_on.append(a); merged_off.append(b)
            else:
                if a - merged_off[-1] <= refr:
                    merged_off[-1] = max(merged_off[-1], b)
                else:
                    merged_on.append(a); merged_off.append(b)
        onsets  = merged_on
        offsets = merged_off

    """
    # --- optional blink veto using a provided vertical speed (veto_speed) ---
    if veto_speed is not None and len(onsets):
        vsp = np.abs(veto_speed)
        # robust vertical threshold, then a high veto level
        vthr = _robust_thr(vsp, k)
        veto_hi = veto_factor * vthr
        bad = vsp > veto_hi

        # pad ±100 ms around bad samples
        pad = int(0.1 * fs)
        if pad > 0:
            from numpy import convolve, ones
            bad = convolve(bad.astype(int), ones(2 * pad + 1, dtype=int), mode='same') > 0

        keep_on, keep_off = [], []
        for a, b in zip(onsets, offsets):
            if not bad[a:b + 1].any():
                keep_on.append(a); keep_off.append(b)
        onsets, offsets = keep_on, keep_off

    """
    onsets  = np.array(onsets,  int)
    offsets = np.array(offsets, int)
    return {'velocity': v, 'speed': speed, 'thr_high': threshold, 'thr_low': low, 'onsets': np.array(onsets, int), 'offsets': np.array(offsets, int)}



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

