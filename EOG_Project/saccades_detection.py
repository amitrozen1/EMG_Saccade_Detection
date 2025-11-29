"""
Saccade detection for EOG signals.
Based on Engbert & Kliegl (2003) method.
"""

import numpy as np
import matplotlib.pyplot as plt
from differentiation import differentiate


def smooth_ma(x, win=5):
    """Moving average smoothing"""
    x = np.asarray(x, float)
    if win <= 1:
        return x
    k = int(win) | 1  # make sure it's odd
    ker = np.ones(k, float) / k
    return np.convolve(x, ker, mode='same')


def engbert_sigma_hat(v):
    """Robust std estimation - from Engbert & Kliegl 2003"""
    v = np.asarray(v, float)
    med = np.median(v)
    return np.sqrt(np.median((v - med)**2))


def classify_direction_by_energy(h_vel, v_vel, eta=6.0, smooth_win=5):
    """Classify direction by energy ratio"""
    def smooth_ma_local(x, k=5):
        if k <= 1:
            return x
        k = int(k) if int(k) % 2 == 1 else int(k) + 1
        ker = np.ones(k) / k
        return np.convolve(x, ker, mode='same')

    hv = smooth_ma_local(h_vel, smooth_win)
    vv = smooth_ma_local(v_vel, smooth_win)

    # robust sigma
    def sigma_hat_abs(x):
        a = np.abs(x)
        med = np.median(a)
        return np.sqrt(np.median((a - med)**2) + 1e-12)

    sx = sigma_hat_abs(hv)
    sy = sigma_hat_abs(vv)
    
    nx = hv / (eta * sx + 1e-12)
    ny = vv / (eta * sy + 1e-12)

    # energy in each direction
    def energy_ratio(x, y, i0, i1):
        ex = np.sum(x[i0:i1+1]**2)
        ey = np.sum(y[i0:i1+1]**2)
        tot = ex + ey + 1e-12
        return ex/tot, ey/tot

    return energy_ratio


def detect_saccades_engbert_2d(vx, vy, fs, eta=6.0, smooth_win=5, min_samples=10, max_ms=None,
                                refractory_ms=0, use_hysteresis=False, edge_frac=0.3):
    """2D saccade detection - Engbert & Kliegl 2003"""
    vx = smooth_ma(vx, smooth_win)
    vy = smooth_ma(vy, smooth_win)

    sigx = engbert_sigma_hat(np.abs(vx))
    sigy = engbert_sigma_hat(np.abs(vy))
    gamx, gamy = eta * sigx, eta * sigy

    # elliptic criterion: S > 1 means saccade
    S = (vx / gamx)**2 + (vy / gamy)**2
    th = 1.0
    edge = (edge_frac * th) if use_hysteresis else th

    n = len(S)
    events = []
    i = 0
    max_samp = int(max_ms * fs / 1000) if max_ms else None
    refrac = int(refractory_ms * fs / 1000)

    while i < n:
        if S[i] > th:
            pk = i
            while pk + 1 < n and S[pk+1] >= S[pk]:
                pk += 1
            onset = pk
            while onset > 0 and S[onset-1] >= edge:
                onset -= 1
            offset = pk
            while offset + 1 < n and S[offset+1] >= edge:
                offset += 1
            dur = offset - onset + 1
            if dur >= min_samples and (max_samp is None or dur <= max_samp):
                events.append((onset, pk, offset))
            i = offset + refrac if refrac > 0 else offset + 1
        else:
            i += 1
    return events


def find_saccades_on_channel(signal, fs, method='l11', k=3.0, low_frac=0.3, min_dur_s=0.01):
    """Velocity threshold detection with hysteresis"""
    time_step = 1.0 / fs
    v = differentiate(signal.astype(np.float64), step=time_step, method=method)

    speed = np.abs(v)
    
    threshold = speed.mean() + k * speed.std(ddof=1)
    low = low_frac * threshold

    cand = speed > threshold
    idx = np.flatnonzero(cand)
    onsets, offsets = [], []
    
    if idx.size:
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, idx.size - 1]

        for s, e in zip(idx[starts], idx[ends]):
            # expand to low threshold
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


def find_saccades_adaptive(vel, fs, k_sigma=4.0, edge_frac=0.3,
                           min_ms=10, max_ms=40, refrac_ms=60):
    """Adaptive threshold with baseline estimation"""
    v = np.abs(vel)
    base = np.percentile(v, 20)  # baseline
    std = np.std(v[v < base*2]) + 1e-12
    th = base + k_sigma * std
    edge = edge_frac * th

    events = []
    i = 0
    n = len(v)
    min_samp = int(min_ms * fs / 1000.0)
    max_samp = int(max_ms * fs / 1000.0)
    refrac = int(refrac_ms * fs / 1000.0)

    while i < n:
        if v[i] > th:
            pk = i
            while pk+1 < n and v[pk+1] >= v[pk]:
                pk += 1
            
            onset = pk
            while onset > 0 and v[onset-1] >= edge:
                onset -= 1
            offset = pk
            while offset+1 < n and v[offset+1] >= edge:
                offset += 1
            
            dur = offset - onset + 1
            if min_samp <= dur <= max_samp:
                events.append((onset, pk, offset, th))
            
            i = offset + refrac
        else:
            i += 1
    return events


def plot_velocity_with_method(signal, fs, method='l11', threshold=None, title=None):
    """Plot velocity with threshold"""
    # debug prints
    print("signal dtype/shape:", signal.dtype, signal.shape)
    print("signal finite?  ", np.isfinite(signal).all(), 
          " min/max:", np.nanmin(signal), np.nanmax(signal))
    
    h = 1.0 / fs
    velocity = differentiate(signal, h, method)
    print("velocity finite? ", np.isfinite(velocity).all(),
          " min/max:", np.nanmin(velocity), np.nanmax(velocity))
    print("velocity any nonzero? ", np.any(np.abs(velocity) > 0))
    speed = np.abs(velocity)

    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(t, velocity, label=f'Velocity ({method})')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [°/s]')
    if title:
        plt.title(title)
    plt.grid(True)

    if threshold is not None:
        idx = np.where(speed > threshold)[0]
        plt.scatter(t[idx], velocity[idx], s=12, c='red', label=f'|v| > {threshold}')

    plt.legend()
    plt.tight_layout()
    plt.show()
