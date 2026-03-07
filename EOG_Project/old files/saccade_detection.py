"""
Saccade detection for EOG-like signals.
Based loosely on Engbert & Kliegl (2003).
"""

import numpy as np
import matplotlib.pyplot as plt
from differentiation import differentiate

epsilon = 1e-12

def smooth_ma(x, win=5):
    """Simple moving-average smoothing."""
    x = np.asarray(x, float)
    if win <= 1:
        return x
    k = int(win) | 1  # make sure k is odd
    ker = np.ones(k, float) / k
    return np.convolve(x, ker, mode='same')


def engbert_sigma_hat(v):
    """Robust std estimation (median-based) as in Engbert & Kliegl."""
    v = np.asarray(v, float)
    med = np.median(v)
    return np.sqrt(np.median((v - med) ** 2))


def detect_saccades_engbert_2d(vx, vy, fs, eta=6.0, smooth_win=5,
                               min_samples=10, max_ms=None,
                               refractory_ms=0,
                               use_hysteresis=False, edge_frac=0.3):
    """
    2D saccade detection using Engbert-style elliptic threshold:
    S = (vx / gamma_x)^2 + (vy / gamma_y)^2 > 1.
    vx – horizontal velocity, vy – vertical velocity.
    """
    vx = smooth_ma(vx, smooth_win)
    vy = smooth_ma(vy, smooth_win)

    sigx = engbert_sigma_hat(np.abs(vx))
    sigy = engbert_sigma_hat(np.abs(vy))
    gamx, gamy = eta * sigx + epsilon, eta * sigy + epsilon # avoid zero division

    S = (vx / gamx) ** 2 + (vy / gamy) ** 2
    th = 1.0
    edge = (edge_frac * th) if use_hysteresis else th

    n = len(S)
    events = []
    i = 0
    max_samp = int(max_ms * fs / 1000) if max_ms else None
    refrac = int(refractory_ms * fs / 1000)

    while i < n:
        if S[i] > th:
            # peak index
            pk = i
            while pk + 1 < n and S[pk + 1] >= S[pk]:
                pk += 1

            # onset (go down to edge)
            onset = pk
            while onset > 0 and S[onset - 1] >= edge:
                onset -= 1

            # offset (forward until edge)
            offset = pk
            while offset + 1 < n and S[offset + 1] >= edge:
                offset += 1

            dur = offset - onset + 1
            if dur >= min_samples and (max_samp is None or dur <= max_samp):
                events.append((onset, pk, offset))

            i = offset + refrac if refrac > 0 else offset + 1
        else:
            i += 1

    return events

def classify_direction_by_angle(h_vel, v_vel, onset, pk, offset):
    vx = h_vel[pk]
    vy = v_vel[pk]

    theta = np.degrees(np.arctan2(vy, vx))

    if abs(theta) < 30:
        return "H"
    elif abs(theta) > 60:
        return "V"
    else:
        return "OBL"

def detect_blinks_standard(v_sig, h_sig, fs,
                           k_vert=4.0, k_horiz=2.0,
                           min_ms=50, max_ms=250):
    """
    Amplitude-based blink detection:
    - large vertical deflection (VEOG)
    - small horizontal deflection (HEOG)
    - duration within [min_ms, max_ms]
    """
    v_sig = np.asarray(v_sig, float)
    h_sig = np.asarray(h_sig, float)

    v_abs = np.abs(v_sig)
    h_abs = np.abs(h_sig)

    # robust vertical threshold (median + k * MAD)
    v_med = np.median(v_abs)
    v_mad = np.median(np.abs(v_abs - v_med)) + 1e-12
    v_thr = v_med + k_vert * v_mad

    # robust horizontal max (קטן יחסית – כדי לוודא שזה באמת מצמוץ ולא סקאדה אלכסונית)
    h_med = np.median(h_abs)
    h_mad = np.median(np.abs(h_abs - h_med)) + 1e-12
    h_thr = h_med + k_horiz * h_mad

    # candidate samples: big VEOG, small HEOG
    mask = (v_abs > v_thr) & (h_abs < h_thr)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    # duration constraints
    min_samp = int(min_ms * fs / 1000.0)
    max_samp = int(max_ms * fs / 1000.0)

    blinks = []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx.size - 1]

    for s, e in zip(idx[starts], idx[ends]):
        dur = e - s + 1
        if min_samp <= dur <= max_samp:
            blinks.append((s, e))

    return blinks

def remove_saccades_overlapping_blinks(saccades, blinks):
    """Return only saccades that do NOT overlap any blink interval."""
    if not blinks or not saccades:
        return saccades

    clean = []
    # הופכים את רשימת המצמוצים למקטעים (onset–offset)
    blink_intervals = [(b[0], b[1]) for b in blinks]

    for onset, pk, offset in saccades:
        overlaps = False
        for b_on, b_off in blink_intervals:
            # יש חפיפה בזמן?
            if not (offset < b_on or onset > b_off):
                overlaps = True
                break
        if not overlaps:
            clean.append((onset, pk, offset))
    return clean


def plot_velocity_with_method(signal, fs, method='l11', threshold=None, title=None):
    """Plot velocity (and optionally mark points above threshold)."""
    print("signal dtype/shape:", signal.dtype, signal.shape)
    print("signal finite? ", np.isfinite(signal).all(),
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
    plt.ylabel('Velocity [a.u.]')
    if title:
        plt.title(title)
    plt.grid(True)

    if threshold is not None:
        idx = np.where(speed > threshold)[0]
        plt.scatter(t[idx], velocity[idx], s=12,
                    label=f'|v| > {threshold}')

    plt.legend()
    plt.tight_layout()
    plt.show()
