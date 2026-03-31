"""
Saccade detection for EOG-like signals.
Based loosely on Engbert & Kliegl (2003).
"""

import numpy as np
import matplotlib.pyplot as plt
from differentiation import differentiate

epsilon = 1e-12

def smooth_ma(x, win=5):
    x = np.asarray(x, float)
    if win <= 1: return x
    k = int(win) | 1
    ker = np.ones(k, float) / k
    return np.convolve(x, ker, mode='same')

def engbert_sigma_hat(v):
    v = np.asarray(v, float)
    med = np.median(v)
    return np.sqrt(np.median((v - med) ** 2))


def detect_saccades_engbert_2d(vx, vy, fs, eta_h=5.0, eta_v=10.0, smooth_win=15,
                               min_samples=10, max_ms=150, refractory_ms=0,
                               use_hysteresis=False, edge_frac=0.3): 
    
    vx = smooth_ma(vx, smooth_win)
    vy = smooth_ma(vy, smooth_win)

    sigx = engbert_sigma_hat(np.abs(vx))
    sigy = engbert_sigma_hat(np.abs(vy))
    
    # asymmetric thresholds for horizontal and vertical components
    gamx = eta_h * sigx + epsilon
    gamy = eta_v * sigy + epsilon

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
            pk = i
            while pk + 1 < n and S[pk + 1] >= S[pk]: pk += 1
            onset = pk
            while onset > 0 and S[onset - 1] >= edge: onset -= 1
            offset = pk
            while offset + 1 < n and S[offset + 1] >= edge: offset += 1

            dur = offset - onset + 1
            if dur >= min_samples and (max_samp is None or dur <= max_samp):
                events.append((onset, pk, offset))

            i = offset + refrac if refrac > 0 else offset + 1
        else:
            i += 1

    return events

def classify_direction_by_angle(h_vel, v_vel, onset_idx, offset_idx):
    # Classifies saccade direction using the net displacement vector (integral of velocity over the saccade duration).
    dx = np.sum(h_vel[onset_idx : offset_idx + 1])
    dy = np.sum(v_vel[onset_idx : offset_idx + 1])

    theta = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))

    if abs(theta) < 45:
        return "H"
    elif abs(theta) > 60:
        return "V"
    else:
        return "OBL"

def detect_blinks_standard(v_sig, h_sig, fs, k_vert=3.0, k_horiz=5.0, min_ms=50, max_ms=250):
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
    v_mad = np.median(np.abs(v_abs - v_med)) + epsilon
    v_thr = v_med + k_vert * v_mad

    # robust horizontal max
    h_med = np.median(h_abs)
    h_mad = np.median(np.abs(h_abs - h_med)) + epsilon
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

    blink_intervals = [(b[0], b[1]) for b in blinks]

    for onset, pk, offset in saccades:
        overlaps = False
        for b_on, b_off in blink_intervals:
            # Check if saccade interval [onset, offset] overlaps with blink interval [b_on, b_off]
            if not (offset < b_on or onset > b_off):
                overlaps = True
                break
        if not overlaps:
            clean.append((onset, pk, offset))
    return clean

