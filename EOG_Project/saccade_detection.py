import numpy as np

epsilon = 1e-12

def smooth_ma(x, win=5):
    """Applies a moving average filter to smooth the signal."""
    x = np.asarray(x, float)
    if win <= 1: return x
    k = int(win) | 1
    ker = np.ones(k, float) / k
    return np.convolve(x, ker, mode='same')

def engbert_sigma_hat(v):
    """Estimates the velocity noise threshold based on the median absolute deviation."""
    v = np.asarray(v, float)
    med = np.median(v)
    return np.sqrt(np.median((v - med) ** 2))

def detect_saccades_engbert_2d(vx, vy, fs, eta_h=5.0, eta_v=10.0, smooth_win=5,
                               min_samples=10, max_ms=150, refractory_ms=0,
                               use_hysteresis=False, edge_frac=0.3):
    """
    Detects saccades using a modified 2D Engbert & Kliegl (2003) algorithm 
    with asymmetric velocity thresholds.
    """
    vx = smooth_ma(vx, smooth_win)
    vy = smooth_ma(vy, smooth_win)

    sigx = engbert_sigma_hat(np.abs(vx))
    sigy = engbert_sigma_hat(np.abs(vy))
    
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
    """Classifies saccade direction (H, V, OBL) based on the net displacement vector."""
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
    Detects blinks based on robust amplitude thresholds:
    Requires large VEOG deflection and minimal HEOG deflection.
    """
    v_sig = np.asarray(v_sig, float)
    h_sig = np.asarray(h_sig, float)

    v_abs = np.abs(v_sig)
    h_abs = np.abs(h_sig)

    v_med = np.median(v_abs)
    v_mad = np.median(np.abs(v_abs - v_med)) + epsilon
    v_thr = v_med + k_vert * v_mad

    h_med = np.median(h_abs)
    h_mad = np.median(np.abs(h_abs - h_med)) + epsilon
    h_thr = h_med + k_horiz * h_mad

    mask = (v_abs > v_thr) & (h_abs < h_thr)
    idx = np.flatnonzero(mask)
    
    if idx.size == 0:
        return []

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

def remove_saccades_overlapping_blinks(saccades, blinks, fs=500, pre_buffer_ms=20, post_buffer_ms=100):
    """
    Filters out saccades that overlap with blink intervals.
    Applies an asymmetric temporal buffer to mask brief pre-blink artifacts 
    and prolonged trailing blink recovery (crosstalk/capacitive discharge) artifacts.
    """
    if not blinks or not saccades:
        return saccades

    clean = []
    pre_samples = int(pre_buffer_ms * fs / 1000.0)
    post_samples = int(post_buffer_ms * fs / 1000.0)

    # Asymmetric buffering: tighter before the blink, longer after to account for recovery
    blink_intervals = [(max(0, b[0] - pre_samples), b[1] + post_samples) for b in blinks]

    for onset, pk, offset in saccades:
        overlaps = False
        for b_on, b_off in blink_intervals:
            # Check for any temporal overlap between the saccade and padded blink
            if not (offset < b_on or onset > b_off):
                overlaps = True
                break
        if not overlaps:
            clean.append((onset, pk, offset))
            
    return clean