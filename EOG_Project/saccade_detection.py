import numpy as np
from differentiation import differentiate 

def smooth_ma(x, win=5):
    x = np.asarray(x, float)
    if win <= 1: 
        return x
    k = int(win) | 1
    ker = np.ones(k, float) / k
    return np.convolve(x, ker, mode='same')

def engbert_sigma_hat(v):
    v = np.asarray(v, float)
    med_v = np.median(v)
    med_v2 = np.median(v ** 2)
    return np.sqrt(max(med_v2 - med_v ** 2, 0.0))

def detect_saccades_engbert_2d(vx, vy, fs, eta_h=5.0, eta_v=10.0, smooth_win=5, min_samples=10, max_ms=150, 
        refractory_ms=0, use_hysteresis=False, edge_frac=0.3):
    vx = smooth_ma(vx, smooth_win)
    vy = smooth_ma(vy, smooth_win)

    sigx = engbert_sigma_hat(vx)
    sigy = engbert_sigma_hat(vy)
    
    gamx = eta_h * sigx + 1e-12
    gamy = eta_v * sigy + 1e-12

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
            while pk + 1 < n and S[pk + 1] >= S[pk]: 
                pk += 1
            
            onset = pk
            while onset > 0 and S[onset - 1] >= edge: 
                onset -= 1
            
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

def classify_direction_by_angle(h_vel, v_vel, onset_idx, offset_idx):
    dx = np.sum(h_vel[onset_idx : offset_idx + 1])
    dy = np.sum(v_vel[onset_idx : offset_idx + 1])

    theta = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))

    if abs(theta) < 45:
        return "H"
    elif abs(theta) > 60:
        return "V"
    else:
        return "OBL"

def pick_best_heog(h1, h2, fs):
    # Chooses the channel with the larger velocity variability
    time_step = 1.0 / fs
    v1 = differentiate(h1.astype(np.float64), step=time_step, method='l11')
    v2 = differentiate(h2.astype(np.float64), step=time_step, method='l11')
    
    mad1 = np.median(np.abs(v1 - np.median(v1)))
    mad2 = np.median(np.abs(v2 - np.median(v2)))
    return h1 if mad1 > mad2 else h2

def get_local_maxima(x):
    x = np.array(x, dtype=float)
    if len(x) < 3:
        return np.array([], dtype=int)
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    return np.where(is_peak)[0] + 1

def get_fallback_threshold(x, k=3.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return med + k * (mad / 0.6745)

def remove_outliers(peaks):
    # Pettersson step: scale to [0,1] and remove top artifacts if jump >= 0.03
    peaks = np.sort(peaks)
    if len(peaks) < 3:
        return peaks
        
    p_min = peaks[0]
    p_max = peaks[-1]
    if (p_max - p_min) == 0:
        return peaks
        
    scaled = (peaks - p_min) / (p_max - p_min)
    
    stop_idx = len(peaks)
    for i in range(len(scaled) - 2, -1, -1):
        if (scaled[i + 1] - scaled[i]) < 0.03:
            break
        stop_idx = i + 1
        
    return peaks[:stop_idx]

def calculate_lse_threshold(x, min_peaks=20):
    idx = get_local_maxima(x)
    peaks = x[idx]
    peaks = np.sort(peaks[peaks > 0])
    
    if len(peaks) < min_peaks:
        return get_fallback_threshold(x)
        
    peaks = remove_outliers(peaks)
    N = len(peaks)
    
    # Paper requires splitting from N=10 to N/2
    if N < min_peaks or (N // 2) <= 10:
        return get_fallback_threshold(x)
        
    best_error = float('inf')
    best_split = 10
    x_axis = np.arange(N)
    
    for split_idx in range(10, N // 2):
        x1, y1 = x_axis[:split_idx], peaks[:split_idx]
        x2, y2 = x_axis[split_idx:], peaks[split_idx:]
        
        # Two-line least squares fit
        poly1 = np.polyfit(x1, y1, 1)
        poly2 = np.polyfit(x2, y2, 1)
        
        err1 = np.sum((y1 - np.polyval(poly1, x1)) ** 2)
        err2 = np.sum((y2 - np.polyval(poly2, x2)) ** 2)
        
        total_error = err1 + err2
        if total_error < best_error:
            best_error = total_error
            best_split = split_idx
            
    # Add 2% margin to the chosen split peak
    return 1.02 * peaks[best_split]

def detect_blinks(v_sig, v_diff, fs=500):
    v_sig = np.asarray(v_sig, float)
    v_diff = np.asarray(v_diff, float)

    v_signal_abs = np.abs(v_sig)
    v_diff_abs = np.abs(v_diff)

    blink_amp_threshold = max(
        calculate_lse_threshold(v_signal_abs),
        get_fallback_threshold(v_signal_abs, k=5.0)
    )

    blink_diff_threshold = max(
        calculate_lse_threshold(v_diff_abs),
        get_fallback_threshold(v_diff_abs, k=4.0)
    )
    
    # Get blink candidates based on vertical velocity
    diff_peaks = get_local_maxima(v_diff_abs)
    blink_candidates = []
    for p in diff_peaks:
        if v_diff_abs[p] > blink_diff_threshold:
            blink_candidates.append(p)
    
    min_blink_samples = int(0.09 * fs)        
    max_blink_samples = int(1.2 * fs)
    half_window = int(0.30 * fs)
    
    blinks = []
    n_samples = len(v_sig)
    skip_until = 0
    
    for cand in blink_candidates:
        if cand < skip_until:
            continue
            
        # Search window for actual amplitude peak around velocity peak
        win_start = max(0, cand - half_window)
        win_end = min(n_samples, cand + half_window)
        window_data = np.abs(v_sig[win_start:win_end])
        
        if len(window_data) == 0:
            continue
            
        actual_peak = win_start + np.argmax(window_data)
        peak_amp = np.abs(v_sig[actual_peak])
        
        if peak_amp <= blink_amp_threshold:
            continue
            
        # 2. Find overall blink boundaries (10% of local peak amplitude)
        bound = 0.10 * peak_amp
        
        onset = actual_peak
        while onset > 0 and np.abs(v_sig[onset - 1]) > bound:
            onset -= 1
            
        offset = actual_peak
        while offset + 1 < n_samples and np.abs(v_sig[offset + 1]) > bound:
            offset += 1
            
        blink_dur = offset - onset + 1
        if (blink_dur < min_blink_samples) or (blink_dur > max_blink_samples):
            continue
            
        # 3. Verify morphology: both positive and negative peaks in velocity
        velocity_window = v_diff[onset:offset+1]
        has_pos = np.any(velocity_window > blink_diff_threshold)
        has_neg = np.any(velocity_window < -blink_diff_threshold)
        
        if has_pos and has_neg:
            blinks.append([onset, offset])
            skip_until = offset + int(0.15 * fs)
            
    if len(blinks) == 0:
        return []
        
    # Merge overlapping detections
    merged = [blinks[0]]
    for i in range(1, len(blinks)):
        current_blink = blinks[i]
        last_merged = merged[-1]
        
        if current_blink[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current_blink[1])
        else:
            merged.append(current_blink)
            
    return [(b[0], b[1]) for b in merged]


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