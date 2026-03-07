import numpy as np

def detect_saccades_velocity_threshold(signal, sampling_rate, velocity_threshold):

    velocity = np.gradient(signal) * sampling_rate  # approx derivative

    above_threshold = np.abs(velocity) > velocity_threshold # True if velocity exceeds threshold

    saccades = []
    in_saccade = False
    start_idx = 0

    for i, val in enumerate(above_threshold):  
        if val and not in_saccade:
            start_idx = i
            in_saccade = True
        elif not val and in_saccade:
            end_idx = i
            saccades.append((start_idx, end_idx))
            in_saccade = False

    # If still in a saccade at the end of the signal, close it
    if in_saccade:
        saccades.append((start_idx, len(signal)-1))

    return saccades, velocity
