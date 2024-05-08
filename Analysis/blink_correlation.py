from Analysis.blinking_for_shoval import *
from Analysis.blink_detection import *

def blink_correlation_rate_based(rated_blinks_participant1, rated_blinks_participant2, delay=50, rate_based=True):
    """
    Calculate the correlation between the blinks of two participants
    The correlation is based on the rate of the blinks
    rated_blinks_participant1 - a list of the blinks of the first participant
    rated_blinks_participant2 - a list of the blinks of the second participant
    delay - the maximal delay between the blinks of the two participants
    rate_based - if True, the correlation is based on the rate of the blinks
    returns the correlation between the two participants - between 0 and 1
    """

    total_matches = 0
    max_possible_matches = min(len(rated_blinks_participant1), len(rated_blinks_participant2)) 
    
    for blink1 in rated_blinks_participant1:
        start_time_1 = blink1[0][0]
        rate1 = blink1[1]
        for blink2 in rated_blinks_participant2:
            start_time_2 = blink2[0][0]
            rate2 = blink2[1]

            if abs(start_time_1 - start_time_2) < delay:
                if rate_based:
                    total_matches += 1 - abs(rate1 - rate2)
                else:
                    total_matches += 1
                break
    
    correlation_index = total_matches / max_possible_matches if max_possible_matches > 0 else 0
    return correlation_index

 # Convert spike times to phase
def spike_times_to_phase(spike_times):
    # Simplified example: Assume phase progresses linearly between spikes
    phases = np.linspace(0, 2*np.pi, len(spike_times), endpoint=False)
    return phases
    
def blink_correlation_phase_based(blinks_participant1, blinks_participant2):
    """
    Calculate the correlation between the blinks of two participants
    The correlation is based on the frequency of the blinks
    blinks_participant1 - a list of the blinks of the first participant - spike train
    blinks_participant2 - a list of the blinks of the second participant - spike train
    The algorithm is based on PLV (Phase Locking Value) - a measure of phase synchronization
    returns the correlation between the two participants - between 0 and 1
    """

    phase1 = spike_times_to_phase(blinks_participant1)
    phase2 = spike_times_to_phase(blinks_participant2) 
    # Calculate phase differences
    phase_diff = np.exp(1j * (phase1 - phase2))
    # Calculate PLV
    plv = np.abs(np.mean(phase_diff))
    return plv

def correlation_main(first_blink, second_blink, single_component_signal1, single_component_signal2, plv=False, rate_based=False):
    """
    The main function for the correlation analysis
    """
    win1 = create_filter(first_blink, 4)
    spike_train_1 = convert_patten_to_spike_train(single_component_signal1, win1, 4000)
    win2 = create_filter(second_blink, 4)
    spike_train_2 = convert_patten_to_spike_train(single_component_signal2, win2, 4000)
    
    # Calculate the correlation between the blinks of the two participants
    if plv:
        correlation = blink_correlation_phase_based(spike_train_1, spike_train_2)
    else:
        rated_blink1 = blink_detection(single_component_signal1, original_frequency, first_blink)
        rated_blink2 = blink_detection(single_component_signal2, original_frequency, second_blink)
        correlation = blink_correlation_rate_based(rated_blink1, rated_blink2, delay=50, rate_based=rate_based)
    return correlation

    