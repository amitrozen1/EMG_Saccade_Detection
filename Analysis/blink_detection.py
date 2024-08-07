# Detection of blinks in the data
# Rating the blinks in the data

from Analysis.blinking_for_shoval import *

def blink_peaks_detect(one_blink, blinking_component_signal, height, Flag, hz=4000):
    win1 = one_blink
    # spike_train, height1 = convert_patten_to_spike_train(blinking_component_signal, win1, hz, height, Flag)
    # to do: instead of getting one blink, find a blink (using extract_blink_from_vector) 
    # blink = extract_blink_from_vector(blinking_component_signal)

    spike_train, height1 = convert_patten_to_spike_train(blinking_component_signal, win1, hz, height, Flag)


    binary_blinks_list = []
    
    for i in range(len(spike_train)):
        if spike_train[i] == 1:
            binary_blinks_list.append(i)
    
    print("binary_blinks_list: ", binary_blinks_list)
    return binary_blinks_list, height1

def create_filter(sig, fft_order):
    """
        creates a filter of a pattern of a participant using FFT in order to recognize it in a signal
    :param sig: EMG signal of a participant
    :param fft_order: the requested number of Fourier coefficients to calculate for the filter
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a filter of a pattern
    """
    n = len(sig)
    fftsig = np.fft.fft(sig)
    np.put(fftsig, range(fft_order + 1, n), 0.0)
    win = np.fft.ifft(fftsig)
    win1 = np.real(np.fft.ifft(fftsig)) / np.sqrt(np.sum(np.square(win)))
    win2 = win1 - np.mean(win1)
    plt.figure(figsize=(9,5))
    plt.rcParams['font.size'] = '16'
    plt.xlabel("fs*sec filter", fontsize=16)
    plt.ylabel("Amplitude filter", fontsize=16)
    plt.plot(win1)
    plt.show()
    return win1

def normalize_signal(sig):
    """
        normalizing a given signal
    :param sig: a signal of a participant
    """
    return sig/np.sqrt(np.sum(np.power(sig,2), axis=0))

def norm_stand(sig):
    """
        normalizing and standardizing a given signal
    :param signal: signal of a participant
    """
    normalized_sig = (1 / np.max(np.abs(sig))) * sig
    mean = np.mean(normalized_sig)
    std = np.std(normalized_sig, axis=0)
    normalized_standardized_sig = (normalized_sig - mean) / std
    return normalized_standardized_sig


def convert_patten_to_spike_train(sig, filter, fs, height=0.3, Flag=False):
    """
     gets an EMG signal and a blinking pattern filter and converts the signal to a spike train of blinks
    :param sig: EMG signal of a participant
    :param filter: a blinking pattern filter
    :param fs: sampling frequency of the measuring device
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a spike train of blinks
    """
    # reduce moving mean from the signal in the length of filter
    # sig = sig[0:-len(filter)+1] - np.convolve(sig, np.ones(len(filter)) / len(filter), mode='valid')
    

    # #ploting the signal before normalization
    # plt.figure(figsize=(9,5))
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("signal before normalization", fontsize=16)
    # plt.plot(sig)
    # plt.show()
    sig = norm_stand(sig)
    # sig = normalize_signal(sig)


    # # ploting the signal after normalization
    # plt.figure(figsize=(9,5))
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("signal after normalization", fontsize=16)
    # plt.plot(sig)
    # plt.show()
    
    N = len(sig)
    t = np.linspace(1 / fs, N, N)
    corr = signal.correlate(sig, filter, mode='same') / len(filter)

    if Flag == True:    
        plt.figure(figsize=(9,5))
        plt.rcParams['font.size'] = '16'
        plt.xlabel("correlated signal", fontsize=16)
        # positive corr - only the positive values of the correlation (above 0)
        positive_corr = np.maximum(corr, 0)
        avr = np.average(positive_corr)
        plt.plot(corr)
        plt.axhline(y=avr, color='r', linestyle='--')
        plt.legend(["signal" ,"average"])
        plt.show()

    # asks the user to input the height of the peaks
    if Flag == True:
        height = float(input("Enter the height of the peaks: "))
    
    peaks = signal.find_peaks(corr, height=height, distance=20)

    spike_train = np.zeros_like(t)
    spike_train[peaks[0]] = 1
    plt.figure(figsize=(8, 5))
    plt.rcParams['font.size'] = '16'
    plt.plot(t, np.real(corr))
    plt.plot(peaks[0], peaks[1]['peak_heights'], '*')
    plt.xlabel("fs*sec peaks", fontsize=16)
    plt.ylabel("cross-correlation peaks", fontsize=16)
    plt.show()
    plt.close()
    plt.show()
    plt.close()
    """
    plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.xlabel("fs*sec peaksss", fontsize=14)
    plt.ylabel("occurrence of a spike", fontsize=14)
    plt.stem(t, spike_train)
    plt.show()
    plt.close()
    """
    return spike_train, height

def find_blink_duration(signal, binary_blinks_list, threshold=0.05):
    """
        For each blink find its ending (the duraiton of the blink)
    gets as input a list with the timing (binary, the begining of each blink) of the blinks in the signal and the signal
    :param threshold: the minimum difference between two minima to be considered still part of the blink
    """
    window_length = 5
    blinks_duration = []
    end_threshold = 0.1
    for blink in binary_blinks_list:
        # when signal is the single component (blinking) signal - expecting a vector over time 
        t_start = blink
        t_end = blink
        t_peak = blink
        i = 1

        # find the peak of the blink
        while signal[t_peak] <= signal[t_peak + i*window_length]:
            if t_peak + (i+1)*window_length < len(signal):
                if signal[t_peak] < signal[-1]:
                    t_peak = len(signal) - 1
                break
                
            t_peak = blink + i*window_length
            i += 1 

        got_to_normal_flag = False
        minima = signal[t_peak]
        while not got_to_normal_flag:
            # checking if we got to the end of the signal
            if blink + i*window_length > len(signal):
                new_minima = signal[-1]
                t_end = len(signal)
                got_to_normal_flag = True
            else:
                new_minima = signal[blink + i*window_length]

            if new_minima < minima:
                minima = new_minima
                i += 1
            # at the second part of the blink there is usually a rise in the signal which is part of the blink
            elif new_minima - signal[t_start] > end_threshold:
                i += 1
            else:
               t_end = blink + i*window_length
               got_to_normal_flag = True
        
        blinks_duration.append([blink, t_start, t_end])

    return blinks_duration      

def rate_blinks(blinks_timing):
    """
        For each blink rate it between 0-1
    The shortest blink (of the subject) will be rated zero while the longest will be rated 1
    blinks_timing - a list of blinks with their start time and end time) 
    returns a list of the blinks with their rate
    """
    shortest = 0
    longest = 0
    blinks_duration = []
    for blink in blinks_timing:
        length = blink[2] - blink[1]
        blinks_duration.append([blink, length])
        if length < shortest | shortest == 0:
            shortest = length
        if length > longest:
            longest = length
    rated_blinks = []
    for blink in blinks_duration:
        rate = (blink[1] - shortest)/(longest - shortest)
        rated_blinks.append([blink, rate])

    return rated_blinks

def blink_detection(single_component_signal, fs, one_blink=None):
    """
        Gets a single component signal and returns a list of the blinks in the signal with their rates 
    """
    # filtered_signal = butter_bandpass_filter(single_component_signal, lowcut=, highcut=, fs)
    # find (binary) the blinks
    # one_blink = extract_blink_from_vector(single_component_signal)
    if one_blink is None:    
        one_blink = single_component_signal[7175:7215]
    plt.figure(figsize=(9,5))
    plt.rcParams['font.size'] = '16'
    plt.xlabel("one blink", fontsize=16)
    plt.plot(one_blink)
    plt.show()
    
    binary_blinks_list = blink_peaks_detect(one_blink, single_component_signal)
    # after finding all peaks of blinking find their duration
    blinks_timing = find_blink_duration(single_component_signal, binary_blinks_list)
    # for each blink rate it between 0-1, zero for the shortest, 1 for the longest
    print("got to rate_blinks")
    rated_blinks = rate_blinks(blinks_timing)
    
    return rated_blinks

def main():
    # extracting the blinking component from the data
    vector, chunks, comp = extract_main()
    
    # rating the blinks in the data of the participant
    rated_blinks = blink_detection(vector, original_frequency)
    print(rated_blinks)

    # comparing the blinks of two participants
