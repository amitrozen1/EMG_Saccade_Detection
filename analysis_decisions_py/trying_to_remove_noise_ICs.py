import numpy as np
import matplotlib.pyplot as plt
from General.init import *
from EDF.SharedIcaRunner import LoadComponents
from General.HelperFunctions import GetSessions, getWeightsFileNameWithPath, particiapnt_to_char, getPathsFromSessionFolder
from EDF.PlotHeatmap import get_heatmap
from EDF.EdfAnalyzer import EdfAnalyzer
from scipy import signal
import math
from EDF.ComponentValidator import get_noise_components_by_sort_w
import os
from scipy.stats import zscore


def plot_frequency_spectrum(signal, sampling_rate):
    """
    Plots the frequency spectrum of a given EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Compute the FFT (Fast Fourier Transform) of the signal
    fft_result = np.fft.fft(signal)
    n = len(signal)
    timestep = 1 / sampling_rate
    freq = np.fft.fftfreq(n, d=timestep)

    # Only take the positive frequencies
    n_half = n // 2
    freq = freq[:n_half]
    fft_result = np.abs(fft_result)[:n_half]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(freq, fft_result)
    plt.title("Frequency Spectrum of the EMG Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

    return freq, fft_result

def create_frequency_spectrum_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the frequency spectrum of a given EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Compute the FFT (Fast Fourier Transform) of the signal
    fft_result = np.fft.fft(signal)
    n = len(signal)
    timestep = 1 / sampling_rate
    freq = np.fft.fftfreq(n, d=timestep)

    # Only take the positive frequencies
    n_half = n // 2
    freq = freq[:n_half]
    fft_result = np.abs(fft_result)[:n_half]

    # Plotting
    ax.plot(freq, fft_result)
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    return freq, fft_result

def plot_frequency_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the frequency spectrum of a given EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Compute the FFT (Fast Fourier Transform) of the signal
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_frequency_spectrum_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()


def plot_time_domain_signal(signal, sampling_rate):
    """
    Plots the time-domain representation of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    time = np.arange(len(signal)) / sampling_rate

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.title("Time-Domain Representation of the EMG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    return time, signal

def create_time_domain_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the time-domain representation of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    time = np.arange(len(signal)) / sampling_rate

    # Plotting
    ax.plot(time, signal)
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    return time, signal

def plot_time_domain_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the time-domain representation of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_time_domain_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()


def create_signal_density_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the density of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Plotting
    ax.hist(signal, bins=100)
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")

    return signal

def plot_signal_density_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the density of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_signal_density_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()

def create_signal_density_welch_method_subplot(Y, sampling_rate, ax, is_noise = False):
    """
    Plots the density of an EMG signal using the Welch method.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """
    window = signal.get_window('hamming', 500)
    f_smooth, Pxx_den_smooth = signal.welch(Y, sampling_rate, window=window, noverlap=250, nperseg=500, nfft=512,
                                            scaling='density')

    # all
    len_window = int(len(Y) / 4) #Y.shape[1]
    window = signal.get_window('hamming', len_window)
    f, Pxx_den = signal.welch(Y, sampling_rate, window=window, nperseg=len_window,
                              nfft=2 ** math.ceil(math.log2(abs(len_window))), scaling='density')
    
    # Plotting
    ax.plot(f, Pxx_den, color='black')
    ax.plot(f_smooth, Pxx_den_smooth, color='blue', linewidth=2)

    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")

    return signal

def plot_signal_density_welch_method_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the density of an EMG signal using the Welch method.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_signal_density_welch_method_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()


def create_moving_window_rms_downsampled_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the moving window RMS of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate moving window RMS
    window_size = 0.05  # seconds
    window_size = int(window_size * sampling_rate)
    rms = np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode="same"))

    # Downsample
    downsample_factor = 100
    signal = signal[::downsample_factor]
    rms = rms[::downsample_factor]

    # Plotting
    ax.plot(signal, label="Signal")
    ax.plot(rms, label="RMS")
    ax.legend()
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    return signal

def create_moving_window_rms_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the moving window RMS of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate moving window RMS
    window_size = 0.05  # seconds
    window_size = int(window_size * sampling_rate)
    rms = np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode="same"))

    # Plotting
    ax.plot(signal, label="Signal")
    ax.plot(rms, label="RMS")
    ax.legend()
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    return signal

def plot_moving_window_rms_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the moving window RMS of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.
    """

    # Calculate time vector
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_moving_window_rms_downsampled_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()

def create_color_density_spectrum_subplot(signal, sampling_rate, ax, is_noise = False):
    """
    Plots the color density spectrum of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
    numpy array: The color density spectrum of the signal.
    """

    # Calculate the FFT (Fast Fourier Transform) of the signal
    fft_result = np.fft.fft(signal)
    n = len(signal)
    timestep = 1 / sampling_rate
    freq = np.fft.fftfreq(n, d=timestep)

    # Only take the positive frequencies
    n_half = n // 2
    freq = freq[:n_half]
    fft_result = np.abs(fft_result)[:n_half]

    # Plotting
    ax.scatter(freq, fft_result, c=freq, cmap="jet")
    if ( is_noise ):
        ax.set_title("noise")
    else:
        ax.set_title("signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    return fft_result

def plot_color_density_spectrum_subplots_of_all_components(components, sampling_rate, num_components, noise_ics = []):
    """
    Plots the color density spectrum of an EMG signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
    numpy array: The color density spectrum of the signal.
    """

    # Calculate time vector
    fig, axs = plt.subplots(num_components, 1, figsize=(10, 6))
    for i in range(num_components):
        create_color_density_spectrum_subplot(components[i], sampling_rate, axs[i], i in noise_ics)
    plt.show()

def integrate_abs_of_signal(signal, sampling_rate):
    """
    Calculates the integral of the absolute value of the signal.

    Parameters:
    signal (numpy array): The EMG signal (one IC).
    sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
    float: The integral of the absolute value of the signal.
    """

    return np.sum(np.abs(signal)) / sampling_rate

def get_a_list_of_numbers_from_user(user_text):
    """
    Gets a string from the user, and returns a list of numbers.
    The numbers are separated by spaces.

    Parameters:
    user_text (str): The text to show the user.

    Returns:
    list: A list of numbers.
    """

    user_input = input(user_text)
    user_input = user_input.split(" ")
    #remove empty strings:
    user_input = [int(x) for x in user_input if x != ""] 
    return user_input


def get_smile_data(components, chunks, p=None):
    """
    Gets the data of the smile chunks of a participant.

    Parameters:
    components (numpy array): The components of the participant.
    chunks (dict): The chunks of the participant.
    p (int): The participant. if none, assumes only data of one particiapnt was sent.

    Returns:
    numpy array: The data of the smile chunks of the participant.
    """
    if ( p!= None):
        smile_data = []
        for i in range(3):
            s = int(chunks[p][f"smile__{i}"].Start.Time)
            e = int(chunks[p][f"smile__{i}"].End.Time)
            smile_data.append(components[p][:, s:e])
        return np.concatenate(smile_data, axis=1)
    else:
        smile_data = []
        # for i in range(3):
        #     s = int(chunks[f"smile__{i}"].Start.Time)
        #     e = int(chunks[f"smile__{i}"].End.Time)
        #     smile_data.append(components[:, s:e])
        # return np.concatenate(smile_data, axis=1)
        s = int(chunks[f"smile__0"].Start.Time)
        e = int(chunks[f"smile__2"].End.Time)
        return components[:, s:e]


def shared_main(session, p):
    #load components for the first session:
    components, chunks = LoadComponents(data_path, session, doRms=False)

    # get the weights file:
    w = np.loadtxt(getWeightsFileNameWithPath(data_path, session), delimiter=",")
    
    #get noise ICs using heatmap:
    list_of_noise_ics = [4,7]
    if (len(list_of_noise_ics) == 0):
        get_heatmap("", num_components, w, default_pic=True)
        list_of_noise_ics = get_a_list_of_numbers_from_user("insert noise comps:\n")

    list_of_noise_ics = [x-1 for x in list_of_noise_ics]

    #plot frequency spectrum of all components:
#    plot_time_domain_subplots_of_all_components(components[A], original_frequency, num_components, list_of_noise_ics)
    #plot_signal_density_subplots_of_all_components(components[A], original_frequency, num_components, list_of_noise_ics)
    data = get_smile_data(components, chunks, p)
    plot_signal_density_welch_method_subplots_of_all_components(data, original_frequency, num_components, list_of_noise_ics)
    plot_signal_density_subplots_of_all_components(data, original_frequency, num_components, list_of_noise_ics)
    plot_time_domain_subplots_of_all_components(data, original_frequency, num_components, list_of_noise_ics)


def find_location_in_array_by_value(array, value):
    return np.where(array == value)[0][0]

def single_main(session, p):
    paths = getPathsFromSessionFolder(os.path.join(data_path, session))
    pathProperties = paths[p]
    Y, chunks, freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
    #Y = zscore(Y, axis=1)
    w, x = EdfAnalyzer.ICA(Y, num_components)
    #auto_noise_list = get_noise_components_by_sort_w(w)
    #extra_headers = [i in auto_noise_list for i in range(1, num_components + 1)]
    
    data = get_smile_data(x, chunks, None) #x
    #noise_order =np.argsort([integrate_abs_of_signal(data[i], freq) for i in range(num_components)])
    
    #extra_headers = [find_location_in_array_by_value(noise_order, i) for i in range(0, num_components)]

    get_heatmap("", num_components, w, default_pic=True)

    list_of_noise_ics = get_a_list_of_numbers_from_user("insert noise comps:\n")
    list_of_noise_ics = list(np.array(list_of_noise_ics) - 1)

    #plot_moving_window_rms_subplots_of_all_components(data, freq, num_components, list_of_noise_ics)

    #plot_signal_density_subplots_of_all_components(data, freq, num_components, list_of_noise_ics)

    plot_color_density_spectrum_subplots_of_all_components(x, freq, num_components, list_of_noise_ics)
    

if __name__ == "__main__":
    sessions = GetSessions(data_path)
    session = sessions[0]
    shared = False
    if (shared):
        shared_main(session, A)
    else:
        single_main(session, A)
    
