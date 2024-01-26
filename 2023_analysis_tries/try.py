from General.CorrolationMethods import min_dtw
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from General.CorrolationMethods import z_alpha_score
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

def pearson_two_arrays(x, y):
    """
    Calculates the pearson correlation between two arrays.
    """
    return np.corrcoef(x, y)[0, 1]


def noisy_sinus_function(t, noise_amplitude=0.5):
    """
    Generates data similar to the sine function but with noise.
    """
    signal = np.sin(2.5 * np.pi * t)
    noise = noise_amplitude * np.random.normal(size=len(t))
    return signal + noise


def noisy_cosinus_function(t, noise_amplitude=0.5):
    """
    Generates data similar to the cosine function but with noise.
    """
    signal = np.cos(2.5 * np.pi * t)
    noise = noise_amplitude * np.random.normal(size=len(t))
    return signal + noise

def plot_signals_in_two_graphs(t, signal1, signal2):
    """
    Plots two signals in two different graphs.
    """
    plt.subplot(2, 1, 1)
    plt.plot(t, signal1)
    plt.title('Signal 1')
    plt.subplot(2, 1, 2)
    plt.plot(t, signal2)
    plt.title('Signal 2')
    plt.tight_layout()
    plt.show()

def dtai_dist_dtw(mat1, mat2, max_window =1000):
    path = dtw.warping_path(mat1, mat2, window = max_window)
    dtwvis.plot_warping(mat1, mat2, path)
    p1 = [p[0] for p in path]
    p2 = [p[1] for p in path]
    plt.show()
    plt.cla()
    t = np.arange(0,len(p1))/len(p1)
    plot_signals_in_two_graphs(t, mat1[p1], mat2[p2])

    plt.title(f"similiarity: {pearson_two_arrays(mat1[p1],mat2[p2])}")
    plt.show()
    plt.cla()

def compute_auc(x,y):
    return np.trapz(y, x)

def dtw_aoc(mat1, mat2, max_window=1000):
    path = dtw.warping_path(mat1, mat2, window = max_window)
    p1 = [p[0] for p in path]
    p2 = [p[1] for p in path]
    max_len = max(len(mat1), len(mat2))
    auc_diag = max_len*max_len/2
    dtwvis.plot_warping(mat1, mat2, path)
    plt.show()
    plt.clf()
    t = np.arange(0,len(p1))
    plt.plot(p1, p2)
    plt.plot(p1,p1, color='red')
    plt.title(f"auc_diag ={auc_diag}, dtw_diff={(auc_diag-compute_auc(p1,p2))/auc_diag}")
    plt.show()




t = np.arange(1,1000)/1000
sinus_a = noisy_sinus_function(t) 
#sinus_a[200:400] = np.zeros_like(sinus_a[200:400])

sinus_b = noisy_sinus_function(t) 
#sinus_b[100:199] = np.zeros_like(sinus_b[100:199])

cosinus_a = noisy_cosinus_function(t)
#cosinus_a[250:380] = np.zeros_like(cosinus_a[250:380])

sinus_a = z_alpha_score(sinus_a, 1)
sinus_b = z_alpha_score(sinus_b, 1)
cosinus_a = z_alpha_score(cosinus_a, 1)

#dtw_aoc(sinus_a, sinus_b)
#dtw_aoc(sinus_a, cosinus_a)
dtw_aoc(cosinus_a, sinus_a)
exit()

dtai_dist_dtw(sinus_a, sinus_b)

dtai_dist_dtw(sinus_a, cosinus_a)

#plot_signals_in_two_graphs(t, sinus_a, cosinus_a)




