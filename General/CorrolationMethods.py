import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths, hilbert
from General.HelperFunctions import GetSessions, cleanSpace, RemoveRowFromMatrix
from EDF.ComponentValidator import GetNoiseComponentsBySort
import numpy as np
from EDF.SharedIcaRunner import LoadComponents
import matplotlib.pyplot as plt
from General import HtmlTablePlot
from General.HelperFunctions import reshape_num_channels
import seaborn as sns
from dtw import dtw, accelerated_dtw
from dtaidistance import dtw, dtw_ndim
from dtaidistance.dtw_ndim import warping_paths
from dtaidistance.dtw import best_path
from scipy.stats import pearsonr
from scipy.signal import correlate2d, correlate
import statsmodels.api as sm

# matrix channel values are zscored
def decide_by_activation_value(mat1, mat2):
    activation_mean = np.mean(mat1, axis=1) + np.mean(mat2, axis=1)
    return not np.all(activation_mean < 0)

def get_xcorr_vector(mat1, mat2, tMaxMs, num_of_pixels_in_step,downsampleWindowInMs, s, e):
    # tmax is the maximun lag in ms
    # num_of_pixels_in_step is the number of pixels in each step
    # s is the start of the window
    # e is the end of the window
    tMax = tMaxMs // (num_of_pixels_in_step*downsampleWindowInMs)
    xcorr_matrix = np.empty((mat1.shape[0],2*tMax), float)
    i=0
    for lag_ind in np.arange(-tMax,tMax):
        lag = int(lag_ind * num_of_pixels_in_step)

        if ( e+lag < mat2.shape[1] and s+lag>0 ): #and decide_by_activation_value(mat1[:, s:e], mat2[:, (s+lag):(e+lag)])):
            # norm_a = np.sqrt(np.sum(np.power(mat1[:, s:e],2),axis=1))
            # norm_b = np.sqrt(np.sum(np.power(mat2[:, (s+lag):(e+lag)],2),axis=1))
            #xcorr_matrix[:,i] = np.sum((mat1[:, s:e])*(mat2[:, (s+lag):(e+lag)]), axis=1)/ (norm_a*norm_b)
             xcorr_matrix[:,i] = np.mean((mat1[:, s:e])*(mat2[:, (s+lag):(e+lag)]), axis=1)
        else:
            xcorr_matrix[:,i] = np.zeros_like(xcorr_matrix[:,i])
        i = i+1
    
    return xcorr_matrix


def pearson_func(mat1, mat2):
    rownum = mat1.shape[0]
    corr = np.empty((rownum,1), float)
    for i in range(rownum):
        corr[i] = pearsonr(mat1[i,:], mat2[i,:])[0]
    return corr

def mult_func(mat1, mat2):
    return np.mean(mat1*mat2, axis=1)

def get_xcorr_vector_generic(mat1, mat2, tMaxMs ,downsample_window_ms, s ,e , func):
    # tmax is the maximun lag in ms
    # window size for corr is e-s
    # s is the start of the window
    # e is the end of the window

    xcorr_matrix = np.empty((mat1.shape[0],0), float)
    first_lag = np.nan
    last_lag = np.nan
    for lag in np.arange(-tMaxMs, tMaxMs, downsample_window_ms):
        if ( e+lag < mat2.shape[1] and s+lag>0 ):
            xcorr_matrix = np.c_[xcorr_matrix,func(mat1[:, s:e], mat2[:, (s+lag):(e+lag)])]
            if ( np.isnan(first_lag) ):
                first_lag = lag
            last_lag = lag
    
    return xcorr_matrix, first_lag, last_lag

def get_xcorr_vector_pearson(mat1, mat2, tMaxMs, num_of_pixels_in_step, downsampleWindowInMs, s, e):
    # tmax is the maximum lag in ms
    # num_of_pixels_in_step is the number of pixels in each step
    # s is the start of the window
    # e is the end of the window
    tMax = tMaxMs // (num_of_pixels_in_step * downsampleWindowInMs)
    xcorr_matrix = np.empty((mat1.shape[0], 2 * tMax), float)
    i=0
    for lag_ind in np.arange(-tMax, tMax):
        lag = int(lag_ind * num_of_pixels_in_step)

        if e + lag < mat2.shape[1] and s + lag > 0 : #and decide_by_activation_value(mat1[:, s:e], mat2[:, (s + lag):(e + lag)]):
            for j in range(mat1.shape[0]):
                xcorr_matrix[j, i], _ = pearsonr(mat1[j, s:e], mat2[j, (s + lag):(e + lag)])
        else:
            xcorr_matrix[:, i] = np.zeros(mat1.shape[0])
        i = i + 1
    return xcorr_matrix


def is_peak_maximun(vec, peak_index):
    if ( peak_index - 2 > 0 and peak_index + 2 <= len(vec)):
        diffs = np.diff(vec[peak_index-2:peak_index+2])
        return diffs[0] * diffs[2] < 0
    return False

def find_maxiaml_maximum_peak(vec, peaks):
    peaks_index_sorted_by_balue = peaks[np.argsort(vec[peaks])[::-1]]
    for peak in peaks_index_sorted_by_balue:
        if is_peak_maximun(vec, peak):
            return peak
    #wondered if should send a value (not index) and value of zero when no peak is found
    return peaks_index_sorted_by_balue[-1]


def get_xcorr_matrix_peak(xcorr_matrix, num_of_pixels_in_step, tMaxMs, downsampleWindowInMs):
    # xcorr_matrix is the matrix of xcorr vectors
    # returns the peak of the xcorr matrix
    max_corr = np.array([])
    max_ind  = np.array([])

    for i in range(xcorr_matrix.shape[0]):
        vec = xcorr_matrix[i,:]
        peaks = find_peaks(vec)[0]
        
        if ( len(peaks) == 0 ):
            max_corr = np.append(max_corr, 0)
            max_ind  = np.append(max_ind, 0)
        else:
            # Max peak value:
            argmax   = np.argmax(vec[peaks])
            max_corr = np.append(max_corr, vec[peaks[argmax]])
            max_ind_value  = peaks[argmax]

            # Max peak prominance:
            # peak_prom = peak_prominences(vec, peaks)[0]
            # max_prom_ind = np.argmax(peak_prom)
            # max_ind_value = peaks[max_prom_ind]
            # max_corr = np.append(max_corr, vec[max_ind_value])

            # Max peak width:
            # peak_width = peak_widths(vec, peaks)
            # max_width_ind = np.argmax(peak_width[3] - peak_width[2])
            # max_ind_value = peaks[max_width_ind]
            # max_corr = np.append(max_corr, vec[max_ind_value])

            # by vec'(x):
            # max_ind_value = find_maxiaml_maximum_peak(vec, peaks)
            # max_corr = np.append(max_corr, vec[max_ind_value])
            
            max_ind_value = max_ind_value*num_of_pixels_in_step*downsampleWindowInMs - tMaxMs
            max_ind  = np.append(max_ind, max_ind_value)

    return max_corr, max_ind

def get_xcorr_matrix_peak_generic(xcorr_matrix, tMaxMs, downsample_window_ms, first_lag, last_lag):
    max_corr = np.array([])
    max_ind  = np.array([])

    for i in range(xcorr_matrix.shape[0]):
        vec = xcorr_matrix[i,:]
        peaks = find_peaks(vec)[0]
        
        if ( len(peaks) == 0 ):
            max_corr = np.append(max_corr, 0)
            max_ind  = np.append(max_ind, 0)
        else:
            # Max peak value:
            argmax   = np.argmax(vec[peaks])
            max_corr = np.append(max_corr, vec[peaks[argmax]])
            max_ind_value  = peaks[argmax]
            # plus one so it includes the last lag
            max_ind_value = range(first_lag, last_lag+1, downsample_window_ms)[max_ind_value]
            max_ind  = np.append(max_ind, max_ind_value)
    return max_corr, max_ind

# todo - prefer shorter lags?
# instead of argmax - by peak prominenec?

def get_peak_xcorr(mat1, mat2, tMaxMs, num_of_pixels_in_step,downsampleWindowInMs, s, e):
    xcorr_matrix = get_xcorr_vector(mat1, mat2, tMaxMs, num_of_pixels_in_step,downsampleWindowInMs, s, e)
    max_corr, max_ind = get_xcorr_matrix_peak(xcorr_matrix, num_of_pixels_in_step, tMaxMs, downsampleWindowInMs)
    return max_corr, max_ind

def get_peak_xcorr_pearson(mat1, mat2, tMaxMs, num_of_pixels_in_step,downsampleWindowInMs, s, e):
    xcorr_matrix = get_xcorr_vector_pearson(mat1, mat2, tMaxMs, num_of_pixels_in_step,downsampleWindowInMs, s, e)
    max_corr, max_ind = get_xcorr_matrix_peak(xcorr_matrix, num_of_pixels_in_step, tMaxMs, downsampleWindowInMs)
    return max_corr, max_ind

def get_peak_xcorr_generic(mat1, mat2, tMaxMs, downsample_window_ms, s, e, func_name):
    func = None
    if func_name == "pearson":
        func = pearson_func
    elif func_name == "mult":
        func = mult_func
    
    xcorr_matrix, first_lag, last_lag = get_xcorr_vector_generic(mat1, mat2, tMaxMs, downsample_window_ms, s, e, func)
    max_corr, max_ind = get_xcorr_matrix_peak_generic(xcorr_matrix, tMaxMs, downsample_window_ms, first_lag, last_lag)
    return max_corr, max_ind

def get_max_xcorr_2d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, state="listen", shouldReturnXcorrVec = False):
    # I will create the bigger matrix from mat1
    # listener matrix should be mat1 in the case of "read"
    tMax = tMaxMs // downsample_window_ms
    s1=0
    if "read" in state.lower():
        s1 = s
    elif s-tMax>0:
        s1 = s-tMax

    e1 = e+tMax
    if e1>mat1.shape[1]:
        e1 = mat1.shape[1]
    
    mat1 = mat1[:, s1:e1]
    mat2 = mat2[:, s:e]

    xcorr_vec = correlate2d(mat1, mat2, mode='valid')
    argmax = np.argmax(xcorr_vec)

    max_corr = xcorr_vec[0][argmax] / (mat2.shape[0]*mat2.shape[1])
    
    max_ind = (s1 - s + argmax)*downsample_window_ms
    
    if shouldReturnXcorrVec:
        return xcorr_vec
    
    return max_corr, max_ind

def normalize_correlate(mat1, mat2, mode='valid'):
    # Ensure mat1 is the longer array
    if len(mat1) < len(mat2):
        mat1, mat2 = mat2, mat1

    len1, len2 = len(mat1), len(mat2)
    correlation = []

    if mode == 'valid':
        # Loop over mat1 in steps where mat2 can fully overlap
        for i in range(len1 - len2 + 1):
            sub_mat1 = mat1[i:i+len2]
            # Calculate the product of standard deviations and the square root of the product of lengths
            std_product = np.std(sub_mat1) * np.std(mat2)
            length_product_sqrt = np.sqrt(len(sub_mat1) * len(mat2))

            # Normalize the correlation by the product of standard deviations and the square root of the product of lengths
            norm_factor = std_product * length_product_sqrt

            # Avoid division by zero
            if norm_factor == 0:
                correlation.append(0)
            else:
                correlation_value = np.sum(sub_mat1* mat2)
                normalized_correlation = correlation_value / norm_factor
                correlation.append(normalized_correlation)

    return np.array(correlation)



def get_max_xcorr_1d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, state="listen", shouldReturnXcorrVec=False):
    # I will create the bigger matrix from mat1
    # listener matrix should be mat1 in the case of "read"
    tMax = tMaxMs // downsample_window_ms
    s1=0
    if "read" in state.lower():
        s1 = s
    elif s-tMax>0:
        s1 = s-tMax

    mat1 = mat1.flatten()
    mat2 = mat2.flatten()

    e1 = e+tMax
    if e1>len(mat1):
        e1 = len(mat1)
    
    mat1 = mat1[s1:e1]
    mat2 = mat2[s:e]

    xcorr_vec = normalize_correlate(mat1, mat2, mode='valid') 
    #divide by std of each sub-matrix

    #plot the correlation vector:
    lowess_xcorr_vec = sm.nonparametric.lowess(xcorr_vec, np.arange(len(xcorr_vec)), frac=0.25)[:, 1]
    argmax = np.argmax(lowess_xcorr_vec)
    if argmax == 0 or argmax == len(lowess_xcorr_vec) - 1:
        return 0, 0
    
    # plt.plot(lowess_xcorr_vec)
    # plt.title(f"{state}_1d")
    # #mark the maximal corr:
    # plt.plot(argmax, lowess_xcorr_vec[argmax], 'ro')
    # plt.show()
    # plt.clf()

    max_corr = xcorr_vec[argmax]
    
    max_ind = s1 - s + argmax*downsample_window_ms

    if shouldReturnXcorrVec:
        return xcorr_vec

    return max_corr, max_ind


def get_max_pearson_xcorr_2d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, direction="bi-direction", shouldReturnXcorrVec=False, max_ver=False):
    tMax = tMaxMs // downsample_window_ms
    s1 = 0
    if "one" in direction.lower():
        if ( s1 - (500//downsample_window_ms) > 0 ):
            s1 = s - (500//downsample_window_ms)
        else:
            s1 = s
    elif s - tMax > 0:
        s1 = s - tMax

    e1 = e + tMax
    if e1 > mat1.shape[1]:
        e1 = mat1.shape[1]

    mat1 = mat1[:, s1:e1]
    mat2 = mat2[:, s:e]

    correlations = []
    shifts = range(mat1.shape[1] - mat2.shape[1] + 1)
    
    # Compute Pearson correlation for each possible shift
    for shift in shifts:
        temp_corrs = []
        for row in range(mat1.shape[0]):
            if row < mat2.shape[0]:  # Ensure we only compare rows that exist in both matrices
                mat1_row = mat1[row, shift:(shift + mat2.shape[1])]
                mat2_row = mat2[row, :]
                corr, _ = pearsonr(mat1_row, mat2_row)
                if not np.isnan(corr):  # Skip NaN results, which occur if there's no variance
                    temp_corrs.append(corr)
        if temp_corrs:  # Ensure there are correlations to mean
            if ( max_ver ):
                all_rows_corr = max(temp_corrs)
            else:
                all_rows_corr = np.mean(temp_corrs)
            
            correlations.append((all_rows_corr, shift))
    
    if shouldReturnXcorrVec:
        return [x[0] for x in correlations]

    # Find the shift with the maximum mean correlation
    if correlations:
        max_corr, best_shift = max(correlations, key=lambda x: x[0])
    else:
        return None, None  # No valid correlations found
    
    #max_ind = s1 - s + best_shift * downsample_window_ms
    max_ind = np.abs(s1 - s + best_shift) * downsample_window_ms
    return max_corr, max_ind


from scipy.stats import norm
def norm_pdf_weight(window):
    return 1/norm.pdf(np.mean(window), loc=np.mean(window), scale=np.std(window))

def exp_weight(window):
    return np.exp(np.mean(window) + np.std(window))

def simple_weight(window):
    return np.mean(window) + np.std(window)

def mult_weight(window):
    return np.mean(window) * np.std(window)

def min_mean_weight(a, b=[0]):
    return np.min([np.mean(a), np.mean(b)])

#mat 1 is the listener
#mat 2 is the reader\

def find_true_global_maximum(correlations):
    n = len(correlations)
    if n < 3:
        return None  # Not enough points to have a peak
    
    global_max_value = None
    global_max_index = None
    
    # Check each point, except the first and last
    for i in range(1, n-1):
        if correlations[i] > correlations[i-1] and correlations[i] > correlations[i+1]:
            if global_max_value is None or correlations[i] > global_max_value:
                global_max_value = correlations[i]
                global_max_index = i
    
        
    # plt.plot(correlations)
    # plt.title("lowess_xcorr_vec")
    # if global_max_index is not None:
    #     plt.plot(global_max_index, correlations[global_max_index], 'ro')
    # plt.show()
    # plt.clf()


    return global_max_index


def get_max_xcorr_pearson_1d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, direction, shouldReturnXcorrVec=False, weighted_corr = False, lowess=False):
    tMax = tMaxMs // downsample_window_ms
    s1 = 0
    if "one" in direction.lower():
        if ( s1 - (500//downsample_window_ms) > 0 ):
            s1 = s - (500//downsample_window_ms)
        else:
            s1 = s
    elif s - tMax > 0:
        s1 = s - tMax

    mat1 = mat1.flatten()
    mat2 = mat2.flatten()

    e1 = e + tMax
    if e1 > len(mat1):
        e1 = len(mat1)
    
    mat1 = mat1[s1:e1]
    mat2 = mat2[s:e]

    max_corr = -np.inf
    max_ind = -1

    correlations = []
    # Compute Pearson correlation for each shift within the valid range
    for i in range(len(mat1) - len(mat2) + 1):
        window = mat1[i:i+len(mat2)]
        corr = np.corrcoef(window, mat2)[0, 1]
        if weighted_corr:
            # if "read" in state.lower():
            #     corr = corr * min_mean_weight(window)#np.mean(window) #window is always the listener.
            # else:
            corr = corr * np.mean(window) #min_mean_weight(window, mat2)#np.min([np.mean(window), np.mean(mat2)])
        if not np.isnan(corr):
            correlations.append(corr)
        else:
            correlations.append(0)
        
        if corr > max_corr:
            max_corr = corr
            max_ind = i

    if shouldReturnXcorrVec:
        return correlations
    

    if (lowess):
        lowess_xcorr_vec = sm.nonparametric.lowess(correlations, np.arange(len(correlations)), frac=0.25)[:, 1]
        max_ind = find_true_global_maximum(lowess_xcorr_vec)
        if max_ind is None:
            return 0,0
        max_corr = correlations[max_ind]

    
    #old and fault : max_ind = s1 + max_ind * downsample_window_ms
    max_ind = np.abs(s1 - s + max_ind) * downsample_window_ms

    return max_corr, max_ind


def phase_synchrony(y1,y2):
    phase_synchronies = []
    length = min(y1.shape[1],y2.shape[1])
    for i in range(y1.shape[0]):
        al1 = np.angle(hilbert(y1[i,:]),deg=False)[0:length]
        al2 = np.angle(hilbert(y2[i,:]),deg=False)[0:length]
        phase_synchronies += [np.mean(1-np.sin(np.abs(al1-al2)/2))]
    return np.mean(phase_synchronies)

def min_dtw(d1,d2):
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(np.transpose(d1),np.transpose(d2), dist="sqeuclidean")
    return d/(d1.shape[0]*d1.shape[1]), path
    

def dtai_dist_dtw_ndim(mat1, mat2, max_window):
    from dtaidistance import dtw_ndim
    from dtaidistance import dtw_ndim_visualisation as dtwvis

    min_len = min(mat1.shape[1], mat2.shape[1])
    mat1 = mat1[:,:min_len]
    mat2 = mat2[:,:min_len]
    path = dtw_ndim.warping_path(mat1.T, mat2.T, window = max_window)
    dtwvis.plot_warping(mat1, mat2, path)
    plt.show()
    plt.clf()

def dtai_dist_dtw(mat1, mat2, max_window):
    min_len = min(mat1.shape[1], mat2.shape[1])
    means = []
    

    for i in range(mat1.shape[0]):
        mat1_i = mat1[i,:min_len]
        mat2_i = mat2[i,:min_len]
        path = dtw.warping_path(mat1_i, mat2_i, window = max_window)
        # dtwvis.plot_warping(mat1_i, mat2_i, path)
        # plt.show()
        # plt.clf()
        p1 = [p[0] for p in path]
        p2 = [p[1] for p in path]
        means += [np.mean(mat1_i[p1]*mat2_i[p2])]
    
    return np.mean(means)

def dtai_dist_dtw_pearson(mat1, mat2, max_window, is_ndim=True):
    if ( len(mat1.shape) == 1):
        mat1 = mat1.reshape((1,len(mat1)))
        mat2 = mat2.reshape((1,len(mat2)))
        is_ndim = False
    
    min_len = min(mat1.shape[1], mat2.shape[1])
    mat1 = mat1[:, :min_len]
    mat2 = mat2[:, :min_len]
    path = []
    p1 = []
    p2 = []
    if ( is_ndim ):
        path = dtw_ndim.warping_path(mat1.T, mat2.T, window = max_window)
        p1 = [p[0] for p in path]
        p2 = [p[1] for p in path]
    
    corrolations = []
    for i in range(mat1.shape[0]):
        mat1_i = mat1[i,:]
        mat2_i = mat2[i,:]
        if ( not is_ndim ):
            path = dtw.warping_path(mat1_i, mat2_i, window = max_window)
            p1 = [p[0] for p in path]
            p2 = [p[1] for p in path]
        
        #means += [np.mean(mat1_i[p1]*mat2_i[p2])]
        r, _ = pearsonr(mat1_i[p1],mat2_i[p2])
        corrolations += [r]
    return np.arctan(np.mean(corrolations))


def dtw_ndim_with_dist(from_s, to_s, window):
    dist, paths = warping_paths(from_s, to_s, window=window)
    path = best_path(paths)
    return path, dist

def dtw_pearson_1d(signal1, signal2, max_window):
    path, dist = dtw_ndim_with_dist(signal1, signal2, max_window)
    p1 = [p[0] for p in path]
    p2 = [p[1] for p in path]
    r, _ = pearsonr(signal1[p1], signal2[p2])
    return r, dist


def dtai_dist_dtw_pearson_over_seconds(mat1, mat2, max_window, is_ndim=True):
    
    if ( len(mat1.shape) == 1):
        mat1 = mat1.reshape((1,len(mat1)))
        mat2 = mat2.reshape((1,len(mat2)))
        is_ndim = False
    
    min_len = min(mat1.shape[1], mat2.shape[1])
    mat1 = mat1[:, :min_len]
    mat2 = mat2[:, :min_len]
    path = []
    p1 = []
    p2 = []
    dist = 0
    if ( is_ndim ):
        path, dist = dtw_ndim_with_dist(mat1.T, mat2.T, window = max_window)
        p1 = [p[0] for p in path]
        p2 = [p[1] for p in path]
    
    corrolations = []
    for i in range(mat1.shape[0]):
        mat1_i = mat1[i,:]
        mat2_i = mat2[i,:]
        if ( not is_ndim ):
            path = dtw.warping_path(mat1_i, mat2_i, window = max_window)
            p1 = [p[0] for p in path]
            p2 = [p[1] for p in path]
        corrolations_i = []
        for curr_window in range(0, len(p1), max_window):
            m1_in_window = mat1_i[p1[curr_window:curr_window+max_window]]
            m2_in_window = mat2_i[p2[curr_window:curr_window+max_window]]
            
            if(len(m1_in_window) < max_window/3 ):
                break
            
            r, _ = pearsonr(m1_in_window, m2_in_window)
            if ( not np.isnan(r) ):
                corrolations_i += [r]
                
        corrolations += [np.mean(corrolations_i)]
    return np.arctan(np.mean(corrolations)), dist

def dtai_dist_dtw_pearson_over_seconds_blank_removed(mat1, mat2, max_window, BlankRemover, is_ndim=True):
    min_len = min(mat1.shape[1], mat2.shape[1])
    mat1 = mat1[:, :min_len]
    mat2 = mat2[:, :min_len]
    path = []
    p1 = []
    p2 = []
    if ( is_ndim ):
        path = dtw_ndim.warping_path(mat1.T, mat2.T, window = max_window)
        p1 = [p[0] for p in path]
        p2 = [p[1] for p in path]
    
    corrolations = []
    for i in range(mat1.shape[0]):
        mat1_i = mat1[i,:]
        mat2_i = mat2[i,:]
        corrolations_i = []

        if ( not is_ndim ):
            path = dtw.warping_path(mat1_i, mat2_i, window = max_window)
            p1 = [p[0] for p in path]
            p2 = [p[1] for p in path]
        
        for curr_window in range(0, len(p1), max_window):
            m1_in_window = mat1_i[p1[curr_window:curr_window+max_window]]
            m2_in_window = mat2_i[p2[curr_window:curr_window+max_window]]
            if(len(m1_in_window) < max_window/3 or len(m2_in_window) < max_window/3):
                break

            if ( BlankRemover.shouldAnalyzeCurrentWindow_byComponent([m1_in_window, m2_in_window], i)):
                r, _ = pearsonr(m1_in_window ,m2_in_window)
                if ( not np.isnan(r) ):
                    corrolations_i += [r]
        if(len(corrolations_i) == 0):
            corrolations_i = [0]
        corrolations += [np.mean(corrolations_i)]
    return np.arctan(np.mean(corrolations))

def get_column_array(arr):
    return arr[:,np.newaxis]

def normalize(mat):
    return (mat - get_column_array(np.min(mat, axis=1)))/get_column_array((np.max(mat, axis=1)-np.min(mat, axis=1)))

def z_alpha_score(mat, alpha):
    if ( len(mat.shape) == 1 or mat.shape[0] == 1):
        return (mat - alpha*np.mean(mat))/np.std(mat)
    
    return (mat - alpha*get_column_array(np.mean(mat, axis=1)))/get_column_array(np.std(mat, axis=1))

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

# gpt generated :
# todo - check if I get the same results as with mypearson function.
def windowed_lagged_cross_correlation(signal1, signal2, wSize, wInc, tMax, tInc):
    """
    Perform Windowed Lagged Cross-Correlation analysis on two time series signals.
    
    Args:
    - signal1, signal2 (numpy array): The two time series signals for cross-correlation.
    - wSize (int): Window size in number of samples.
    - wInc (int): Window increment in number of samples.
    - tMax (int): Maximum lag in number of samples.
    - tInc (int): Lag increment in number of samples.
    
    Returns:
    - DataFrame: Cross-correlation values for each window and lag.
    """
    n = get_minimal_vecs_len(signal1, signal2)
    lags = np.arange(-tMax, tMax + 1, tInc)
    rows = int((n - wSize - tMax) / wInc)
    cols = len(lags)
    result = np.zeros((rows, cols))

    max_correlations = []
    max_lags = []

    for i in range(rows):
        start1 = i * wInc
        end1 = start1 + wSize
        window1 = signal1[start1:end1]

        for j, lag in enumerate(lags):
            if lag < 0:
                start2 = max(0, start1 - lag)
                end2 = min(n, end1 - lag)
            elif lag > 0:
                start2 = min(n - wSize, start1 + lag)
                end2 = start2 + wSize
            else:
                start2 = start1
                end2 = end1

            window2 = signal2[start2:end2]
            if len(window1) == len(window2):
                result[i, j] = pearsonr(window1, window2)[0]
            else:
                result[i, j] = np.nan  # Mark as NaN when windows are not of equal length
        lowess_xcorr_vec = sm.nonparametric.lowess(result[i], np.arange(len(result[i])), frac=0.25)[:, 1]
        argmax = np.argmax(lowess_xcorr_vec)
        if argmax == 0 or argmax == len(lowess_xcorr_vec) - 1:
            max_correlations.append(0)
            max_lags.append(0)
        else:
            max_correlations.append(lowess_xcorr_vec[argmax])
            max_lags.append(lags[argmax])
    
    # maximum_lags = lags[np.argmax(result, axis=1)]
    # return np.mean(np.max(result, axis=1))
    return np.nanmean(max_correlations)

def cross_correlation_max_pearson(v1, v2, tMax, weighted_corr=False):
    """
    Compute cross-correlation between two vectors, v1 and v2, with a maximum shift of tMax.
    Only the overlapping parts of the vectors are considered (no padding), and the shift
    with the highest Pearson correlation is selected.

    Parameters:
    - v1, v2: Input vectors
    - tMax: Maximum shift in either direction

    Returns:
    - max_pearson: The highest Pearson correlation value
    - best_shift: The shift corresponding to the highest Pearson correlation
    """
    max_pearson = -np.inf
    best_shift = None

    for shift in range(-tMax, tMax + 1):
        if shift < 0:
            shifted_v1 = v1[:shift]  # Shift left
            shifted_v2 = v2[-shift:]
        elif shift > 0:
            shifted_v1 = v1[shift:]  # Shift right
            shifted_v2 = v2[:-shift]
        else:  # No shift
            shifted_v1 = v1
            shifted_v2 = v2

        # Calculate Pearson correlation if both arrays are non-empty
        if len(shifted_v1) > 0 and len(shifted_v2) > 0:
            r, _ = pearsonr(shifted_v1, shifted_v2)
            if ( weighted_corr ):
                r = r * np.min(shifted_v1)#min_mean_weight(shifted_v1, shifted_v2) #np.mean(shifted_v1) #1 is always the listner.
                #np.min([np.mean(shifted_v1), np.mean(shifted_v2)])

            if r > max_pearson:
                max_pearson = r
                best_shift = shift

    return max_pearson, best_shift

import numpy as np
from scipy.stats import pearsonr

def cross_correlation_max_pearson_ndim(arr1, arr2, tMax):
    """
    Compute cross-correlation between each row of two n-dimensional arrays, arr1 and arr2,
    with a maximum shift of tMax. The shift with the highest mean-over-rows Pearson correlation
    is selected for all rows.

    Parameters:
    - arr1, arr2: Input arrays (n-dimensional)
    - tMax: Maximum shift in either direction

    Returns:
    - max_mean_pearson: The highest mean Pearson correlation value over rows
    - best_shift: The shift corresponding to the highest mean Pearson correlation
    """
    max_mean_pearson = -np.inf
    best_shift = None
    best_corrolation_arr = []

    for shift in range(-tMax, tMax + 1):
        correlations = []
        for row in range(arr1.shape[0]):  # Iterate through rows
            if shift < 0:
                shifted_row1 = arr1[row, :shift]  # Shift left
                shifted_row2 = arr2[row, -shift:]
            elif shift > 0:
                shifted_row1 = arr1[row, shift:]  # Shift right
                shifted_row2 = arr2[row, :-shift]
            else:  # No shift
                shifted_row1 = arr1[row]
                shifted_row2 = arr2[row]

            # Calculate Pearson correlation if both arrays are non-empty, and at least have 50 samples
            if len(shifted_row1) > 40 and len(shifted_row2) > 40:
                r, _ = pearsonr(shifted_row1, shifted_row2)
                correlations.append(r)

        # Compute mean Pearson correlation for the current shift
        if correlations:  # Ensure correlations list is not empty
            mean_pearson = np.mean(correlations)
            if mean_pearson > max_mean_pearson:
                max_mean_pearson = mean_pearson
                best_shift = shift
                best_corrolation_arr = correlations

    return max_mean_pearson, best_shift, best_corrolation_arr

# Example usage (assuming you have EMG signals `emg_signal1` and `emg_signal2`)
# Preprocess the signals
# fs = 20Hz (as per the paper)
# emg_signal1_processed = preprocess_emg_signal(emg_signal1, 20)
# emg_signal2_processed = preprocess_emg_signal(emg_signal2, 20)

# Perform Windowed Lagged Cross-Correlation analysis
# wSize, wInc, tMax, tInc should be determined based on your specific requirements
# wcc_result = windowed_lagged_cross_correlation(emg_signal1_processed, emg_signal2_processed, wSize, wInc, tMax, tInc)