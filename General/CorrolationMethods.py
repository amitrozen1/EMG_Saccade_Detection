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

def get_max_xcorr_2d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, state="listen"):
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

    # plt.plot(xcorr_vec[0])
    # plt.title(f"{state}_2d")
    # #mark the maximal corr:
    # plt.plot(argmax, xcorr_vec[0][argmax], 'ro')
    # plt.show()
    # plt.clf()


    max_corr = xcorr_vec[0][argmax] / (mat2.shape[0]*mat2.shape[1])
    
    max_ind = s1 - s + argmax*downsample_window_ms
    return max_corr, max_ind

def get_max_xcorr_1d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, state="listen"):
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

    xcorr_vec = correlate(mat1, mat2, mode='valid')

    #plot the correlation vector:

    argmax = np.argmax(xcorr_vec)

    # plt.plot(xcorr_vec)
    # plt.title(f"{state}_1d")
    # #mark the maximal corr:
    # plt.plot(argmax, xcorr_vec[argmax], 'ro')
    # plt.show()
    # plt.clf()

    max_corr = xcorr_vec[argmax] / len(mat2)
    
    max_ind = s1 - s + argmax*downsample_window_ms
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

if __name__=="__main__":

    data_path = r"C:\Liron\DataEmg\Done"
    num_channels = 16
    A = 0
    B = 1
    downsampleWindowInMs = 20 # 20MS == 50HZ.
    orig_freq = 4000

    freq = orig_freq // (4*downsampleWindowInMs)
    windowForCorrInMs = 1000
    numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
    numOfPixelsForXCorrStep = 50 // downsampleWindowInMs # 100ms
    
    A=0
    B=1

    current_session = GetSessions(data_path)[0]

    independentComponents, chunks = LoadComponents(data_path, current_session, downsampleWindowInMs)

    print("Zscore")
    for p in [A, B]:
        #independentComponents[p] = normalize(independentComponents[p]) #zscore(independentComponents[p], axis=1)
        # independentComponents[p] = differencing(independentComponents[p])
        independentComponents[p] = z_alpha_score(independentComponents[p], 0.5)
    cleanSpace()
    
    rows_to_remove = GetNoiseComponentsBySort(data_path, current_session) #returns sorted.
    for p in [A,B]:
        rows_removed = 0
        for row in rows_to_remove:
            # comes sorted ? 
            independentComponents[p] = RemoveRowFromMatrix(independentComponents[p] , row - rows_removed)
            rows_removed = rows_removed + 1 #so you delete the right row.

    d = HtmlTablePlot.initHtmlDictionary(["key", "is_max_peak_max", "max_lags","dtai_val", "plot"])

    results = []
    for key in chunks[0]:
        chunk_data = [[],[]]
        
        extra_offset = [0, numOfPixelsForXCorrWindow]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mat1 = chunk_data[A]
        mat2 = chunk_data[B]
        
        dtai_val = dtai_dist_dtw(mat1, mat2, numOfPixelsForXCorrWindow)

        #d, path = min_dtw(mat1, mat2)
        #print(d)

        # p_s = phase_synchrony(mat1, mat2)
        # print(p_s)


        #something does not add up with step and window relationship.

        jumps = numOfPixelsForXCorrWindow
        max_len = min(mat1.shape[1], mat2.shape[1])
        num_windows = (max_len - numOfPixelsForXCorrWindow) // jumps +1
        all_windows = np.array([])
        max_corr_all = np.array([])
        max_corr_ind = np.array([])
        max_lag_ms = 1000

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            xcorr_mat = get_xcorr_vector(mat1, mat2, max_lag_ms, numOfPixelsForXCorrStep,downsampleWindowInMs, s, e)
            max_corr, max_ind = get_xcorr_matrix_peak(xcorr_mat, numOfPixelsForXCorrStep, max_lag_ms, downsampleWindowInMs)
            max_corr_all = np.append(max_corr_all, max_corr)
            max_corr_ind = np.append(max_corr_ind, max_ind)
            xcorr_mat = np.max(xcorr_mat, axis=0)
            all_windows = np.append(all_windows, xcorr_mat)
            #plot_chunk_data_one_tick_per_component(xcorr_mat, key ,ticks=max_ind, num_components=xcorr_mat.shape[0])
            #plt.show()
            # savePlot("temp.jpg")
            # clearPlot()
        xcorr_mat = reshape_num_channels(all_windows,num_windows)
        title = f"key : {key}, mean_corr : {np.mean(max_corr)}, mean_lag : {np.mean(max_corr_ind)}"
        sns.heatmap(xcorr_mat, vmin=-1, vmax=1)
        plt.title(title)
        plt.show()
    
        HtmlTablePlot.addRow(d, [HtmlTablePlot.toTextHtmlCol(key), 
                HtmlTablePlot.toTextHtmlCol(max_corr == np.max(xcorr_mat, axis=1)), 
                HtmlTablePlot.toTextHtmlCol(max_ind-10),
                HtmlTablePlot.toTextHtmlCol(dtai_val),
                HtmlTablePlot.toImageHtmlColFromPath("temp.jpg")])
        
    HtmlTablePlot.SaveHtmlFile(d, "test_dtai.html")


