import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
import math
import pywt
from scipy import ndimage, signal
from scipy.stats import zscore
import cv2

def ssim(sig1, sig2):
    # return structural_similarity(sig1, sig2, gaussian_weights = True,
    #                              sigma = 1.5, use_sample_covariance = False, multichannel=False, win_size=3)
    return structural_similarity(sig1, sig2)

def computeCorr(sig1, sig2):
    sharedLen = min(len(sig1), len(sig2))
    sig1 = np.array(sig1[0:sharedLen])
    sig2 = np.array(sig2[0:sharedLen])
    r,p =  pearsonr(sig1, sig2)
    ssim_val  = ssim(sig1,sig2)
    return r, ssim_val

def maxCrossCor(sig1, sig2, window_size):
    sig1 = sig1 # - np.mean(sig1)
    sig2 = sig2 # - np.mean(sig2)
    xcorrs = []
    for i in range(0, 2*window_size):
        xcorrs.append(np.sum(sig1*sig2[i:i+window_size]))
    return np.max(xcorrs)/(window_size*np.std(sig1)*np.std(sig2))


def crossCorTwoStories(sig1, sig2, window_size, startIndex1, startIndex2, endIndex1, endIndex2):
    slidingLen = min(endIndex1-startIndex1, endIndex2-startIndex2)
    xcorss = []
    for i in range(0,slidingLen, window_size):
        startIndex1 += window_size
        startIndex2 += window_size
        #Spontaneous facial mimicry in response to dynamic facial expressions,
        #https://doi.org/10.1016/j.cognition.2006.05.001.
        # mimicry happens in 850ms +- 200ms windows
        xcorr = []
        for constOffset in [-850*4, 850*4]:
            xcorr.append(maxCrossCor(sig1[startIndex1 : startIndex1 + window_size], sig2[startIndex2 + constOffset - window_size : startIndex2  + constOffset + (2*window_size)], window_size))
        xcorss.append(max(xcorr))
    return np.mean(xcorss)

def movingCorr(sig1, sig2, cmp_func, lagMs, windowMs, freq):
    shapeIndex = 1
    if (len(sig1.shape) == 1 ):
        shapeIndex = 0
    end = min(sig1.shape[shapeIndex], sig2.shape[shapeIndex])

    lag = (lagMs*freq) // 1000
    window = (windowMs*freq) // 1000
    max_vals = []

    #sig1 = zscore(sig1, axis=1)
    #sig2 = zscore(sig2, axis=1)

    for i in range(lag, end-lag-window):
        a_start = i-lag
        b_start = i+lag
        a_end = a_start + window
        b_end = b_start + window
        x1 = []
        sa = []
        sb = []
        if ( shapeIndex == 1 ):
            x1 = sig1[:,i:i+window]
            sa = sig2[:, a_start:a_end]
            sb = sig2[:, b_start:b_end]

        else:
            x1 = sig1[i:i+window]
            sa = sig2[a_start:a_end]
            sb = sig2[b_start:b_end]
        
        sa_cmp_val = float(cmp_func(x1,sa))
        sb_smp_val = float(cmp_func(x1,sb))
        max_vals.append(max(sa_cmp_val, sb_smp_val))
    return np.mean(max_vals)

def movingSsimSpecificComponents(sig1, sig2,components, lagMs = 800, windowMs = 1000, freq=4000):
    return movingCorr(sig1[components, :], sig2[components, :], ssim, lagMs, windowMs, freq)

def movingSsim(sig1, sig2, lagMs = 800, windowMs = 1000, freq=4000 ):
    return movingCorr(sig1, sig2, ssim, lagMs, windowMs, freq)

def movingMatrixCorr(sig1, sig2, lagMs = 800, windowMs = 1000, freq=4000):
    return movingCorr(sig1, sig2,matrixCorrelation, lagMs, windowMs, freq)

def specificSsim(sig1, sig2, components):
    end = min(sig1.shape[1], sig2.shape[1])
    return ssim(zscore(sig1[components, :end], axis=1), zscore(sig2[components, :end], axis=1))

def matrixCorrelation(mat1, mat2, comps=[]):
    cols1 = mat1.shape[1]
    cols2 = mat2.shape[1]
    cols = min(cols1, cols2)
    a=0
    if ( len(comps) == 0 ):
        a= np.corrcoef(mat1[:, :cols], mat2[:, :cols])
    else:
        a= np.corrcoef(mat1[comps, :cols], mat2[comps, :cols], rowvar=True)
    return np.mean(a)

def orb_sim(sig1, sig2):
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(sig1, None)
    kp_b, desc_b = orb.detectAndCompute(sig2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc_a, desc_b)
    similiar_regions = [i for i in matches if i.distance < 50 ]
    if ( len(matches) == 0):
        return 0
    return len(similiar_regions)/len(matches)


def pearson_rows(mat1, mat2):
    '''
    input : 2 matrices of size mxn
    output: 1xm size vector of the rows pearson
    matrix dimension must agree
    '''
    corrs = []
    for i in range(mat1.shape[0]):
        r, _ = pearsonr(mat1[i, :], mat2[i, :])
        corrs.append(r)
    return np.array(corrs)


def xwt_coherence(x1, x2, fs, nNotes=12, detrend=True, normalize=True):
    
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2),   "error: arrays not same size"
    
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x1 = signal.detrend(x1,type='linear')
        x2 = signal.detrend(x2,type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef1, freqs1=pywt.cwt(x1,scales,'cmor1.5-1.0')
    coef2, freqs2=pywt.cwt(x2,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # Calculates the cross transform of xs1 and xs2.
    coef12 = coef1 * np.conj(coef2)

    ###########################################################################
    # coherence
    scaleMatrix = np.ones([1, N]) * scales[:, None]
    S1 = ndimage.gaussian_filter((np.abs(coef1)**2 / scaleMatrix), sigma=2)
    S2 = ndimage.gaussian_filter((np.abs(coef2)**2 / scaleMatrix), sigma=2)
    S12 = ndimage.gaussian_filter((np.abs(coef12 / scaleMatrix)), sigma=2)
    WCT = S12**2 / (S1 * S2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return WCT, times, frequencies, coif


def TestCheck(res, isPass, TestName):
    print(res)
    if ( isPass ):
        print (f"Test Pass: {TestName}")
    else:
        print(f"Test Failed: {TestName}")


from pyrqa.time_series import TimeSeries, EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius, FAN, Unthresholded
from pyrqa.analysis_type import Cross, Classic
from pyrqa.metric import EuclideanMetric, MaximumMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator


def prepare_settings_for_rqa(data_points_1, data_points_2, distance_for_settings = 0.1):
    time_series_1 = TimeSeries(data_points_1,
                            embedding_dimension=1,
                            time_delay=0)
    
    time_series_2 = TimeSeries(data_points_2,
                            embedding_dimension=1,
                            time_delay=0) 
    
    time_series = (time_series_1,time_series_2)

    settings = Settings(time_series,
                        analysis_type=Cross,
                        neighbourhood = FixedRadius(distance_for_settings), 
                        similarity_measure=MaximumMetric,
                        theiler_corrector=1)
    
    return settings

def cross_rqa_plot(data_points_1, data_points_2, distance_for_settings = 0.1):
    settings = prepare_settings_for_rqa(data_points_1, data_points_2)
    computation = RPComputation.create(settings)
    result = computation.run()
    print(result)
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix, "plot.jpg")

def cross_rqa(data_points_1, data_points_2):
    settings = prepare_settings_for_rqa(data_points_1, data_points_2)
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result

def cross_rqa_for_matrix(mat1, mat2):
    e1 = EmbeddedSeries(mat1)
    e2 = EmbeddedSeries(mat2)
    embedded_series = (e1,e2)
    settings = Settings(embedded_series,
                    analysis_type=Cross, # could also be Classic
                    neighbourhood=FixedRadius(0.1),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=0, )
    
    computation = RQAComputation.create(settings,verbose=False)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2
    return result

# vec1 = [0.9, 0.1, 0.2, 0.3, 0.5, 1.7, 0.4, 0.8, 1.5]
# vec2 = [0.3, 1.3, 0.6, 0.2, 1.1, 1.9, 1.3, 0.4, 0.7, 0.9, 1.6]
# print(len(vec1))
# print(len(vec2))
# res = cross_rqa(vec1 , vec2)
# print(res.recurrence_rate)
# print(res.determinism)
# print(res.average_diagonal_line)
# exit()

################# TESTS ########################

if __name__ == "__main__2":

    TestName = "Half vector the same"
    a = np.ones((16,10*4000))
    b = np.append(np.ones((16,5*4000)), np.zeros((16, 5*4000)), axis = 1)

    res = float(movingSsim(a,b))
    
    TestCheck(res, res - 0.5 < 0.01, TestName)
    
    TestName = "SinusPhase"
    x = np.arange(120*40)/100
    lag = 1000
    shift = (lag/1000)*4

    res = movingSsim(a,b,lag)
    
    TestCheck(res, math.fabs(res - 1)<0.01, TestName)

if __name__ == "__main__":
    x = np.linspace(0, 10*np.pi, 1000)
    x = np.sin(x)
    y = np.sin(x+2*np.pi)

    s = np.std(x)

    cross_rqa_plot(x, y,5)