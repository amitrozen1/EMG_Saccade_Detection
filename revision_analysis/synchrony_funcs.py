import numpy as np

def lin_concordance_correlation_coefficient(y_true, y_pred):
    """Calculate Lin's Concordance Correlation Coefficient (CCC)"""
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    
    var_y_true = np.var(y_true)
    var_y_pred = np.var(y_pred)
    
    covariance = np.cov(y_true, y_pred, ddof=0)[0, 1]
    
    numerator = 2 * covariance
    denominator = var_y_true + var_y_pred + (mean_y_true - mean_y_pred) ** 2
    
    return numerator / denominator

def get_max_xcorr_CCC_1d(mat1, mat2, tMaxMs, downsample_window_ms, s, e, direction, shouldReturnXcorrVec=False):
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
        corr = lin_concordance_correlation_coefficient(window, mat2)
        correlations.append(corr)
        
        if (corr is not np.nan) and (corr > max_corr):
            max_corr = corr
            max_ind = i
     
    max_ind = np.abs(s1 - s + max_ind) * downsample_window_ms
    
    if shouldReturnXcorrVec:
        return max_corr, max_ind, correlations

    return max_corr, max_ind