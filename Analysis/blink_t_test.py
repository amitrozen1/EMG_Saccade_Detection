from scipy.stats import ttest_ind
import numpy as np

def calculate_t_scores(plv_scores, rate_based_scores, actual_sync, threshold=0.7):
    """
    Calculate t-scores for PLV and rate-based scores
    :param plv_score: PLV scores of the sessions (a list of lists with PLV scores for each session)
    :param rate_based_score: rate-based scores of the sessions (a list of lists with rate-based scores for each session)
    :param actual_sync: actual synchronization of the sessions - True or False (a list of liss with booleans for each session)
    returns t-scores for PLV and rate-based scores
    """
    # change each list of lists to one long list
    plv_scores = [score for session in plv_scores for score in session]
    rate_based_scores = [score for session in rate_based_scores for score in session]
    actual_sync = [sync for session in actual_sync for sync in session] 

    # Determine predicted sync based on threshold
    predicted_sync_plv = [score >= threshold for score in plv_scores]
    predicted_sync_rate_based = [score >= threshold for score in rate_based_scores]
    
    # Convert actual_sync to numerical values (True = 1, False = 0)
    actual_sync_numeric = [1 if sync else 0 for sync in actual_sync]
    
    # Calculate differences between predicted sync and actual sync
    diff_plv = np.abs(np.array(predicted_sync_plv) - np.array(actual_sync_numeric))
    diff_rate_based = np.abs(np.array(predicted_sync_rate_based) - np.array(actual_sync_numeric))
    
    # Perform t-test
    t_score_plv, p_value_plv = ttest_ind(diff_plv, actual_sync_numeric)
    t_score_rate_based, p_value_rate_based = ttest_ind(diff_rate_based, actual_sync_numeric)
    
    return t_score_plv, t_score_rate_based