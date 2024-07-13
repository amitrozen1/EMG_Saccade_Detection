from scipy.stats import ttest_ind
import numpy as np
from Analysis.blink_correlation import *
from General.init import *

def calculate_accuracy_rate(plv_scores, rate_based_scores, actual_sync, threshold=0.7):
    """
    Calculates whivh of the two methods is more accurate in predicting synchronization - PLV or rate-based 
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

def t_score_test(sync_group, non_sync_group):
    """
    gets as input two groups, one group is synchronized and the other is not
    sync_group - a list of the PLV or rate_based scores of the synchronized sessions
    non_sync_group - a list of the PLV or rate_based (same as the synced group) scores of the non-synchronized sessions
    returns the t-score of the two groups
    """
    
    # In case the groups are not of the same size, uses the smaller size
    if len(sync_group) > len(non_sync_group):
        sync_group = sync_group[:len(non_sync_group)]
    elif len(sync_group) < len(non_sync_group):
        non_sync_group = non_sync_group[:len(sync_group)]   

    sync_group = np.array(sync_group)
    non_sync_group = np.array(non_sync_group)

    sync_group = sync_group[~np.isnan(sync_group)]
    non_sync_group = non_sync_group[~np.isnan(non_sync_group)]

    if np.var(sync_group) == 0 or np.var(non_sync_group) == 0:
        raise ValueError("The variance of one of the groups is zero")

    t_score, p_value = ttest_ind(sync_group, non_sync_group)
    return t_score       

def PLV_vs_rate_based_ttest(plv_scores, rate_based_scores, plv_scores_asynchronized, rate_based_scores_asynchronized):
    """
    compares the PLV and rate-based scores of the synchronized and asynchronized sessions
    returns the t-scores of the two groups
    """

    t_score_plv = t_score_test(plv_scores, plv_scores_asynchronized)
    t_score_rate_based = t_score_test(rate_based_scores, rate_based_scores_asynchronized)
    print("t_score_plv: ", t_score_plv)
    print("t_score_rate_based: ", t_score_rate_based)
    if t_score_plv > t_score_rate_based:
        print("PLV is better than rate-based")
        print ("t_score_plv: ", t_score_plv)
        print ("t_score_rate_based: ", t_score_rate_based)
    else:
        print("Rate-based is better than PLV")
        print ("t_score_plv: ", t_score_plv)
        print ("t_score_rate_based: ", t_score_rate_based)
    
    return t_score_plv, t_score_rate_based
