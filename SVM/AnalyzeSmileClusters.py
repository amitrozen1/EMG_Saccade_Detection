from SVM.SmileTypeClustering import ClusterSmiles, to_particiapnt_string
import numpy as np
from General.HelperFunctions import GetSessions, cleanSpace
from tqdm import tqdm
import os
from PsychData.LoadChoicesJson import loadChoiceFromSession
import pandas as pd
from scipy.stats import zscore
from General.StatisticsFunctions import cross_rqa
from SVM.StateCorrelation import rms_downsample_single, getDataFileName, getJsonFileName, loadJsonToChunks, prepareStreamForXCorr
from General.HelperFunctions import particiapnt_to_char

downsampleWindowInMs = 100 #10 # downsamples to 100HZ

orig_freq = 4000
freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = int(windowForCorrInMs//downsampleWindowInMs)
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow//10

A=0
B=1

def xCorrSmiles(current_session, smileComponents, chunks, overlap=False):

    results = []
    starts = []
    ends = []

    for p in [A,B]:
        chunks_list = list(chunks[p].values())
        starts.append(chunks_list[9].Start.Time) #skip smile x3, angry x3, blink x3
        ends.append(chunks_list[-1].End.Time)
    
    mat1 = smileComponents[A][starts[A] : ends[A]]
    mat2 = smileComponents[B][starts[B] : ends[B]]

    #max_len = min(mat1.shape[1], mat2.shape[1])
    maxValueForWindow = np.array([])

    jumps = numOfPixelsForXCorrWindow
    if ( overlap ):
        jumps = jumps//2

    for i, s in enumerate(range(0, len(mat1) - numOfPixelsForXCorrWindow , jumps)):
        e = s+numOfPixelsForXCorrWindow
        corrmat = np.array([])
        
        # if ( np.mean(mat1[s:e]) < -0.5 ):
        #     continue

        for lag_ind in np.arange(0,11): #00 - 1000 ms lag XXX - it was 11
            #the sliding window (that slides through A) and the window that moves in matB are the same?
            lag = int(lag_ind * numOfPixelsForXCorrStep) # 100 ms at a lag
            vec = np.array([])
            if ( e + lag < len(mat2) ):
                vec = np.mean((mat1[s:e])*(mat2[(s+lag):(e+lag)]))
            if ( s - lag > 0 and e-lag > 0 and lag != 0):
                vec = np.append(vec, np.mean((mat1[s:e])*(mat2[(s-lag):(e-lag)])))
            
            corrmat = np.append(corrmat, vec)

        maxValueForWindow = np.append(maxValueForWindow, np.max(corrmat))
    
    
    maxValueCorr = np.mean(maxValueForWindow) #16x1
    maxValueForWindowVar = np.arctan(np.var(maxValueForWindow))

    if ( len(maxValueForWindow) == 0 ):
        maxValueCorr = 0 
        maxValueForWindowVar = 0 
    
    # results.append([current_session, np.arctan(maxValueCorr), maxValueForWindowVar, np.mean(mat1), np.var(mat1), np.mean(mat2), np.var(mat2)])
    # return results[0]

    means = [np.mean(mat1), np.mean(mat2)]
    vars = [np.var(mat1), np.var(mat2)]
    for p in [A,B]:
        results.append([current_session, particiapnt_to_char(p),np.arctan(maxValueCorr), maxValueForWindowVar, means[p], vars[p] ])
    return results

def getXcorrHeaders():
    return ["session", "particiapnt", "mean_corr", "var_corr", "mean_smile", "var_smile"]

data_path = r"C:\liron\dataEmg\Try"
svmType = "svr"

if ( __name__ == "__main__"):
    csv_path = os.path.join(data_path, "smileCluster_oneXcorr_oneLine_10Hz_1000msWindow_svr.csv")

    sessions = GetSessions(data_path)

    all_correlations = []
    headers = getXcorrHeaders() + ['smileType']

    smile_clustes = ClusterSmiles(data_path, K=3, num_of_representation_comps= 8, plot_fig=True)

    for session in tqdm(sessions):
        svm_folder = os.path.join(data_path, session, "smile", svmType)
        chunks = [[], []]
        smileComponents = [[],[]]
        for p in [A,B]:
            chunks[p] = loadJsonToChunks(os.path.join(svm_folder, getJsonFileName(p)), downsampleWindowInMs)
            x = np.loadtxt(os.path.join(svm_folder, getDataFileName(p)))
            x = rms_downsample_single(x, downsampleWindowInMs)
            x = zscore(x) #prepareStreamForXCorr(x) #zscore(x) #
            smileComponents[p] = x
        
        cleanSpace()
        
        corr_results = xCorrSmiles(session, smileComponents, chunks)
        
        for p in [A,B]:
            corr_results[p] += [smile_clustes[to_particiapnt_string(session,p)][0]]
        
        all_correlations += corr_results
        del smileComponents
        cleanSpace()
        print("finished session " + session)

    df = pd.DataFrame(all_correlations)
    df.columns = headers
    df.to_csv(csv_path)