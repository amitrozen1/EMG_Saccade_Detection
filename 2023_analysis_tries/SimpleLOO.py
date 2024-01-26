import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from General.HelperFunctions import GetSessions
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance


A=0
B=1
data_path = r"C:\Liron\DataEmg\Done"
csv_path = os.path.join(data_path, "XCorrResults","Rms50Hz_1000msWindow_maxLag1s_dtw_path.csv")

def leave_one_out(df : pd.DataFrame, out_session : str, isSameChoices = False, arctan=False):
    #   SVM RESULTS:
    #   0 == choose first, 1 == choose second
    x = []
    y = []
    x_test = []
    y_test = []
    for i in range(0,len(df), 2):
        
        if ( isSameChoices and df.iloc[i]["isChoiceA"] != df.iloc[i]["isChoiceB"]):
            continue
        
        isTest = df.iloc[i]["session"] == out_session
        
        m_i = float(df.iloc[i]["mean_peak_corr"])
        m_i_1 = float(df.iloc[i+1]["mean_peak_corr"])
        v_i = float(df.iloc[i]["var_peak_corr"])
        v_i_1 = float(df.iloc[i+1]["var_peak_corr"])
        
        x_i =  [m_i / m_i_1] #, v_i - v_i_1]

        # m_i - m_i_1 : 64% in gaussian NB, 63% in SVM RBF, 64% in Poly
        # m_i / m_i_1: 65% poly.

        if ( arctan ):
            x_i = np.arctan(x_i).tolist()

        if (isTest):
            x_test.append(x_i)
        else:
            x.append(x_i)

        reader = df.iloc[i]["whoseReading"] - 1
        listener = (reader + 1) % 2
        listenerChoiceCol = "isChoiceA"
        if(listener == B):
            listenerChoiceCol = "isChoiceB"
        
        y_i = 1
        if ( int(df.iloc[i][listenerChoiceCol]) == 1):
            y_i = 0 # Chose first story

        #y_i = int(df.iloc[i]["isChoiceA"]) + int(df.iloc[i]["isChoiceB"])

        # not a great option!!! 
        if(isTest):
            y_test.append(y_i)
        else:
            y.append(y_i)
    
    if (len(x) == 0 or len(x_test) == 0):
        return [],np.array([]) , np.array([])
    

    classifier = SVC(kernel="poly") #SVC(kernel="linear") #GaussianNB()   MLPClassifier(max_iter=3000,hidden_layer_sizes=(5,2))
    classifier.fit(x,y)

    y_pred = classifier.predict(x_test)
    
    return y_test, y_pred , np.array([])

keysOfInterest = ['mean_peak_corr', 'var_peak_corr']
def normalize_corr_values_per_session(df):
    for session in df["session"].unique():
        for key in keysOfInterest:
            #df.loc[df["session"] == session, key] = np.arctan(df.loc[df["session"] == session, key])
            df.loc[df["session"] == session, key] = (df.loc[df["session"] == session, key] - df.loc[df["session"] == session, key].mean()) / df.loc[df["session"] == session, key].std()

if __name__== "__main__":
    file = pd.read_csv(csv_path)
    df = pd.DataFrame(file)
    sessions = GetSessions(data_path)

    indexes = df["key"].str.contains("Read")   
    reading_sessoins = df[indexes]

    #normalize_corr_values_per_session(reading_sessoins)

    y_test = []
    y_pred = []
    coefs = np.array([])
    for session in sessions:
        y_test_i, y_pred_i, coefs_i = leave_one_out(reading_sessoins, session, isSameChoices = False, arctan=False)
        y_test += y_test_i
        y_pred += y_pred_i.tolist()
        if ( len(coefs) == 0):
            coefs = coefs_i
        elif ( len(coefs_i) > 0):
            coefs = coefs + coefs_i
    
    print(classification_report(y_test, y_pred))