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

def check_parameter_importance(svc, X_test, y_test, feature_names):
    perm_importance = permutation_importance(svc, X_test, y_test)
    features = np.array(feature_names)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()
    plt.cla()

def plot_points(df : pd.DataFrame, arctan=False, isSameChoice=True):
    x = []
    y = []
    for i in range(0,len(df), 2):
        if ( isSameChoice and df.iloc[i]["isChoiceA"] != df.iloc[i]["isChoiceB"]):
            continue

        x_i = [float(df.iloc[i]["mean_corr"]), float(df.iloc[i+1]["mean_corr"])]
        #x_i = [float(df.iloc[i]["1-mean_recurrence_rate"]), float(df.iloc[i+1]["1-mean_recurrence_rate"])]
        
        x.append(x_i)
        reader = df.iloc[i]["whoseReading"] - 1
        listener = (reader + 1) % 2
        listenerChoiceCol = "isChoiceA"
        if(listener == B):
            listenerChoiceCol = "isChoiceB"
        
        y_i = 1
        if ( df.iloc[i][listenerChoiceCol] == 1):
            y_i = 0 # Chose first story

        #y_i = int(df.iloc[i]["isChoiceA"]) + int(df.iloc[i]["isChoiceB"])
        y.append(y_i)

    if(arctan):
        x = np.arctan(x)
    
    #x = np.array(x) / np.max(np.array(x), axis=0)
    #m, b = np.polyfit(x[:,0], x[:,1], 1)
    plt.plot(x[:,0], x[:,0], color='orange')

    plt.legend("Blue = 1st story")
    for i in range(len(y)):
        color = "b"
        if ( y[i] == 1):
            color = "r"
        if ( y[i] == 2):
            color = "g"
        plt.scatter(x[i][0], x[i][1], c=color)
        

    plt.show()


def leave_one_out_boot(df : pd.DataFrame, out_session : str, isSameChoices):
    # leave one out svm.


    #   SVM RESULTS:
    #   0 == choose first, 1 == choose second
    x = []
    y = []
    x_test = []
    y_test = []
    
    
    for i in range(0,len(df), 2):

        choiceA = random.randint(0, 1)
        choiceB = random.randint(0, 1)

        if ( isSameChoices and  choiceA != choiceB ):
            continue
        
        isTest = df.iloc[i]["session"] == out_session

        x_i = [float(df.iloc[i]["mean_corr"]) -  float(df.iloc[i+1]["mean_corr"])]
        #x_i += [float(df.iloc[i]["mean_corr_smiles"]), float(df.iloc[i+1]["mean_corr_smiles"])]
        #x_i += [x_i[0] ** 2, x_i[1] ** 2, (x_i[0]-x_i[1])/(x_i[1])]
        

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
        


        # y_i = int(int(df.iloc[i]["isChoiceA"]) == int(df.iloc[i]["isChoiceB"]))

        y_i = int(df.iloc[i]["isChoiceA"]) * int(df.iloc[i]["isChoiceB"])

        if(isTest):
            y_test.append(y_i)
        else:
            y.append(choiceA)
    
    if ( len(x) == 0 or len(x_test) == 0):
        return [],np.array([])
    
    classifier = SVC(kernel="poly") #GaussianNB()   
    classifier.fit(x,y)

    y_pred = classifier.predict(x_test)
    
    return y_test, y_pred

def leave_one_out(df : pd.DataFrame, out_session : str, isSameChoices = False,  debug=False, arctan=False):
    # leave one out svm.
    
    if ( debug ):
        out_session = "NotAsession"

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
        x_i = []
        m_i = float(df.iloc[i]["mean_corr"])
        m_i_1 = float(df.iloc[i+1]["mean_corr"])
        v_i = float(df.iloc[i]["mean_var_corr"])
        v_i_1 = float(df.iloc[i+1]["mean_var_corr"])
        x_i =  [m_i - m_i_1, v_i - v_i_1, float(df.iloc[i+1]["isOther"])]#, m_i/m_i_1, v_i , v_i_1, v_i/v_i_1]
        #x_i += [(v_i - v_i_1) ]
        #x_i += [float(df.iloc[i]["1-mean_determinism"]) ,  float(df.iloc[i+1]["1-mean_determinism"])]
        #x_i += [float(df.iloc[i]["1-mean_recurrence_rate"]), float(df.iloc[i+1]["1-mean_recurrence_rate"])]
        #x_i += [float(df.iloc[i]["smile_mean_corr"]) -  float(df.iloc[i+1]["smile_mean_corr"])]
        #x_i += [float(df.iloc[i]["angry_mean_corr"]) -  float(df.iloc[i+1]["angry_mean_corr"])]
        #x_i += [float(df.iloc[i]["mean_diff_corr"]), float(df.iloc[i+1]["mean_diff_corr"]),float(df.iloc[i]["mean_diff_corr_var"]), float(df.iloc[i+1]["mean_diff_corr_var"])]
        #x_i += [x_i[0] ** 2, x_i[1] ** 2, (x_i[0])/(x_i[1])]

        # if ( arctan ):
        #     x_i = np.arctan(x_i).tolist()

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

        #y_i = int(int(df.iloc[i]["isChoiceA"]) == int(df.iloc[i]["isChoiceB"]))

        y_i = int(df.iloc[i]["isChoiceA"]) * int(df.iloc[i]["isChoiceB"])
        #y_i = int(df.iloc[i]["isChoiceA"]) + int(df.iloc[i]["isChoiceB"])

        # not a great option!!! 
        if(isTest):
            y_test.append(y_i)
        else:
            y.append(y_i)
    
    if ( (not debug) and (len(x) == 0 or len(x_test) == 0)):
        return [],np.array([]) , np.array([])
    

    classifier = SVC(kernel="rbf") #GaussianNB()   MLPClassifier(max_iter=3000,hidden_layer_sizes=(5,2))
    classifier.fit(x,y)


    check_parameter_importance(classifier, x_test, y_test, ["m_i - m_i_1", "v_i - v_i_1", "isOther"])#, "smile_diff", "anger_diff"])

    if ( debug ):
        print(classifier.coef_)
        return

    y_pred = classifier.predict(x_test)
    
    return y_test, y_pred , np.array([])#np.array(classifier.coef_)
    
def random_leave_one_out_runner(df, sessions):
    y_test = []
    y_pred = []
    for session in sessions:
        y_test_i, y_pred_i = leave_one_out_boot(df, session, True) #, 
        y_test += y_test_i
        y_pred += y_pred_i.tolist()
    
    return accuracy_score(y_test,y_pred)


data_path = r"C:\Liron\DataEmg\Done"
A = 0
B = 1




#csv_path = os.path.join(data_path, "xcorr_and_choices_result_all_sessions_noOverlap.csv")
#csv_path = os.path.join(data_path, "XCorrResults", "NoOverLap_YesArcTan_1500Lag_BiggerMat2.csv")
#csv_path_smiles = os.path.join(data_path, "SmileXcorrResults", "svc.csv")
#csv_path = os.path.join(data_path, "XCorrResults", "NoOverLap_YesArcTan_200-1100Lag_BigMat_250msWindow.csv")
#csv_path = os.path.join(data_path, "XCorrResults", "Rms50Hz_NoOverLap_YesArcTan_0-1500Lag_BigMat_500msWindow_withHighestVars.csv")
# csv_path = os.path.join(data_path, "XCorrResults","Rms50Hz_NoOverLap_YesArcTan_0-1500Lag_BigMat_500msWindow_withHighestVars+Angry+Smiles.csv")
#csv_path = os.path.join(data_path, "XCorrResults","Rms50Hz_NoOverLap_0-1500Lag_BigMat_500msWindow_RQA_removeNoiseComp.csv")  
#csv_path = os.path.join(data_path, "XCorrResults", "Rms50Hz_NoOverLap_0-1500Lag_BigMat_500msWindow_RQA_removeNoiseComp_blankRemover.csv")
csv_path = os.path.join(data_path, "XCorrResults", "Rms20Hz_NoOverLap_0-1500Lag_BigMat_500msWindow_RQA_removeNoiseComp_blankRemover.csv")


justAcc = True

if ( __name__ == "__main__"):
    file = pd.read_csv(csv_path)
    df = pd.DataFrame(file)
    sessions = GetSessions(data_path)

    #indexes = df["key"].str.contains("Read")
    indexes = df["key"].str.contains("Listen")  # fix that.
    #indexes = df["rtChoiceA"] != 0
    #indexes = df["isOther"] == 1

    reading_sessoins = df[indexes]
    

    # plot_points(reading_sessoins, True, True)
    # exit()

    # leave_one_out(reading_sessoins, "", True)
    # exit()

    #predictListener(reading_sessoins)
    
    acc_list = []

    if ( not justAcc):
        for i in tqdm(range(0,1000)):
            acc = random_leave_one_out_runner(reading_sessoins, sessions)
            acc_list.append(acc)
        print(np.mean(acc_list))
        
        
        sns.distplot(acc_list, hist=True, kde=True )
    
    y_test = []
    y_pred = []
    coefs = np.array([])
    for session in sessions:
        y_test_i, y_pred_i, coefs_i = leave_one_out(reading_sessoins, session, isSameChoices = False, arctan=True)
        y_test += y_test_i
        y_pred += y_pred_i.tolist()
        if ( len(coefs) == 0):
            coefs = coefs_i
        elif ( len(coefs_i) > 0):
            coefs = coefs + coefs_i
    
    if ( justAcc ):
        if ( len(coefs)!=0):
            print(coefs/len(sessions))
    
    if ( not justAcc):
        m = np.mean(acc_list)
        std = np.std(acc_list)
        print(f"mean: {m}")
        print(f"Var: {std}")
        accuracy = accuracy_score(y_test,y_pred)
        z_val = (accuracy - m)/std
        import scipy.stats
        p_val =  scipy.stats.norm.sf(abs(z_val))
        print(f"accuracy : {accuracy} Z-Value : {z_val}, p: {p_val}")

        null_p_val = np.count_nonzero(np.array(acc_list) > accuracy)/len(acc_list)
        print(f"null_p_val : {null_p_val}")
        
        plt.axvline(x=accuracy, color = 'r')
        plt.show()
    
    else:
        print(classification_report(y_test, y_pred))

    #print(accuracy_score(y_test,y_pred))
    #print(classification_report(y_test, y_pred))

    