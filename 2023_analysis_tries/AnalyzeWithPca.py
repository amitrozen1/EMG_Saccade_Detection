import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from General.HelperFunctions import GetSessions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



from tqdm import tqdm

headers_for_pca = ["mean_corr", "mean_var_corr", "4_highest_mean", "max_corr_comp", "2highestVarMeanComp", "4highestVarMeanComp", "6highestVarMeanComp", "8highestVarMeanComp", "smile_mean_corr", "smile_mean_var_corr", "smile_mean_A", "smile_var_A", "smile_mean_B", "smile_var_B", "angry_mean_corr", "angry_mean_var_corr", "angry_mean_A", "angry_var_A", "angry_mean_B", "angry_var_B"]

def plot_pca(df : pd.DataFrame):
    x = []
    y = []
    for i in range(0,len(df), 2):
        if (df.iloc[i]["isChoiceA"] != df.iloc[i]["isChoiceB"]):
            continue
        #x_i = [2*(df.iloc[i][h] - df.iloc[i+1][h])/(df.iloc[i][h] + df.iloc[i+1][h]) for h in headers_for_pca]
        x_i = [df.iloc[i][h] - df.iloc[i+1][h] for h in headers_for_pca]
        y_i = df.iloc[i]["isChoiceA"]
        x.append(x_i)
        y.append(y_i)
    # pca = PCA(n_components=len(headers_for_pca))
    # x = pca.fit(x)

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel("num components")
    # plt.ylabel("cummulative var explained")
    # plt.show()

    pca = PCA(n_components=5)
    x = pca.fit_transform(x)
    
    plt.xlabel = "PCA 0"
    plt.ylabel = "PCA 1"
    for i in range(len(y)):
        color = "b"
        if ( y[i] == 1):
            color = "r"
        plt.scatter(x[i][0], x[i][1],  c=color)

    plt.show()

def prepareDF(df : pd.DataFrame, num_components):
    s = []
    x = []
    y = []
    for i in range(0,len(df), 2):
        if (df.iloc[i]["isChoiceA"] != df.iloc[i]["isChoiceB"]):
            continue
        #x_i = [2*(df.iloc[i][h] - df.iloc[i+1][h])/(df.iloc[i][h] + df.iloc[i+1][h]) for h in headers_for_pca]
        
        x_i = [df.iloc[i][h] - df.iloc[i+1][h] for h in headers_for_pca]
        y_i = df.iloc[i+1]["isChoiceA"]
        
        s.append(df.iloc[i]["session"])
        x.append(x_i)
        y.append(y_i)

    x = np.array(x)
    x = zscore(x, axis=1)
    

    pca = PCA(n_components=num_components)
    x = pca.fit_transform(x)
    x = x.tolist()

    new_x = []
    for i in range(len(s)):
        new_x.append([s[i]] + x[i] + [y[i]])

    headersForDf = ["session"] + [f"PCA_{i}" for i in range(num_components)] + ["tag"]
    returnDf = pd.DataFrame(data=new_x, columns= headersForDf )
    return returnDf

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # svc = SVC(kernel="linear")
    # svc.fit(X_train, y_train)
    # y_pred = svc.predict(X_test)

    # print(accuracy_score(y_test, y_pred))

def leave_one_out(df, session, num_components):
    x_headers = [f"PCA_{i}" for i in range(num_components)]
    x = []
    y = []
    x_test = []
    y_test = []

    for i in range(len(df)):
        x_i = [df.iloc[i][header] for header in x_headers]
        y_i = df.iloc[i]["tag"]

        if ( df.iloc[i]["session"] == session):
            x_test.append(x_i)
            y_test.append(y_i)
        else:
            x.append(x_i)
            y.append(y_i)
    

    if ( len(x_test) == 0 ):
        return np.array([]),[]

    svc = SVC(kernel="linear")
    svc.fit(x, y)
    y_pred = svc.predict(x_test)

    return y_pred, y_test


data_path = r"C:\Liron\DataEmg\Done"
A = 0
B = 1
csv_path = os.path.join(data_path, "XCorrResults","Rms50Hz_NoOverLap_YesArcTan_0-1500Lag_BigMat_500msWindow_withHighestVars+Angry+Smiles_myIdea.csv")



if ( __name__ == "__main__"):
    file = pd.read_csv(csv_path)
    df = pd.DataFrame(file)
    sessions = GetSessions(data_path)

    #indexes = df["key"].str.contains("Read")
    indexes = df["key"].str.contains("Listen")  # fix that.
    #indexes = df["rtChoiceA"] != 0
    reading_sessoins = df[indexes]

    # for header in headers_for_pca: 
    #     norm = reading_sessoins[header].abs().max()
    #     reading_sessoins[header] = reading_sessoins[header] / norm
    

    # plot_pca(reading_sessoins)

    num_components = 10
    df = prepareDF(reading_sessoins, num_components)

    y_test = []
    y_pred = []
    for session in GetSessions(data_path):
        y_i_pred, y_i_test = leave_one_out(df, session, num_components)
        y_pred += y_i_pred.tolist()
        y_test += y_i_test
    
    print( accuracy_score(y_test, y_pred) )