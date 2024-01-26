import numpy as np
from General.HelperFunctions import GetSessions
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# line - session, p, key, action, mean_smile, std_smile, mean_angry, std_angry, my_choice, other_choice, isReading, my_rt, other_rt, diffs

A=0
B=1


def predict_from_mean_and_var(data_path):
    csv_path = os.path.join(data_path, "smiles+Angrys_mean_over_listening_svr_with_diff_1.csv")
    csvFile = pd.read_csv(csv_path)
    df = pd.DataFrame(csvFile)

    x = []
    y = []

    for i in range(len(df)):
        x.append([df.iloc[i]["diffMean_angry"]/df.iloc[i]["std_angrys"], df.iloc[i]["isOther"], df.iloc[i]["std_angrys"] ])
        y.append(df.iloc[i]["choice"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    classifier = SVC(kernel= "linear")
    classifier.fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    print(classification_report(y_test, y_predicted))


leave_one_out_variables = ["smile_mean_corr", "angry_mean_corr" ] #, "smile_mean_var_corr", "angry_mean_var_corr"]
leave_one_out_variables += ["angry_mean_A", "angry_var_A", "angry_mean_B", "angry_var_B"]
leave_one_out_variables += ["smile_mean_A", "smile_var_A", "smile_mean_B", "smile_var_B"]

#leave_one_out_variables = ["angry_mean_corr", "angry_mean_var_corr", "angry_mean_A", "angry_var_A", "angry_mean_B", "angry_var_B"]

def leave_one_out(df, out_session, var_headers, isSameChoice = False):
    x_test = []
    y_test = []
    x = []
    y = []
    y_predicted = []

    for i in range(0,len(df)-1,2):
        if ( isSameChoice and df.iloc[i]["isChoiceA"] != df.iloc[i]["isChoiceB"]):
            continue
        
        x_i = [(df.iloc[i][header] - df.iloc[i+1][header]) for header in var_headers]
      
        reader = df.iloc[i]["whoseReading"] - 1
        listener = (reader + 1) % 2
        listenerChoiceCol = "isChoiceA"
        if(listener == B):
            listenerChoiceCol = "isChoiceB"
        
        y_i = 1 #second story
        if ( int(df.iloc[i][listenerChoiceCol]) == 1):
            y_i = 0 #first story
        
    
        isTest = df.iloc[i]["session"] == out_session
        if ( isTest ):
            x_test.append(x_i)
            y_test.append(y_i)
        else:
            x.append(x_i)
            y.append(y_i)

    if ( len(x_test) == 0 or len(x) == 0):
        return [],[]
    
    classifier = SVC(kernel = "rbf")
    classifier.fit(x, y)
    y_predicted = classifier.predict(x_test)

    return y_predicted.tolist(), y_test


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def getHeaderPowerSet(variables = leave_one_out_variables):
    powerset_list = list(powerset(variables))
    return [x for x in powerset_list if len(x) > 1]

def GetAccuracyFromHeader(df, sessions, headers):
    y_predicted = []
    y_true = []
    for session in sessions:
        y_predicted_i, y_true_i = leave_one_out(df, session, headers, False)
        if (len(y_predicted_i)!=0):
            y_predicted += y_predicted_i
            y_true += y_true_i

    #print(classification_report(y_true, y_predicted))
    return accuracy_score(y_true, y_predicted)


def GoOverHeader(df, sessions, headers, accurcies_list, i):
    curr_acc = GetAccuracyFromHeader(df, sessions, headers)
    accurcies_list[i] = curr_acc



data_path = r"C:\Liron\DataEmg\Done"
#csv_path = os.path.join(data_path,"SmileXcorrResults", "smile+angry_corr+means_svc_downsample to 10HZ_atan.csv")
#csv_path = os.path.join(data_path, "XCorrResults","Rms50Hz_NoOverLap_YesArcTan_0-1500Lag_BigMat_500msWindow_withHighestVars+Angry+Smiles_myIdea.csv")
csv_path = os.path.join(data_path, "SmileXcorrResults", "all_svr.csv")
all_sessions = GetSessions(data_path)
train_sessions = all_sessions[0:-3]
cv_sessions = all_sessions[-3:]

single = False

df = pd.DataFrame(pd.read_csv(csv_path))

indexes = df["action"] == "Reading"

df = df[indexes]


new_headers = ["smile_mean_corr", "angry_mean_corr" ,"1 - angry_rqa.recurrence_rate", "1 - angry_rqa.determinisim", "1 - smile_rqa.recurrence_rate", "1 - smile_rqa.determinisim", "1-smile_rqa.longest_diagonal_line", "1-smile_rqa.average_diagonal_line", "1-angry_rqa.longest_diagonal_line", "1-angry_rqa.average_diagonal_line"]
#new_headers = ["1 - smile_rqa.determinisim", "1 - angry_rqa.determinisim"]
#print(GetAccuracyFromHeader(df, GetSessions(data_path), new_headers))

# started to train on training set, but did not use CV set.
if ( not single ):
    header_power_set = getHeaderPowerSet(new_headers)

    accurcies = list(range(len(header_power_set)))
    highest_headers = None


    for i, headers in enumerate(tqdm(header_power_set)):
        GoOverHeader(df, all_sessions, headers, accurcies, i)

    indexes_sorted = np.argsort(accurcies)[::-1]
    with open('indexes_of_highest_accuracies.txt', 'w') as f:
        for m in indexes_sorted:
            print(f"{header_power_set[m]} : {accurcies[m]}", file=f)

    for i in [0,1,2,3,4,5,6]:
        m = indexes_sorted[i]
        print(f"{header_power_set[m]} : {accurcies[m]}")
   