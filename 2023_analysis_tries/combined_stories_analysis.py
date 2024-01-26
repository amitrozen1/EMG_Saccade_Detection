import pandas as pd
from General.HelperFunctions import GetSessions
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

keysOfInterest = ["mean_peak_corr", "dtw_pearson_seconds_norm", "dtw_norm_blank_removed"] #["mean_peak_corr", "var_peak_corr", "mean_peak_lag", "var_peak_lag", "dtw_pearson"]
#["mean_corr", "1-mean_recurrence_rate", "1-mean_determinism", "mean_var_corr", "mean_corr_equal_lag", "equal_lag"]#, "new_corr"]
choicesAndRts = ["choiceA", "choiceB", "whoseReading", "rtChoiceA", "rtChoiceB", "choiceA==choiceB", "listenerChoice", "isOther", "chose1", "chose2"]
questionnaireKeys = ["cfa_csv", "cfa_atsc", "iri_cognitive_empathy", "iri_emotional_empathy"]
A = 0
B = 1

def combine_story_strings(storyA, storyB):
    if storyA < storyB:
        return storyA + "_" +  storyB
    else:
        return storyB + "_" + storyA

def get_dict_for_keysOfInterest(df, i):
    out_dict = {}
    for key in keysOfInterest:
        out_dict[key +"_reduce"] = df.iloc[i][key] - df.iloc[i+1][key]
        out_dict[key +"_divide"] = df.iloc[i][key] / df.iloc[i+1][key]
    return out_dict

def normalize_corr_values_per_session(df):
    for session in df["session"].unique():
        for key in keysOfInterest:
            #df.loc[df["session"] == session, key] = np.arctan(df.loc[df["session"] == session, key])
            df.loc[df["session"] == session, key] = (df.loc[df["session"] == session, key] - df.loc[df["session"] == session, key].mean()) / df.loc[df["session"] == session, key].std()


def get_questionnaire_data(session):
    questionnaire = pd.DataFrame(pd.read_csv(r"C:\Liron\neuroscience\PHD\First year experiment - EMG\questionnaires\female_particiapnts_analyzed_withChoice_16_4.csv"))
    data = dict()
    for p in ["A","B"]:
        location = questionnaire["session"] == session+ "_" + p
        if(location.any()):
            row = questionnaire[location].iloc[0]
            for key in questionnaireKeys:
                if (key not in data):
                    data[key] = float(row[key])
                else:
                    data[key] = (data[key] + float(row[key]))/2
        
    return data

def get_questionnaire_data_df_row(df, i):
    session = df.iloc[i]["session"]
    return get_questionnaire_data(session)

def extract_choices_and_rts(df, i):
    out_dict = {}
    out_dict["chose1"] = 0
    out_dict["chose2"] = 0

    out_dict["choiceA"] = 1 - int(df.iloc[i]["isChoiceA"]) # 0 if chose the first story, 1 if chose the second.
    out_dict["choiceB"] = 1 - int(df.iloc[i]["isChoiceB"])

    if( out_dict["choiceA"] == 0):
        out_dict["chose1"] += 1
    else:
        out_dict["chose2"] += 1
    
    if( out_dict["choiceB"] == 0):
        out_dict["chose1"] += 1
    else:
        out_dict["chose2"] += 1


    whose_reading = df.iloc[i]["whoseReading"]
    out_dict["whoseReading"] = whose_reading
    
    if ( whose_reading != 0):
        listener = ["choiceA", "choiceB"][whose_reading % 2]
        out_dict["listenerChoice"] = out_dict[listener]
    else:
        out_dict["listenerChoice"] = -1

    out_dict["isOther"] = int(df.iloc[i]["isOther"])

    out_dict["rtChoiceA"] = df.iloc[i]["rtChoiceA"]
    out_dict["rtChoiceB"] = df.iloc[i]["rtChoiceB"]
    out_dict["choiceA==choiceB"] = out_dict["choiceA"] == out_dict["choiceB"]
    
    return out_dict

def get_just_story_string(story):
    return story.split("_")[1]

data_path = r"C:\Liron\DataEmg\Done"
#csv_path = os.path.join(data_path, "XCorrResults", "Rms20Hz_NoOverLap2_32Lag_BigMat_1000msWindow_RQA_removeNoiseComp_blankRemover_maxCompEachTimeCorr.csv")
#csv_path = os.path.join(data_path, "XCorrResults", "Rms50Hz_NoOverLap_0-1500Lag_BigMat_500msWindow_RQA_removeNoiseComp.csv")
#csv_path = os.path.join(data_path, "XCorrResults", "Rms100Hz_1000msWindow_maxLag1s_dtw_path_and_pearson.csv")
csv_path = os.path.join(data_path, "XCorrResults", "Rms100Hz_1000msWindow_maxLag1s_dtw_path_and_norm_pearson_blankRemoved_fixedcheck_1.1.csv")
combined_csv_path = os.path.splitext(csv_path)[0] + "_combined_stories_with_div.csv"

isCreateCsv = True
isAnalyze = False

if(isCreateCsv):
    file = pd.read_csv(csv_path)
    df = pd.DataFrame(file)
    sessions = GetSessions(data_path)

    #skip the calibration part
    df = df[df["rtChoiceA"] != 0]
    #normalize_corr_values_per_session(df)

    out_df = pd.DataFrame()

    for i in range(0, len(df),2):
        current_dict = {"sesssion" : df.iloc[i]["session"], 
                        "stories" : combine_story_strings(get_just_story_string(df.iloc[i]["key"]), get_just_story_string(df.iloc[i+1]["key"])),
                        "action" : df.iloc[i]["action"]}
        current_dict.update(get_dict_for_keysOfInterest(df, i))
        current_dict.update(extract_choices_and_rts(df, i))
        current_dict.update(get_questionnaire_data_df_row(df, i))
        out_df = pd.concat([out_df, pd.DataFrame(current_dict, index=[0])], ignore_index=False)

    #save the file
    out_df.to_csv(combined_csv_path)

    # todo - normalize the corr values
    # add other corr methods.

    # my anova is not good. if I substract I care if they chose the second or the first, not together.

if(isAnalyze):
    file = pd.read_csv(combined_csv_path)
    df = pd.DataFrame(file)

    df_read = df[df["action"] == "Reading"]
    preds = []
    actuals = []
    for i in range(0, len(df_read)):
        if(df_read.iloc[i]["mean_corr"] > 0.1):
            preds.append(0)
        elif ( df_read.iloc[i]["mean_corr"] > -0.1):
            preds.append(1)
        else:
            preds.append(2)
        
        actuals.append(df_read.iloc[i]["chose2"])

    print(accuracy_score(actuals, preds))
    print(classification_report(actuals, preds))

    df_listen = df[df["action"] == "Listening"]
    preds = []
    actuals = []
    for i in range(0, len(df_listen)):
        if(abs(df_listen.iloc[i]["mean_corr"]) > 0.7):
            preds.append(1)
        else:
            preds.append(0)
        
        actuals.append(df_listen.iloc[i]["isOther"])
    
    print(accuracy_score(actuals, preds))
    print(classification_report(actuals, preds))