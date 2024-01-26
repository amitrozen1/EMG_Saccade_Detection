import pandas as pd
import re
from General.HelperFunctions import GetSessions
from PsychData.LoadChoicesJson import loadChoiceFromSessionForPsychData



def listOfQuestionsToHeaders(lst, header):
    return [f"{header}_{x}" for x in lst]

cfaCsvQuestions = listOfQuestionsToHeaders([1, 10, 13, 16, 19, 4, 7], "CFA")
cfaAtsc2Questions = listOfQuestionsToHeaders([11, 12, 14, 15, 17, 18, 2, 20, 3, 5, 6, 8], "CFA")

# Cognitive empathy
iriFantasy = listOfQuestionsToHeaders([1, 5, 7, 12, 16, 23, 26], "IRI")
iriPrespectiveTaking = listOfQuestionsToHeaders([3,8,11,15,21,25,28], "IRI")
# Emotional empathy
iriEmpathicConcern = listOfQuestionsToHeaders([2,4,9,14,18,20,22], "IRI")
iriPersonalDistress = listOfQuestionsToHeaders([6,10,13,17,19,24,27], "IRI")

def handleValueIRI(value):
    if pd.notnull(value):
        return int(value)
    return None

def handleValueCFA(value):
    if pd.notnull(value):
        return int(value)
    #Todo - Ask Yaara
    return None

def handleValueCOI(value):
    if pd.notnull(value):
        return float(value)
    return None

def remove_none_from_list(lst):
    return [x for x in lst if x is not None]

def createSessionFromLine(df, line):
    date = df["Q2"][line]
    date = re.sub(r'[^0-9]', '', date)
    time = df["Q3"][line]
    time = re.sub(r'[^0-9]', '', time)
    if ( len(time) == 3 ):
        time = "0" + time
    particiapnt = str.upper(df["Q4"][line])
    return f"{date}_{time}_{particiapnt}"

def computeCFAFromLine(df, line):
    csvAns = [handleValueCFA(df[x][line]) for x in cfaCsvQuestions]
    atscAns = [handleValueCFA(df[x][line]) for x in cfaAtsc2Questions]
    csvAns = remove_none_from_list(csvAns)
    atscAns = remove_none_from_list(atscAns)
    #return iriCSV, iriAtsc
    return sum(csvAns)/len(csvAns), sum(atscAns)/len(atscAns)

def computeIRIFromLine(df,line):
    # Cognitive empathy
    fantasy = [handleValueIRI(df[x][line]) for x in iriFantasy ]
    prespectiveTaking = [handleValueIRI(df[x][line]) for x in iriPrespectiveTaking ]
    # Emotional empathy
    empathicConcern = [handleValueIRI(df[x][line]) for x in iriEmpathicConcern ]
    personalDistress = [handleValueIRI(df[x][line]) for x in iriPersonalDistress ]
    #return CognitiveEmpathy, EmotionalEmpathy
    fantasy = remove_none_from_list(fantasy)
    prespectiveTaking = remove_none_from_list(prespectiveTaking)
    empathicConcern = remove_none_from_list(empathicConcern)
    personalDistress = remove_none_from_list(personalDistress)

    return (sum(fantasy) + sum(prespectiveTaking))/(len(fantasy) + len(prespectiveTaking)) , (sum(empathicConcern) + sum(personalDistress))/(len(empathicConcern) + len(personalDistress))

def GetIosScore(df,line):
    return float(df["iosScoreSet1"][line])

def GetTrust(df,line):
    # both reported after the experiment
    trust_after = df["Trust_1"][line]
    trust_before = df["Trust_2"][line]
    
    #return trust_after, trust_before
    return trust_after, trust_before

def GetStatsHeaderWithouSession(filename):
    file = pd.read_csv(filename)
    df = pd.DataFrame(file)
    return df.columns.tolist()[1:]

def GetStatsRowsFromSession(filename, session):
    file = pd.read_csv(filename)
    df = pd.DataFrame(file)
    result = []

    for p in ["A", "B"]:
        session_p = f"{session}_{p}"
        indexlist = df.index[df["session"] == session_p]
        if ( len(indexlist) == 0 ):
            return False
        index = int(indexlist[0])
        result.append(df.iloc[index].tolist()[1:])
    return result

# if __name__ == "__main__":
#     output = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\questionnaires\female_particiapnts_analyzed.csv"
#     print(GetStatsRowsFromSession(output, "04122022_1300"))

if __name__ == "__main__":
    filename = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\questionnaires\Survey+After+EMG+Experiment+-+F_April+16,+2023_00.16.csv"
    session_folder = r"C:\Liron\DataEmg\Try"
    sessions = GetSessions(session_folder)

    document = pd.read_csv(filename)
    df = pd.DataFrame(document)
    output = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\questionnaires\female_particiapnts_analyzed_withChoice_16_4_30First.csv"
    results = []

    for i in range(7, len(df)):
        if(pd.isnull(df.iloc[i]["Q2"])):
            continue
        key = createSessionFromLine(df,i)
        if ( key[0:-2] in sessions):
            ios = GetIosScore(df,i)
            CFA_csv, CFA_atsc = computeCFAFromLine(df,i)
            IRI_cognitiveEmpathy, IRI_EmotionalEmpathy = computeIRIFromLine(df,i)
            trust_after, trust_before = GetTrust(df, i)
            results += [
                    [key, ios, CFA_csv, CFA_atsc, IRI_cognitiveEmpathy, IRI_EmotionalEmpathy, trust_after, trust_before]
                    + loadChoiceFromSessionForPsychData(session_folder, key[0:-2], key[-1])]

    results_df = pd.DataFrame(results)
    results_df.columns = ['session', 'ios', 'cfa_csv', 'cfa_atsc', 'iri_cognitive_empathy', 
        'iri_emotional_empathy', 'trust_after', 'trust_before'] + ['chooseTogetherCount', 
        'mean_rt', 'std_rt']
    results_df.to_csv(output, index=False)

