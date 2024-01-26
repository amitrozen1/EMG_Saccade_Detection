import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom_test

def plotDictionaryAsHistogram(dict):
    keys = list(dict.keys())
    values = list(dict.values())
    
    # Compute the mean and standard deviation of the values
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean)**2 for x in values) / len(values))**0.5

    # Compute the 95% confidence interval
    alpha = 0.05
    z_score = 1.96  # for a two-tailed test
    ci_lower = mean - z_score * (std_dev / len(values)**0.5)
    ci_upper = mean + z_score * (std_dev / len(values)**0.5)
    # Create histogram plot
    plt.bar(keys, values)

    # Compute the centers of each bar
    bar_centers = np.arange(len(keys))
    plt.errorbar(bar_centers, values, yerr=z_score*(std_dev / len(values)**0.5), fmt='none', capsize=5, ecolor="red")

    # Set the title and axis labels
    plt.title("Histogram")
    plt.xlabel("story")
    plt.ylabel("Frequency")
    plt.xticks(rotation=20)

    # Display the plot
    plt.show()

def GetThreadIdFromKey(key):
    tid = str.split(key, "_")[-1]


def getChoosingSessionsDf(file_path):
    file = pd.read_csv(file_path)
    df = pd.DataFrame(file)

    #indexes_of_stories = df["rtChoiceA"] != 0
    indexes_of_stories = df["key"].str.contains("Listen")
    return df[indexes_of_stories]

def plotStoriesHistogramsAndBinom(file_path):
    choosingSessions = getChoosingSessionsDf(file_path)
    storyChoicesDic = dict()
    rtDic = dict()

    for i in range(len(choosingSessions)):
        key = choosingSessions.iloc[i]["key"]
        if not key in storyChoicesDic:
            storyChoicesDic[key] = 0
            rtDic[key] = []
        for choice in ["isChoiceA", "isChoiceB"]:
            storyChoicesDic[key] = storyChoicesDic[key] + int(choosingSessions.iloc[i][choice])
        for rt in ["rtChoiceA", "rtChoiceB"]:
            rtDic[key].append(float(choosingSessions.iloc[i][rt]))

    number_of_stories_in_session = 16
    N = (len(choosingSessions)// number_of_stories_in_session)*2

    for key in storyChoicesDic:
        x = storyChoicesDic[key]
        p = binom_test(x, N, p=0.5)
        print(f"{key} : binom-p : {p}, rejected?: {p<0.05}")

    plotDictionaryAsHistogram(storyChoicesDic)
    # for key in rtDic:
    #     rtDic[key] = np.mean(rtDic[key])
    # plotDictionaryAsHistogram(rtDic)


def PlotChoiceHistogramAndBinom(file_path):
    choiceDict = dict()
    choiceDict["First"] = 0
    choiceDict["Second"] = 0

    choosinSessions = getChoosingSessionsDf(file_path)

    for i in range(0, len(choosinSessions), 2):
        #always iterating the firstStory:
        for p in ["isChoiceA", "isChoiceB"]:
            choiceDict["First"] += int(choosinSessions.iloc[i][p])
            choiceDict["Second"] += 1 - int(choosinSessions.iloc[i][p])
    
    p = binom_test(choiceDict["First"], len(choosinSessions), p=0.5)
    print(f"binom-p : {p}, rejected?: {p<0.05}")

    plotDictionaryAsHistogram(choiceDict)
    

if __name__ == "__main__":
    file_path = r"C:\Liron\DataEmg\PsychTask.csv"
    # For each story
    # plotStoriesHistogramsAndBinom(file_path)
    # results - there's a tendency to choose story L16 over L8
    # there's a trend in choosing L12 over L6 and R4 over R2

    # is there a tendency to choose first or second:
    PlotChoiceHistogramAndBinom(file_path)



