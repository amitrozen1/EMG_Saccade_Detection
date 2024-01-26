import json
import os
import numpy as np

def loadChoiceFromSession(sessionFolder, session):
    '''
    return array that contains
    16 x ['story' , 'whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
    '''
    return loadChoiceJson(os.path.join(sessionFolder, session, "smile", "svc","A.list"))


def loadChoiceFromSessionForPsychData(sessionFolder, session, particiapnt):
    choice_results = loadChoiceJson(os.path.join(sessionFolder, session, "smile", "svc","A.list"))
    chooseTogetherCount = 0
    rt = []  
    for choice_result in choice_results:
        chooseTogetherCount = chooseTogetherCount + choice_result[-1]
        if ( "a" in str.lower(particiapnt)):
            rt.append(choice_result[4]) 
        else:
            rt.append(choice_result[5])
    return [chooseTogetherCount/2 , float(np.mean(rt)), float(np.std(rt)) ]


def loadChoiceJson(jsonFile):
    '''
    return array that contains
    16 x ['story' , 'whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
    '''
    result = []
    with open(jsonFile, 'r') as f:
        data_arrays = json.load(f)
        current_story = 0
        storyKeys = ["story1", "story2"]
        whoseReading = 0
        current_whose_reading = 0
        whose_reading_arr = [1,1,2,2,1,1,2,2]
        for dic in data_arrays:
            if ( "read" in str.lower(dic["key"])):
                whoseReading = whose_reading_arr[current_whose_reading]
                current_whose_reading = current_whose_reading + 1
            elif ("listen" in str.lower(dic["key"])):
                whoseReading = 0
            else:
                continue #smile\frown\blink
            
            choiceA = dic["choiceA"]
            choiceB = dic["choiceB"]
            isChoiceA = int(str(choiceA) == str(current_story+1))
            isChoiceB = int(str(choiceB) == str(current_story+1))
            rtA = float(dic["rtA"])
            rtB = float(dic["rtB"])
            isOther = int(dic["isOther"])
            
            story = dic["key"].split("_")[1].split(".")[0]

            #Todo - Chose together should be 1 when same story is chosen and also when same story is not chosen ?
            choseTogether = int(choiceA == choiceB)
        
            result += [[story, whoseReading,isChoiceA, isChoiceB, rtA, rtB, isOther, choseTogether ]]
            current_story = (current_story + 1) %2
    return result

if __name__ == "__main__":
    #print(len(loadChoiceJson(r"C:\Liron\DataEmg\Done\26122022_1015\smile\svc\A.list")))
    choices = loadChoiceFromSession(r"C:\Liron\DataEmg\Done","06122022_1215")
    for choice in choices:
        if ( choice[0] == '14'):
            print(choice)
            break
