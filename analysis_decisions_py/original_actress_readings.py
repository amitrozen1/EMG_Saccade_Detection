from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import load_special_components_if_exists, get_special_component_indexes, save_new_w, get_special_component_names
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts
from scipy.stats import zscore
from General.HelperFunctions import remove_rows_from_matrix
import numpy as np
import os
from EDF.PlotHeatmap import default_heatmap_with_channels_removal, get_heatmap
import pyedflib
from General.PlotFunctions import plotChunkData
from dataclasses import dataclass

rms_size_ms = 100
downsampleWindowInMs = 50
edf_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\EDF\1_Recording_00_SD.edf"
image_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\img_cropped_mirror.png"
correct_value = -0.5 # the correct value is just a random RT value as I did not find a way to get the reall correct value.
# it doesn't matter to fit the actual timing in reality but to the start of the signal when she speaks as I edited the audio files according to that.
#elecs_to_remove = [0] #maybe it's not that bad..

def get_folder_path(p=edf_path):
    return os.path.dirname(p)

def open_w_if_exisis():
    try:
        #get folder path from the edf path:
        return np.load(os.path.join(get_folder_path(), "w.npy"))
    except:
        return None
    
def save_w(w):
    np.save(os.path.join(get_folder_path(), "w.npy"), w)

def is_my_annotation(annotation):
    s = str(annotation).lower()
    return "story_" in s

def extract_story_number(annotation):
    s = str(annotation).lower()
    return int(s.split("_")[1])

def extract_story_name(annotation):
    s = str(annotation).lower()
    #regex to extarct 15 and "start" from : "story_Slide15.PNG_start"
    s = s[len("story_slide"):]
    parts = s.split(".")
    story_number = parts[0]
    annotation_type = parts[1].split("_")[1]

    return story_number, annotation_type

@dataclass
class MyChunk:
    name: str
    start: int
    end: int

def read_annotations(f, freq, downsampleWindowInMs, correction):
    annotations = f.readAnnotations()
    chunks = []
    N = len(annotations[0])
    for i in range(N):
        anotation  = [annotations[0][i], annotations[1][i], annotations[2][i]]
        if is_my_annotation(anotation[2]):
            timing = anotation[0] - correction
            timing = (timing * freq)/(4*downsampleWindowInMs) # 4 because original freq is 4000
            timing = int(timing)

            story_number, annotation_type = extract_story_name(anotation[2])
            if ( annotation_type == "start"):
                chunks.append(MyChunk(story_number, timing, -1))
            if ( annotation_type == "end"):
                chunks[-1].end = timing
    return chunks


def main():
    f = pyedflib.EdfReader(edf_path)
    Y, freq = EdfAnalyzer.readEdf(f, True, False, hard_noise_removal=True)

    chunks = read_annotations(f, freq, downsampleWindowInMs, correct_value)

    #Y = remove_rows_from_matrix(Y, elecs_to_remove)
    Y = zscore(Y, axis=1)

    w = open_w_if_exisis()
    if w is None:
        w, independentComponents = EdfAnalyzer.ICA(Y, 16) #-len(elecs_to_remove))
        save_w(w)
    else:
        independentComponents = np.dot(w, Y)

    independentComponents = remove_emg_artifacts(independentComponents)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

    #plot the heatmap
    #default_heatmap_with_channels_removal(w, elecs_to_remove)
    #get_heatmap(image_path, 16, w)
    #1-based before 50hz + :
    #special_components_indexes = [13,6,8,15]
    #1-based after 50hz + :
    special_components_indexes = [4, 16, 8, 13]
    #0-based:
    special_components_indexes = [x-1 for x in special_components_indexes]

    special_max = np.max(independentComponents, axis=0)
    independentComponents = np.vstack((independentComponents, special_max))

    #plot the special components:
    titles = get_special_component_names() + ["max"]
    #toPlot = independentComponents[ : , chunks[0].start:chunks[0].end]
    toPlot = independentComponents
    ticks = []
    for chunk in chunks:
        ticks.append(chunk.start)
        ticks.append(chunk.end)
    
    plotChunkData(toPlot, "actrss", ticks, titles, num_components=5, show=True)
    


