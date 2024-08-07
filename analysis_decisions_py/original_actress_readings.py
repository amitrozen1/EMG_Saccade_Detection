from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import load_special_components_if_exists, get_special_component_indexes, get_special_component_names
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts
from scipy.stats import zscore
from General.HelperFunctions import remove_rows_from_matrix
import numpy as np
import os
from EDF.PlotHeatmap import default_heatmap_with_channels_removal, get_heatmap
import pyedflib
from General.PlotFunctions import plotChunkData
from dataclasses import dataclass
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from scipy import signal

edf_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\EDF\actress_filtered.edf"
image_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\img_cropped_mirror.png"
correct_value = -0.5 # the correct value is just a random RT value as I did not find a way to get the reall correct value.
# it doesn't matter to fit the actual timing in reality but to the start of the signal when she speaks as I edited the audio files according to that.
#elecs_to_remove = [0] #maybe it's not that bad..

def get_folder_path(p=edf_path):
    return os.path.dirname(p)

def open_w_if_exisis(hard=False, dwt=False):
    if dwt:
        name="w_dwt_db15.npy"
    elif hard:
        name = "w_hard.npy"
    else:
        name = "w.npy"
    try:
        #get folder path from the edf path:
        return np.load(os.path.join(get_folder_path(), name))
    except:
        return None

def open_whiten_if_exisis():
    try:
        return np.load(os.path.join(get_folder_path(), "white.npy"))
    except:
        return None
    
def save_w(w, dwt=False):
    if dwt:
        name="w_dwt_db15.npy"
    else:
        name = "w.npy"
    np.save(os.path.join(get_folder_path(), name), w)

def is_my_annotation(annotation):
    s = str(annotation).lower()
    return "story_" in s

def extract_story_number(annotation):
    s = str(annotation).lower()
    return int(s.split("_")[1])

def parse_annotation(annotation):
    s = str(annotation).lower()
    #regex to extarct 15 and "start" from : "story_Slide15.PNG_start"
    s = s[len("story_slide"):]
    parts = s.split(".")
    story_number = parts[0]
    annotation_type = parts[1].split("_")[1]

    return story_number, annotation_type

@dataclass
class ActressChunk:
    story_number: str
    start: int
    end: int
    data: np.ndarray = None

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

            story_number, annotation_type = parse_annotation(anotation[2])
            if ( annotation_type == "start"):
                chunks.append(ActressChunk(story_number, timing, -1))
            if ( annotation_type == "end"):
                chunks[-1].end = timing
    return chunks


def get_listening_stories():
    return [10, 5, 12, 6, 11, 3, 8, 16]

#todo : 
# 1. check for better filtering for the data.
# 2. choose best components in that case.

def build_actress_chunks(chunks, downsampleWindowInMs):
    chosen_chunk_numbers = get_listening_stories()
    chunks = [chunk for chunk in chunks if int(chunk.story_number) in chosen_chunk_numbers]

    chunks_as_dict = {}
    for chunk in chunks:
        #taking the last instance:
        chunks_as_dict[chunk.story_number] = chunk

    for key in chunks_as_dict:
        print(f"{key}: len: {(chunks_as_dict[key].end - chunks_as_dict[key].start)*downsampleWindowInMs/1000}")

    return chunks_as_dict

def export_actress_chunked_data_with_dwt(downsampleWindowInMs=20, rms_size_ms=100, hard_noise_removal = False):
    f = pyedflib.EdfReader(edf_path)
    Y, freq = EdfAnalyzer.readEdf(f)
    chunks = read_annotations(f, freq, downsampleWindowInMs, correct_value)
    elecs_to_remove = []
    #Y, chunks, freq = EdfAnalyzer.Read(edf_path, correct_value, downsample_window_in_ms=downsampleWindowInMs)
    chunks_as_dict = build_actress_chunks(chunks, downsampleWindowInMs)
    Y = remove_rows_from_matrix(Y, elecs_to_remove)
    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)

    Y = whiten(Y)

    w = open_w_if_exisis(dwt=True)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16 - len(elecs_to_remove), whiten=False)
        save_w(w, dwt=True)
    else:
        independentComponents = np.matmul(w,Y)

    #change according to heatmap.
    special_comps_index = [5, 15, 2, 14, 4, 10, 1] #[8,14,0,9]
    #get_special_component_indexes(w, elecs_to_remove)
    #get_heatmap(image_path, 16, w, electrodes_to_remove=elecs_to_remove)

    independentComponents = independentComponents[special_comps_index, :]
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    hz = int(1000/downsampleWindowInMs)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)
    
    # special_max = np.max(independentComponents, axis=0)
    # independentComponents = np.vstack((independentComponents, special_max))

    for key in chunks_as_dict:
        chunk = chunks_as_dict[key]
        chunks_as_dict[key].data = independentComponents[:, chunk.start:chunk.end]
    
    #     #plot the special components:
    # titles = get_special_component_names() + ["max"]
    # #toPlot = independentComponents[ : , chunks[0].start:chunks[0].end]
    # toPlot = independentComponents
    # ticks = []
    # for chunk in chunks:
    #     ticks.append(chunk.start)
    #     ticks.append(chunk.end)
    
    # plotChunkData(toPlot, "actrss", ticks, titles, num_components=5, show=True)
    
    
    return chunks_as_dict


def export_actress_chunked_data(downsampleWindowInMs=20, rms_size_ms=100, hard_noise_removal = False):
    f = pyedflib.EdfReader(edf_path)
    Y, freq = EdfAnalyzer.readEdf(f, True, False, hard_noise_removal=hard_noise_removal)

    chunks = read_annotations(f, freq, downsampleWindowInMs, correct_value)
    chunks_as_dict = build_actress_chunks(chunks, downsampleWindowInMs)

    Y = zscore(Y, axis=1)

    w = open_w_if_exisis(hard_noise_removal)
    if w is None:
        w, independentComponents = EdfAnalyzer.ICA(Y, 16) #-len(elecs_to_remove))
        save_w(w)
    else:
        independentComponents = np.dot(w, Y)

    #plot the heatmap
    #default_heatmap_with_channels_removal(w, elecs_to_remove)
    get_heatmap(image_path, 16, w)

    #please insrt the special components indexes here:
    #str_input = input("Please insert the special components indexes here and press enter")
    #special_components_indexes = [int(x) for x in str_input.split(" ")]
    #1-based before 50hz + :
    
    if hard_noise_removal:
        special_components_indexes = [4, 16, 8, 15]
    else:
        special_components_indexes = [13, 6, 8, 15]

    #0-based:
    special_components_indexes = [x-1 for x in special_components_indexes]

    independentComponents = independentComponents[special_components_indexes, :]

    independentComponents = remove_emg_artifacts(independentComponents)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

    special_max = np.max(independentComponents, axis=0)
    independentComponents = np.vstack((independentComponents, special_max))


    for key in chunks_as_dict:
        chunk = chunks_as_dict[key]
        chunks_as_dict[key].data = independentComponents[:, chunk.start:chunk.end]
    
    return chunks_as_dict

    # #plot the special components:
    # titles = get_special_component_names() + ["max"]
    # #toPlot = independentComponents[ : , chunks[0].start:chunks[0].end]
    # toPlot = independentComponents
    # ticks = []
    # for chunk in chunks:
    #     ticks.append(chunk.start)
    #     ticks.append(chunk.end)
    
    # plotChunkData(toPlot, "actrss", ticks, titles, num_components=5, show=True)
    


