import os
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer
from Hilas_component_classification.classifying_ica_components import wavelet_denoising
import pickle

def load_single_data(session, p):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(get_data_path(), session))
    #now 0.1 - 30HZ
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_in_ms=1, forEEG=True)
    Y = wavelet_denoising(Y, freq, 'db15')

    reading_order = [A, A, B, B, A, A, B, B]

    #take only when keys contains "RS" or "RE":
    keys = [k for k in chunks.keys() if "reading" in k.lower()]
    chunks = {k: chunks[k] for k in keys}

    participants_readings = dict()

    for i, key in enumerate(chunks.keys()):
        if reading_order[i] == p:
            current_chunk = chunks[key]
            current_chunk.data = Y[:, int(current_chunk.Start.Time):int(current_chunk.End.Time)]
            participants_readings[key] = current_chunk
    
    return participants_readings

def save_participant_readings_as_pkl(session, p):
    participant_readings = load_single_data(session, p)
    #save whole dictionary as pkl file
    path = os.path.join(get_data_path(), session, f"participant_{p}_readings_EOG.pkl")

    with open(path, 'wb') as f:
        pickle.dump(participant_readings, f)
    print(f"Saved participant {p} readings to {path}")
