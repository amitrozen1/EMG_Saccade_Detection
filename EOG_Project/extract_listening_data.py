from __future__ import annotations

import os
import pickle
import sys

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from EDF.EdfAnalyzer import EdfAnalyzer
from Hilas_component_classification.classifying_ica_components import wavelet_denoising

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def save_listening_data(session_prefix: str = "04122022_1545"):
    session_edfs = {"A": os.path.join(DATA_DIR, f"{session_prefix}_A_SD_filtered.edf"),
        "B": os.path.join(DATA_DIR, f"{session_prefix}_B_SD_filtered.edf")}
    
    participants = ['A', 'B']
    reading_order = ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']

    for p in participants:
        edf_path = session_edfs[p]
        if not os.path.exists(edf_path):
            alt_edf_path = os.path.join(PROJECT_DIR, "..", "data", f"{session_prefix}_{p}_SD_filtered.edf")
            if os.path.exists(alt_edf_path):
                edf_path = alt_edf_path
            else:
                raise FileNotFoundError(f"Could not find EDF file for participant {p} at: {edf_path}")

        print(f"Reading EDF for participant {p} using EdfAnalyzer.Read...")
        Y, chunks, freq = EdfAnalyzer.Read(edf_path, 0.0, downsample_window_in_ms=1, forEEG=True)
        
        print(f"Applying wavelet denoising (db15) for participant {p}...")
        Y = wavelet_denoising(Y, freq, 'db15')

        keys = [k for k in chunks.keys() if "reading" in k.lower()]
        filtered_chunks = {k: chunks[k] for k in keys}

        listener_chunks = dict()
        other_p = 'B' if p == 'A' else 'A'

        for i, key in enumerate(filtered_chunks.keys()):
            if i < len(reading_order) and reading_order[i] == other_p:
                current_chunk = filtered_chunks[key]
                start_idx = int(current_chunk.Start.Time)
                end_idx = int(current_chunk.End.Time)
                current_chunk.data = Y[:, start_idx:end_idx]
                current_chunk.Data = Y[:, start_idx:end_idx]
                listener_chunks[key] = current_chunk

        out_path = os.path.join(DATA_DIR, f"participant_{p}_listening_EOG.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(listener_chunks, f)
        print(f"Saved participant {p} listening windows to {out_path}")

if __name__ == "__main__":
    save_listening_data()