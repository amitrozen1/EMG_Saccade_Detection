import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'sync_figures')

BECCERA_SHARED_DIR = os.path.join(PROJECT_DIR, 'becerca_code', 'src')
if BECCERA_SHARED_DIR not in sys.path:
    sys.path.insert(0, BECCERA_SHARED_DIR)
    
FREQ = 4000
FS_DOWN = 500

CHANNELS = {
    'v_eog_up': 13,       # above the right eyebrow tail (lateral)
    'v_eog_down': 9,        # below the right eye
    'h_eog_right': 11,       # corner of the right eye
    'ref_l': 15,    # at the top of the forhead (left)
    'ref_r': 14     # at the top of the forhead (right)
}

# Engbert algorithm parameters tailored for Peri-Ocular EMG
ENGBERT_PARAMS = {
    'eta_h': 6,
    'eta_v': 11,
    'smooth_win': 15,
    'min_samples': 10,
    'max_ms': 150,
    'refractory_ms': 15,
    'use_hysteresis': False,
    'edge_frac': 0.3,
}