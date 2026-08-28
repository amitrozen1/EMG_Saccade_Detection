import os
import sys
import argparse

import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
code_dir = os.path.join(project_dir, 'emg_data_analysis')
data_dir = os.path.join(os.path.dirname(__file__), 'data')

sys.path.append(project_dir)
sys.path.append(code_dir)
sys.stdout.reconfigure(encoding='utf-8')

from filtering import load_pkl_file, downsample_signal, lowpass_filter
from differentiation import differentiate
from saccade_detection import (detect_saccades_engbert_2d,detect_blinks,remove_saccades_overlapping_blinks,pick_best_heog)
from extract_listening_data import save_listening_data
from results_plots_review import plot_reader_listener_overlay


freq = 4000
fs_down = 500

channels = {
    'v_eog_up': 13,
    'v_eog_down': 9,
    'h_eog_right': 11,
    'ref_l': 15,
    'ref_r': 14,}


def _valid_listener_data(path, expected_story_ids):
    if not os.path.exists(path):
        return False
    try:
        data = load_pkl_file(path)
    except (EOFError, OSError, ValueError):
        return False

    if not expected_story_ids.issubset(data):
        return False
    return all(
        np.asarray(data[story_id].data).ndim == 2
        and np.asarray(data[story_id].data).shape[0] > max(channels.values())
        and np.asarray(data[story_id].data).shape[1] > 0
        for story_id in expected_story_ids)


def load_pair_data():
    listening_a_path = os.path.join(data_dir, 'participant_A_listening_EOG.pkl')
    listening_b_path = os.path.join(data_dir, 'participant_B_listening_EOG.pkl')

    reading = {'A': load_pkl_file(os.path.join(data_dir, 'participant_A_readings_EOG.pkl')),
                'B': load_pkl_file(os.path.join(data_dir, 'participant_B_readings_EOG.pkl')),}

    expected_a = set(reading['B'])
    expected_b = set(reading['A'])
    if not _valid_listener_data(listening_a_path, expected_a) or not _valid_listener_data(listening_b_path, expected_b):
        print('Listening pickle files are missing or invalid; extracting them from EDF...')
        save_listening_data()

    listening = {
        'A': load_pkl_file(listening_a_path),
        'B': load_pkl_file(listening_b_path),
    }

    resolved = []
    for reader_participant, listener_participant in (('A', 'B'), ('B', 'A')):
        shared_stories = set(reading[reader_participant]) & set(listening[listener_participant])
        for story_id in sorted(shared_stories):
            resolved.append((reader_participant, story_id, listener_participant, story_id))

    if not resolved:
        raise RuntimeError('No matching Speaker/Listener story IDs were found')

    return reading, listening, resolved


def build_eog_views(chunk_dict):
    outputs = {}

    for story_id, chunk in chunk_dict.items():
        chunk_data = np.asarray(chunk.data)
        if chunk_data.ndim != 2 or chunk_data.shape[0] <= max(channels.values()) or chunk_data.shape[1] == 0:
            raise ValueError(f'{story_id}: invalid EOG chunk shape {chunk_data.shape}')
        story_signals = {ch: chunk_data[ch] for ch in channels.values()}

        v_eog = story_signals[channels['v_eog_up']] - story_signals[channels['v_eog_down']]
        h_eog_lf = story_signals[channels['h_eog_right']] - story_signals[channels['ref_l']]
        h_eog_rf = story_signals[channels['h_eog_right']] - story_signals[channels['ref_r']]

        pseudo = {
            'v_eog_saccade': v_eog,
            'v_eog_blink': v_eog,
            'h_eog_lf': h_eog_lf,
            'h_eog_rf': h_eog_rf,
        }

        filtered = {}
        for eog_type, signal_data in pseudo.items():
            cutoff = 15 if 'v_eog_blink' in eog_type else 25
            filtered[eog_type] = lowpass_filter(signal_data, fs=freq, cutoff=cutoff, order=4)

        downsampled = {name: downsample_signal(sig, freq, fs_down) for name, sig in filtered.items()}

        h_eog_raw = downsampled.get('h_eog_lf')
        h_eog_alt = downsampled.get('h_eog_rf')
        if h_eog_raw is not None and h_eog_alt is not None:
            h_eog = pick_best_heog(h_eog_raw, h_eog_alt, fs_down)
        else:
            h_eog = h_eog_raw

        time_step = 1.0 / fs_down
        v_eog_sac = downsampled.get('v_eog_saccade')
        v_eog_blink = downsampled.get('v_eog_blink')

        result = {
            'h_eog_used': h_eog,
            'v_eog_used': v_eog_blink,
            'downsampled': downsampled,
            'saccades': [],
            'saccades_no_blinks': [],
            'blinks': [],
        }

        if v_eog_sac is not None and h_eog is not None:
            v_vel = differentiate(v_eog_sac.astype(np.float64), step=time_step, method='l11')
            h_vel = differentiate(h_eog.astype(np.float64), step=time_step, method='l11')

            saccades = detect_saccades_engbert_2d(
                h_vel,
                v_vel,
                fs=fs_down,
                eta_h=6,
                eta_v=11,
                smooth_win=15,
                min_samples=10,
                max_ms=150,
                refractory_ms=15,
                use_hysteresis=False,
                edge_frac=0.3,
            )
            result.update({'saccades': saccades, 'h_vel': h_vel, 'v_vel': v_vel})

            if v_eog_blink is not None:
                v_blink_diff = differentiate(v_eog_blink.astype(np.float64), step=time_step, method='l11')
                blinks = detect_blinks(v_eog_blink.astype(float), v_blink_diff.astype(float), fs_down)
                result['blinks'] = blinks
                result['saccades_no_blinks'] = remove_saccades_overlapping_blinks(saccades, blinks, fs=fs_down)
            else:
                result['blinks'] = []
                result['saccades_no_blinks'] = saccades

        outputs[story_id] = result

    return outputs


def event_lag_score(reader_saccades, listener_saccades, fs, max_lag_sec=3.0):
    if not reader_saccades or not listener_saccades:
        return None, None

    reader_onsets = np.array([onset for onset, _, _ in reader_saccades], dtype=float) / fs
    listener_onsets = np.array([onset for onset, _, _ in listener_saccades], dtype=float) / fs

    lags = []
    for reader_time in reader_onsets:
        best = listener_onsets[np.argmin(np.abs(listener_onsets - reader_time))]
        lag = best - reader_time
        if abs(lag) <= max_lag_sec:
            lags.append(lag)

    if not lags:
        return None, None

    return float(np.mean(lags)), float(np.std(lags))


def _safe_filename(value):
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in value)


def main(output_dir=None, show_plots=True, selected_story=None):
    reading, listening, pairs = load_pair_data()

    if selected_story:
        pairs = [pair for pair in pairs if pair[1] == selected_story]
        if not pairs:
            raise ValueError(f'Story {selected_story!r} was not found in a matching pair')

    output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'sync_figures')
    os.makedirs(output_dir, exist_ok=True)
    saved_figures = []

    for reader_participant, reader_key, listener_participant, listener_key in pairs:
        reader_story = build_eog_views({reader_key: reading[reader_participant][reader_key]})[reader_key]
        listener_story = build_eog_views({listener_key: listening[listener_participant][listener_key]})[listener_key]

        reader_saccades = reader_story.get('saccades_no_blinks', reader_story.get('saccades', []))
        listener_saccades = listener_story.get('saccades_no_blinks', listener_story.get('saccades', []))
        reader_blinks = reader_story.get('blinks', [])
        listener_blinks = listener_story.get('blinks', [])

        mean_lag, std_lag = event_lag_score(reader_saccades, listener_saccades, fs_down)
        print(
            f'{reader_participant}:{reader_key} -> {listener_participant}:{listener_key} | '
            f'reader_saccades={len(reader_saccades)} listener_saccades={len(listener_saccades)} | '
            f'reader_blinks={len(reader_blinks)} listener_blinks={len(listener_blinks)} | '
            f'mean_lag={mean_lag} std_lag={std_lag}'
        )

        title = f'Speaker {reader_participant} vs Listener {listener_participant}: {reader_key}'
        savepath = os.path.join(
            output_dir,
            _safe_filename(f'sync_{reader_participant}_{listener_participant}_{reader_key}.png'),
        )
        fig = plot_reader_listener_overlay(
            reader_story['h_eog_used'],
            listener_story['h_eog_used'],
            reader_saccades,
            listener_saccades,
            fs_down,
            title=title,
            reader_blinks=reader_blinks,
            listener_blinks=listener_blinks,
            savepath=savepath,
            show=show_plots,
        )
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.close(fig)
        saved_figures.append(savepath)

    print(f'Saved {len(saved_figures)} synchronization figures to {output_dir}')
    return saved_figures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze synchronized Speaker/Listener EOG stories')
    parser.add_argument('--story', help='Analyze one story ID, for example Reading_2.png_4')
    parser.add_argument('--output-dir', help='Directory for synchronization figures')
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening plot windows')
    args = parser.parse_args()
    main(output_dir=args.output_dir, show_plots=not args.no_show, selected_story=args.story)
