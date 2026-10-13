import matplotlib.pyplot as plt
import numpy as np
from saccade_detection import classify_direction_by_angle


def plot_reader_listener_overlay(
    reader_signal,
    listener_signal,
    reader_saccades,
    listener_saccades,
    fs,
    title="Speaker vs Listener",
    start_sec=0,
    end_sec=None,
    reader_blinks=None,
    listener_blinks=None,
    savepath=None,
    show=True,
):
    """Plot both participants from the common story onset on one shared time axis."""
    reader_blinks = reader_blinks or []
    listener_blinks = listener_blinks or []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    reader_time = np.arange(len(reader_signal)) / fs
    listener_time = np.arange(len(listener_signal)) / fs

    ax1.plot(reader_time, reader_signal, color="black", linewidth=0.8, label="Speaker HEOG")
    for idx, (onset, _, offset) in enumerate(reader_saccades):
        lbl = "Saccade" if idx == 0 else None
        ax1.axvspan(onset / fs, offset / fs, color="#d62728", alpha=0.28, label=lbl)
    for idx, (onset, offset) in enumerate(reader_blinks):
        lbl = "Blink" if idx == 0 else None
        ax1.axvspan(onset / fs, offset / fs, color="#9467bd", alpha=0.30, label=lbl)
    ax1.set_ylabel("Speaker amplitude")
    ax1.set_title(f"{title} - Speaker")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    ax2.plot(listener_time, listener_signal, color="#1f77b4", linewidth=1, label="Listener HEOG")
    for idx, (onset, _, offset) in enumerate(listener_saccades):
        lbl = "Saccade" if idx == 0 else None
        ax2.axvspan(onset / fs, offset / fs, color="#d62728", alpha=0.28, label=lbl)
    for idx, (onset, offset) in enumerate(listener_blinks):
        lbl = "Blink" if idx == 0 else None
        ax2.axvspan(onset / fs, offset / fs, color="#9467bd", alpha=0.30, label=lbl)
    ax2.set_ylabel("Listener amplitude")
    ax2.set_xlabel("Time from story onset [s]")
    ax2.set_title(f"{title} - Listener")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    if end_sec is not None:
        ax1.set_xlim(start_sec, end_sec)
    else:
        common_end = min(
            reader_time[-1] if len(reader_time) else 0,
            listener_time[-1] if len(listener_time) else 0,
        )
        ax1.set_xlim(start_sec, common_end)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_event_raster(
    reader_saccades,
    listener_saccades,
    fs,
    title="Saccade Event Raster",
    start_sec=0,
    end_sec=None,
    savepath=None,
    show=True,
):
    fig, ax = plt.subplots(figsize=(16, 3))

    for onset, _, offset in reader_saccades:
        ax.axvspan(
            onset / fs,
            offset / fs,
            ymin=0.60,
            ymax=0.90,
            color="#d62728",
            alpha=0.35)

    for onset, _, offset in listener_saccades:
        ax.axvspan(
            onset / fs,
            offset / fs,
            ymin=0.10,
            ymax=0.40,
            color="#1f77b4",
            alpha=0.35)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Listener", "Reader"])
    ax.set_xlabel("Time [s]")
    ax.set_title(title)

    if end_sec is not None:
        ax.set_xlim(start_sec, end_sec)
    else:
        last_event = 0.0
        if reader_saccades:
            last_event = max(last_event, max(offset for _, _, offset in reader_saccades) / fs)
        if listener_saccades:
            last_event = max(last_event, max(offset for _, _, offset in listener_saccades) / fs)
        ax.set_xlim(start_sec, last_event)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig

def plot_saccade_direction_bar(results_engbert, story_id, title=None, savepath=None):
    ev = results_engbert[story_id]
    if "saccades_no_blinks" in ev:
        saccades = ev["saccades_no_blinks"]
    else:
        saccades = ev.get("saccades", [])
    h_vel = ev.get("h_vel")
    v_vel = ev.get("v_vel")

    h_cnt = v_cnt = o_cnt = 0
    for onset, _, offset in saccades:
        label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
        if label == "H": h_cnt += 1
        elif label == "V": v_cnt += 1
        else: o_cnt += 1

    labels = ["Horizontal", "Vertical", "Oblique"]
    values = [h_cnt, v_cnt, o_cnt]

    plt.figure(figsize=(4, 4))
    x = np.arange(len(labels))
    plt.bar(x, values, color=["#5383ec", "#ec5353", "#53c6a8"])
    plt.xticks(x, labels)
    plt.ylabel("Number of saccades")
    plt.title(title if title else f"Saccade directions - {story_id}")
    plt.tight_layout()

    if savepath: plt.savefig(savepath, dpi=300)

def plot_blinks_trace(downsampled_EOG_like, results_engbert, story_id, fs_down, title=None, savepath=None):
    v_sig = np.array(downsampled_EOG_like[story_id].get("v_eog_blink", []), dtype=float)
    blinks = results_engbert[story_id].get("blinks", [])
    t = np.arange(len(v_sig)) / fs_down

    plt.figure(figsize=(8, 3))
    plt.plot(t, v_sig, linewidth=0.8, color="navy")
    
    for (onset, offset) in blinks:
        plt.axvspan(onset / fs_down, offset / fs_down, color="purple", alpha=0.25)

    plt.xlabel("Time [s]")
    plt.ylabel("VEOG-like amplitude")
    plt.title(title if title else f"VEOG signal with blinks - {story_id}")
    plt.tight_layout()

    if savepath: plt.savefig(savepath, dpi=300)
        
def plot_saccades_trace(h_sig, saccades, fs, title="Saccades Trace", start_sec=0, end_sec=None):
    time = np.arange(len(h_sig)) / fs
    
    plt.figure(figsize=(15, 5))
    plt.plot(time, h_sig, label='HEOG Signal', color='black', linewidth=1)
    
    for idx, (onset, _, offset) in enumerate(saccades):
        lbl = 'Retained saccade' if idx == 0 else None
        plt.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=lbl)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    
    if end_sec is not None: plt.xlim(start_sec, end_sec)
        
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pipeline_validation(h_sig, v_sig, saccades, blinks, fs, title="Detection Output", start_sec=0, end_sec=None):
    time = np.arange(len(h_sig)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # HEOG
    ax1.plot(time, h_sig, color='black', linewidth=1, label='HEOG Signal')
    for idx, (onset, _, offset) in enumerate(saccades):
        lbl = 'Retained saccade' if idx == 0 else None
        ax1.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=lbl)
    
    ax1.set_ylabel('HEOG Amplitude')
    ax1.set_title(f"{title} - HEOG with retained saccade events")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # VEOG
    ax2.plot(time, v_sig, color='navy', linewidth=1, label='VEOG Signal')
    for idx, (onset, offset) in enumerate(blinks):
        lbl = 'Detected Blink' if idx == 0 else None
        ax2.axvspan(onset / fs, offset / fs, color='purple', alpha=0.3, label=lbl)
    
    ax2.set_ylabel('VEOG Amplitude')
    ax2.set_xlabel('Time [s]')
    ax2.set_title(f"{title} - Blinks")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    if end_sec is not None: ax1.set_xlim(start_sec, end_sec)
        
    plt.tight_layout()
    plt.show()


def _iter_story_ids(results_engbert, downsampled_EOG_like):
    return [
        story_id
        for story_id in results_engbert
        if story_id in downsampled_EOG_like
    ]


def make_all_figures(results_engbert, downsampled_EOG_like, fs_down):
    for story_id in _iter_story_ids(results_engbert, downsampled_EOG_like):
        plot_saccade_direction_bar(
            results_engbert,
            story_id,
            title=f"Saccade directions - {story_id}",
            savepath=None,
        )
        plt.show()

    for story_id in _iter_story_ids(results_engbert, downsampled_EOG_like):
        plot_blinks_trace(
            downsampled_EOG_like,
            results_engbert,
            story_id,
            fs_down,
            title=f"VEOG signal with detected blinks - {story_id}",
            savepath=None,
        )
        plt.show()

def run_diagnostics(results_engbert, downsampled_EOG_like, fs_down):
    for story_id in _iter_story_ids(results_engbert, downsampled_EOG_like):
        h_sig = results_engbert[story_id].get("h_eog_used", downsampled_EOG_like[story_id]["h_eog_lf"])
        saccades = results_engbert[story_id].get("saccades_no_blinks", [])

        plot_saccades_trace(h_sig, saccades, fs_down, title=f"Saccades Trace - {story_id}")

    for story_id in _iter_story_ids(results_engbert, downsampled_EOG_like):
        h_sig = results_engbert[story_id].get("h_eog_used", downsampled_EOG_like[story_id]["h_eog_lf"])
        v_sig = downsampled_EOG_like[story_id]["v_eog_blink"]
        blinks = results_engbert[story_id].get("blinks", [])

        plot_pipeline_validation(
            h_sig,
            v_sig,
            results_engbert[story_id].get("saccades_no_blinks", []),
            blinks,
            fs_down,
            title=f"Detection Output - {story_id}",
        )