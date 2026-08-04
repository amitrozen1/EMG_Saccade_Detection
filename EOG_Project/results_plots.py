import matplotlib.pyplot as plt
import numpy as np
from saccade_detection import classify_direction_by_angle   

def plot_saccade_direction_bar(results_engbert, story_id, title=None, savepath=None):
    ev = results_engbert[story_id]
    saccades = ev.get("saccades_no_blinks") or ev.get("saccades") or []
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
        lbl = 'Detected Saccade' if idx == 0 else None
        plt.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=lbl)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    
    if end_sec is not None: plt.xlim(start_sec, end_sec)
        
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pipeline_validation(h_sig, v_sig, saccades, blinks, fs, title="Pipeline Validation", start_sec=0, end_sec=None):
    time = np.arange(len(h_sig)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # HEOG
    ax1.plot(time, h_sig, color='black', linewidth=1, label='HEOG Signal')
    for idx, (onset, _, offset) in enumerate(saccades):
        lbl = 'Detected Saccade' if idx == 0 else None
        ax1.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=lbl)
    
    ax1.set_ylabel('HEOG Amplitude')
    ax1.set_title(f"{title} - Horizontal Movements")
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
    
def make_all_figures(results_engbert, downsampled_EOG_like, fs_down):
    plot_saccade_direction_bar(results_engbert, "PB_Reading_14.png_1", 
                               title="Saccade directions - Reading 14 (Participant B)",
                               savepath="fig_saccades_reading14_PB.png")

    plot_saccade_direction_bar(results_engbert, "PA_Reading_13.png_2", 
                               title="Saccade directions - Reading 13 (Participant A)",
                               savepath="fig_saccades_reading13_PA.png")

    plot_blinks_trace(downsampled_EOG_like, results_engbert, "PB_Reading_1.png_1", fs_down,
                      title="VEOG signal with detected blinks - Reading 1 (Participant B)",
                      savepath="fig_blinks_reading1_PB.png")
    plt.show()
    
def run_diagnostics(results_engbert, downsampled_EOG_like, fs_down):
    story_id = 'PA_Reading_2.png_4' 
    if story_id in downsampled_EOG_like and story_id in results_engbert:
        h_sig = results_engbert[story_id].get("h_eog_used",downsampled_EOG_like[story_id]["h_eog_lf"])
        v_sig = downsampled_EOG_like[story_id]['v_eog_blink']
        saccades = results_engbert[story_id].get("saccades_no_blinks", [])
        blinks = results_engbert[story_id].get("blinks", [])
        
        plot_saccades_trace(h_sig, saccades, fs_down, title=f"Saccades Trace - {story_id}")
        plot_pipeline_validation(h_sig, v_sig, saccades, blinks, fs_down, title=f"Pipeline Validation - {story_id}")