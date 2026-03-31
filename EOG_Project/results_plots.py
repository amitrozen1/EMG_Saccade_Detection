import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from saccade_detection import classify_direction_by_angle   


def plot_saccade_distribution(df):
    plt.figure(figsize=(10,6))
    bar_width = 0.25
    x = range(len(df))

    plt.bar([i - bar_width for i in x], df["Horizontal"], width=bar_width, label="Horizontal")
    plt.bar(x, df["Vertical"], width=bar_width, label="Vertical")
    plt.bar([i + bar_width for i in x], df["Oblique"], width=bar_width, label="Oblique")

    plt.xticks(x, df["Story"], rotation=45)
    plt.ylabel("Number of saccades")
    plt.title("Saccade Direction Distribution per Story")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_saccade_distribution_for_story(story_id, h, v, o):
    labels = ["Horizontal", "Vertical", "Oblique"]
    values = [h, v, o]
    
    plt.figure(figsize=(5,4))
    plt.bar(labels, values, color=["#5383ec", "#ec5353", "#53c6a8"])
    plt.title(f"Saccade Direction Distribution – {story_id}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    

def plot_saccade_direction_bar(results_engbert, story_id, title=None, savepath=None):
    """
    Bar plot for number of horizontal / vertical / oblique saccades
    in a single story, using saccades_no_blinks.
    """
    if story_id not in results_engbert:
        raise ValueError(f"Story ID {story_id!r} not found in results_engbert")

    ev = results_engbert[story_id]

    saccades = ev.get("saccades_no_blinks") or ev.get("saccades") or []
    h_vel = ev.get("h_vel", None)
    v_vel = ev.get("v_vel", None)

    if h_vel is None or v_vel is None:
        raise ValueError(f"No velocity signals stored for story {story_id!r}")

    h_cnt = 0
    v_cnt = 0
    o_cnt = 0

    for onset, pk, offset in saccades:
        label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
        if label == "H":
            h_cnt += 1
        elif label == "V":
            v_cnt += 1
        else:
            o_cnt += 1

    labels = ["Horizontal", "Vertical", "Oblique"]
    values = [h_cnt, v_cnt, o_cnt]

    plt.figure(figsize=(4, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("Number of saccades")
    if title is None:
        title = f"Saccade directions – {story_id}"
    plt.title(title)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)


def plot_blinks_trace(downsampled_EOG_like, results_engbert, story_id, fs_down,
                      title=None, savepath=None):
    """
    Plot VEOG-like signal with blink intervals highlighted.
    """
    if story_id not in downsampled_EOG_like:
        raise ValueError(f"Story ID {story_id!r} not found in downsampled_EOG_like")
    if story_id not in results_engbert:
        raise ValueError(f"Story ID {story_id!r} not found in results_engbert")

    signals = downsampled_EOG_like[story_id]
    if "v_eog_blink" not in signals:
        raise ValueError(f"No 'v_eog_blink' channel for story {story_id!r}")

    v_sig = np.asarray(signals["v_eog_blink"], float)
    blinks = results_engbert[story_id].get("blinks", [])

    t = np.arange(len(v_sig)) / fs_down

    plt.figure(figsize=(8, 3))
    plt.plot(t, v_sig, linewidth=0.8, color="navy")
    
    for (onset, offset) in blinks:
        t0 = onset / fs_down
        t1 = offset / fs_down
        plt.axvspan(t0, t1, color="purple", alpha=0.25)

    plt.xlabel("Time [s]")
    plt.ylabel("VEOG-like amplitude")
    if title is None:
        title = f"VEOG-like signal with detected blinks – {story_id}"
    plt.title(title)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        
def plot_saccades_trace(h_sig, saccades, fs, title="Saccades Trace", start_sec=0, end_sec=None):
    """
    Plots the horizontal EOG signal and highlights detected saccades.
    """
    time = np.arange(len(h_sig)) / fs
    
    plt.figure(figsize=(15, 5))
    # ציור הסיגנל האופקי
    plt.plot(time, h_sig, label='HEOG Signal', color='black', linewidth=1)
    
    # הוספת רקע אדום לכל סאקדה (מה-onset ועד ה-offset)
    for idx, (onset, pk, offset) in enumerate(saccades):
        # שמים לייבל רק בראשון כדי לא להציף את המקרא
        label = 'Detected Saccade' if idx == 0 else None
        plt.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=label)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [a.u.]')
    plt.title(title)
    
    # אפשרות להתמקד על חלק מהזמן אם רוצים
    if end_sec is not None:
        plt.xlim(start_sec, end_sec)
        
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pipeline_validation(h_sig, v_sig, saccades, blinks, fs, title="Pipeline Validation", start_sec=0, end_sec=None):
    """
    Creates a dual-plot figure aligning HEOG (with saccades) and VEOG (with blinks) 
    on the exact same time axis to validate the detection pipeline.
    """
    time = np.arange(len(h_sig)) / fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # --- Top Plot: HEOG and Saccades ---
    ax1.plot(time, h_sig, color='black', linewidth=1, label='HEOG Signal')
    for idx, (onset, pk, offset) in enumerate(saccades):
        lbl = 'Detected Saccade' if idx == 0 else None
        ax1.axvspan(onset / fs, offset / fs, color='red', alpha=0.3, label=lbl)
    
    ax1.set_ylabel('HEOG Amplitude [a.u.]')
    ax1.set_title(f"{title} - Horizontal Eye Movements")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom Plot: VEOG and Blinks ---
    ax2.plot(time, v_sig, color='navy', linewidth=1, label='VEOG Signal')
    for idx, (onset, offset) in enumerate(blinks):
        lbl = 'Detected Blink' if idx == 0 else None
        ax2.axvspan(onset / fs, offset / fs, color='purple', alpha=0.3, label=lbl)
    
    ax2.set_ylabel('VEOG Amplitude [a.u.]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title(f"{title} - Vertical Eye Movements (Blinks)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Zoom option
    if end_sec is not None:
        ax1.set_xlim(start_sec, end_sec)
        
    plt.tight_layout()
    plt.show()
    
def make_all_figures(results_engbert, downsampled_EOG_like, fs_down):
    """
    Creates 3 figures:
    1) Bar plot – Reading 14, Participant B  (PB_)
    2) Bar plot – Reading 13, Participant A  (PA_)
    3) Blink trace – Reading 15, Participant B
    """

    story_bar_1 = "PB_Reading_14.png_1"  
    story_bar_2 = "PA_Reading_13.png_2"   
    story_blinks = "PB_Reading_15.png_3"  

    plot_saccade_direction_bar(
        results_engbert,
        story_bar_1,
        title="Saccade directions – Reading 14 (Participant B)",
        savepath="fig_saccades_reading14_PB.png",
    )

    plot_saccade_direction_bar(
        results_engbert,
        story_bar_2,
        title="Saccade directions – Reading 13 (Participant A)",
        savepath="fig_saccades_reading13_PA.png",
    )

    plot_blinks_trace(
        downsampled_EOG_like,
        results_engbert,
        story_blinks,
        fs_down=fs_down,
        title="VEOG-like signal with detected blinks – Reading 15 (Participant B)",
        savepath="fig_blinks_reading15_PB.png",
    )
    
    plt.show()
