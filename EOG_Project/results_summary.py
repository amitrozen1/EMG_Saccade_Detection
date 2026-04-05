# results_summary.py

import numpy as np
from saccade_detection import classify_direction_by_angle

def print_results_table(results_engbert, downsampled_EOG_like):
    """
    Prints a summary table of saccade directions and blink counts per story.
    Story    Horizontal   Vertical   Oblique   Total   Blinks

    Uses saccades_no_blinks if available; otherwise uses original saccades.
    """
    print("Story\tHorizontal\tVertical\tOblique\tTotal\tBlinks")

    for story_id, events in results_engbert.items():
        saccades = events.get("saccades_no_blinks") or events.get("saccades") or []
        blinks = events.get("blinks", [])
        h_vel = events.get("h_vel", None)
        v_vel = events.get("v_vel", None)

        h_cnt = 0
        v_cnt = 0
        o_cnt = 0

        if h_vel is not None and v_vel is not None:
            for onset, pk, offset in saccades:
                label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
                if label == "H":
                    h_cnt += 1
                elif label == "V":
                    v_cnt += 1
                else:
                    o_cnt += 1

        total = len(saccades)
        n_blinks = len(blinks)

        print(f"{story_id}\t{h_cnt}\t{v_cnt}\t{o_cnt}\t{total}\t{n_blinks}")

def print_saccade_sequence(results_engbert, story_id, num_events=40):
    """
    Prints the sequence of saccade directions (e.g., H -> H -> V) 
    to prove the cognitive pattern of reading.
    """
    if story_id not in results_engbert:
        print(f"Error: Story {story_id} not found in results.")
        return

    ev = results_engbert[story_id]
    saccades = ev.get("saccades_no_blinks", [])
    h_vel = ev.get("h_vel")
    v_vel = ev.get("v_vel")
    
    sequence = []
    if h_vel is not None and v_vel is not None:
        for onset, pk, offset in saccades:
            label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
            sequence.append(label)
            
        print(f"\n=== Saccade Sequence for {story_id} ===")
        print(" -> ".join(sequence[:num_events]))
        print("================================================\n")