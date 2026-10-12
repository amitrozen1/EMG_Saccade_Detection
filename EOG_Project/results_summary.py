import numpy as np
from saccade_detection import classify_direction_by_angle

def print_results_table(results_engbert, downsampled_EOG_like):
    print("Story\tHorizontal\tVertical\tOblique\tTotal\tBlinks")

    for story_id, events in results_engbert.items():
        saccades = events["saccades_no_blinks"] if "saccades_no_blinks" in events else events.get("saccades", [])
        blinks = events.get("blinks", [])
        h_vel = events.get("h_vel", None)
        v_vel = events.get("v_vel", None)

        h_cnt = v_cnt = o_cnt = 0

        if h_vel is not None and v_vel is not None:
            for onset, pk, offset in saccades:
                label = classify_direction_by_angle(h_vel, v_vel, onset, offset)
                if label == "H": h_cnt += 1
                elif label == "V": v_cnt += 1
                else: o_cnt += 1

        print(f"{story_id}\t{h_cnt}\t{v_cnt}\t{o_cnt}\t{len(saccades)}\t{len(blinks)}")
      
def print_saccade_sequence(
    results_engbert, story_id, num_events=None, fs=None
):
    if story_id not in results_engbert:
        print(f"Error: Story {story_id} not found.")
        return

    ev = results_engbert[story_id]
    saccades = sorted(
        ev.get("saccades_no_blinks", []),
        key=lambda event: event[0],
    )
    h_vel = ev.get("h_vel")
    v_vel = ev.get("v_vel")

    if h_vel is None or v_vel is None:
        print(f"{story_id}: velocity signals are missing.")
        return

    if num_events is not None:
        saccades = saccades[:num_events]

    sequence = []
    rows = []

    for index, (onset, _, offset) in enumerate(saccades, start=1):
        label = classify_direction_by_angle(
            h_vel, v_vel, onset, offset
        )

        if label == "H":
            dx = np.sum(h_vel[onset:offset + 1])
            if dx > 0:
                label = "H(R)"
            elif dx < 0:
                label = "H(L)"
            else:
                label = "H(?)"

        sequence.append(label)

        if fs is not None:
            rows.append(
                f"{index:>3}   "
                f"{onset / fs:>9.3f}   "
                f"{offset / fs:>9.3f}   "
                f"{label}"
            )

    print(f"\n=== Sequence for {story_id} ===")
    print(f"Retained events shown: {len(saccades)}")
    print(" -> ".join(sequence) if sequence else "No retained events.")

    if fs is not None and rows:
        print("\n No.    Onset [s]   Offset [s]   Label")
        print("\n".join(rows))

    print("================================================\n")