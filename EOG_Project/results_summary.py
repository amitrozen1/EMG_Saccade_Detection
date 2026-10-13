import numpy as np
from saccade_detection import classify_direction_by_angle

def print_results_table(results_engbert, downsampled_EOG_like, fs):
    print(
        "Story\tDuration_s\tHorizontal\tVertical\tOblique"
        "\tTotal\tBlinks\tRetained_events_per_min"
    )

    for story_id, events in results_engbert.items():
        saccades = events.get(
            "saccades_no_blinks", events.get("saccades", [])
        )
        blinks = events.get("blinks", [])
        h_vel = events.get("h_vel")
        v_vel = events.get("v_vel")

        signal = downsampled_EOG_like[story_id]["v_eog_saccade"]
        duration_sec = len(signal) / fs
        event_rate = (
            len(saccades) * 60 / duration_sec
            if duration_sec > 0 else np.nan
        )

        counts = {"H": 0, "V": 0, "OBL": 0}

        if h_vel is not None and v_vel is not None:
            for onset, _, offset in saccades:
                label = classify_direction_by_angle(
                    h_vel, v_vel, onset, offset
                )
                category = label if label in ("H", "V") else "OBL"
                counts[category] += 1

        print(
            f"{story_id}\t{duration_sec:.2f}\t"
            f"{counts['H']}\t{counts['V']}\t{counts['OBL']}\t"
            f"{len(saccades)}\t{len(blinks)}\t{event_rate:.2f}"
        )
              
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