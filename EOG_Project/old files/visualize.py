import matplotlib.pyplot as plt
import numpy as np


def plot_signal_with_saccades(signal, velocity, saccades, sampling_rate):
    time = np.arange(len(signal)) / sampling_rate

    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, label='Signal', color='blue')
    plt.plot(time, velocity, label='Velocity', color='green', alpha=0.4)

    for start, end in saccades:
        plt.axvspan(time[start], time[end], color='red', alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Saccade Detection using Velocity Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def stage_stats(x, name="signal"):
    x = np.asarray(x)
    n = x.size
    n_nan = np.isnan(x).sum()
    n_inf = np.isinf(x).sum()
    finite = np.isfinite(x)
    print(f"[{name}] shape={x.shape}, dtype={x.dtype}, "
          f"NaN={n_nan}, Inf={n_inf}, finite%={finite.mean()*100:.2f}%")
    if finite.any():
        xf = x[finite]
        print(f"[{name}] min={xf.min():.6g}, max={xf.max():.6g}, mean={xf.mean():.6g}, std={xf.std(ddof=1):.6g}")
    else:
        print(f"[{name}] *** no finite values ***")

def downsample_for_plot(x, max_points=5000):
    """שומר על הצורה, רק מדלל כדי שהגרף יהיה מהיר. לא משפיע על החישובים, רק על הצגה."""
    x = np.asarray(x)
    if x.size <= max_points:
        return x, 1
    step = int(np.ceil(x.size / max_points))
    return x[::step], step

def plot_signal_stage(x, fs, title="Stage", show_nan_mask=True, max_points=5000):
    """
    מצייר את האות (עם דילול נקודות להצגה מהירה),
    ובאופציה מצייר למטה מסכה של NaN כדי לראות היכן יש חוסרים.
    """
    x = np.asarray(x)
    xs, step = downsample_for_plot(x, max_points=max_points)
    # על ציר הזמן נשתמש בקפיצה המתאימה אם דיללנו
    t = np.arange(xs.size) * (step / fs)

    plt.figure(figsize=(12, 4))
    plt.plot(t, xs)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if show_nan_mask:
        mask = np.isnan(x)
        if mask.any():
            ms, _ = downsample_for_plot(mask.astype(float), max_points=max_points)
            plt.figure(figsize=(12, 2.5))
            plt.plot(np.arange(ms.size) * (step / fs), ms, drawstyle="steps-mid")
            plt.title(f"NaN mask for: {title} (1=NaN)")
            plt.xlabel("Time [s]")
            plt.yticks([0,1])
            plt.grid(True)
            plt.tight_layout()
            plt.show()
