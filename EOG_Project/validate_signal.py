import numpy as np
import matplotlib.pyplot as plt

from saccades_detection import find_saccades_on_channel
from filtering import interp_nans
from differentiation import differentiate  # method='l11'


def qc_plot_signal_and_velocity(signal, fs, *,method='l11', k=3.0, low_frac=0.3, min_dur_s=0.010, title_prefix=""):
    """
    Plots the signal over time, the velocity (derivative) over time with thresholds and highlighted saccades.  
    """

    x = np.asarray(signal, dtype=float)

    # handle NaNs if any
    if np.isnan(x).any() and interp_nans is not None:
        x = interp_nans(x)

    # find saccades (using μ+k·σ + histerzis low_frac)
    res = find_saccades_on_channel(x, fs=fs, method=method, k=k, low_frac=low_frac, min_dur_s=min_dur_s)
    onsets  = res["onsets"]
    offsets = res["offsets"]
    thr_hi  = res["thr_high"]
    thr_lo  = res["thr_low"]

    v = differentiate(x, 1.0/fs, method)
    t = np.arange(x.size) / fs
    speed = np.abs(v)

    # graph plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # signal in time
    axs[0].plot(t, x, lw=1)
    axs[0].set_title(f"{title_prefix} Signal (fs={fs} Hz)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)

    # highlight detected saccades windows
    for a, b in zip(onsets, offsets):
        axs[0].axvspan(a/fs, b/fs, color='orange', alpha=0.2)

    # velocity and speed in time + thresholds + detected saccades
    axs[1].plot(t, speed, lw=1)
    axs[1].axhline(thr_hi, ls='--', lw=1, label=f"High thr = {thr_hi:.2f}")
    axs[1].axhline(thr_lo, ls=':',  lw=1, label=f"Low thr = {thr_lo:.2f}")

    for a, b in zip(onsets, offsets):
        axs[1].axvspan(a/fs, b/fs, color='orange', alpha=0.2)

    axs[1].set_title(f"{title_prefix} | Velocity | method={method}, k={k}, low_frac={low_frac}")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("|Velocity| (a.u./s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # print stats
    n = int(onsets.size)
    if n:
        dur_ms = (offsets - onsets) * (1000.0 / fs)
        print(f"[QC] events={n}  thr_hi={thr_hi:.2f} thr_lo={thr_lo:.2f}  "
              f"dur(ms) median={np.median(dur_ms):.1f}")
    else:
        print(f"[QC] events=0  thr_hi={thr_hi:.2f} thr_lo={thr_lo:.2f}")
