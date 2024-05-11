import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector as SpanSelector


class SignalSelector:
    def __init__(self, signal):
        self.signal = signal
        self.fig, self.ax = plt.subplots()
        self.ax.plot(signal)
        self.ax.set_title('Click and drag to select part of the signal')
        self.span = None
        self.selected_region = []

    def onselect(self, xmin, xmax):
        indmin, indmax = int(xmin), int(xmax)
        # Ensure indmax does not exceed signal length
        indmax = min(len(self.signal) - 1, indmax)  

        # Extract and store the selected portion of the signal
        self.selected_region.extend(self.signal[indmin:indmax])

        # Visual feedback on selection
        self.ax.axvspan(indmin, indmax, color='yellow', alpha=0.3)  # Highlight selected region
        self.fig.canvas.draw_idle()  # Redraw the figure to update the highlighted selection
        plt.close(self.fig)
        
    def select_part_of_signal(self):
        # Create a span selector for interactive selection
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                 props=dict(alpha=0.5, facecolor='green'), interactive=True, drag_from_anywhere=True)
        plt.show()

    def get_selected_region(self):
        return self.selected_region

def mark_blink_out_of_signal(signal):
    """
    Interactive function to select a portion of a plotted signal
    The user marks the blink in the signal
    :param signal: the blinking component (signal)
    :return: the blink (chosen by the user) part of the signal
    """
    signal_selector = SignalSelector(signal)
    signal_selector.select_part_of_signal()
    return signal_selector.get_selected_region()

def main():
    # Generate a synthetic signal with a blink
    fs = 1000
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    blink = np.zeros_like(signal)
    blink[500:600] = 1
    signal += blink

    # Mark the blink in the signal
    blink_part = mark_blink_out_of_signal(signal)
    plt.plot(blink_part)
    plt.show()