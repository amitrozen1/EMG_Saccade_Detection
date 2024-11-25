from EDF.PlotHeatmap import get_heatmap
import numpy as np

argaman_photo = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\Paper\ppts\Argaman.jpg"
w_path = r"C:\Liron\DataEmg\Try\26122022_1015\DWT_hila_db15\A\w.npy"
w = np.load(w_path)

get_heatmap(argaman_photo, 16, w)