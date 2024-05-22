import General.tam.tam_dtw as tam_dtw
import General.tam.tam as tam
import numpy as np
from scipy.stats import zscore

#### usage ####
# d, c, ac, p = dtw.dtw(x, y)
# dist = tam.tam(p)
# print("Distance %f", dist)

def tam_corr(x, y, tMax):
    x = zscore(x)
    y = zscore(y)
    d, c, ac, p = tam_dtw.dtw(x, y, factor = tMax, window = "sakoe-chiba")
    dist = tam.tam(p, report="distance")
    return dist

if __name__ == "__main__":
    x = np.sin(np.linspace(0, 10, 100))
    y = np.cos(np.linspace(0, 10, 100))

    d, c, ac, p = tam_dtw.dtw(x, y)
    dist = tam.tam(p, report="distance")
    print("Distance %f" % dist)