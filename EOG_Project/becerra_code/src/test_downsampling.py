#!/bin/env python3.9
from os.path import join

from matplotlib import pyplot as plt, use as use_backend

from shared import read_matlab

if __name__ == '__main__':
    records = list(read_matlab('../data/RegScSimul20_1000_allNoisesDC_0.5_Sano.mat'))

    rec = records[0]
    rec_down = rec.downsampled(5)

    print(f'Original Step: {rec.h:.3f}')
    print(f'Original Sampling Frequency: {rec.sampling_frequency:.2f}')
    print(f'Original Samples Count: {len(rec.Y)}')

    print(f'Downsampled Step: {rec_down.h:.3f}')
    print(f'Downsampled Sampling Frequency: {rec_down.sampling_frequency:.2f}')
    print(f'Downsampled Samples Count: {len(rec_down.Y)}')

    use_backend('Qt5Agg')

    plt.subplot(2, 1, 1)
    plt.plot(rec.X, rec.Y)

    plt.subplot(2, 1, 2)
    plt.plot(rec_down.X, rec_down.Y)
    plt.show()
