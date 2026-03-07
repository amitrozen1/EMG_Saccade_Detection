from typing import Iterable
from os import listdir
from os.path import join

from scipy.io import loadmat

from .dataclasses import Record
from .enums import Status


def read_matlab(filename: str) -> Iterable[Record]:
    data = loadmat(filename)
    noise = float(filename.split('_')[-2])

    for record in range(int(data['cnRg'][0][0])):
        yield Record(
            filename=data['nmFichero1'][0],
            angle=int(data['aSc'][0][0]),
            noise=noise,
            h=data['tSm'][0][0],
            status=Status.from_matlab(data['Cat'][0]),
            saccades_count=int(data['cnSc'][0][0]),
            threshold=data['vThr'][0][0],
            X=data['xS'][0][record].flatten(),
            Y=data['yS'][0][record].flatten(),
            V0=data['vS'][0][record].flatten(),
            Y0=data['y0S'][0][record].flatten()
        )


def iterate_matlab_folder(path: str, verbose: bool = False) -> Iterable[Record]:
    for filename in listdir(path):
        if filename.endswith('.mat'):
            yield from read_matlab(join(path, filename))

            if verbose:
                print(f'{filename} completed')
