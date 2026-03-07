from dataclasses import dataclass
from typing import Iterable

from numpy import argmax, array
from scipy.signal import decimate

from .differentiation import METHODS, differentiate
from .enums import Metric, Status
from .math import mse


@dataclass
class DFLine:
    status: Status
    noise: float
    angle: int
    method: str
    metric: Metric
    value: float
    filename: str

    @classmethod
    def columns(cls, metric: Metric) -> list[str]:
        return [
            'Filename',
            'Status',
            'Noise',
            'Angle',
            'Method',
            metric.name,
        ]

    @property
    def df_row(self) -> list:
        return [
            self.filename,
            self.status.value,
            self.noise,
            self.angle,
            self.method,
            self.value,
        ]


@dataclass
class Record:
    filename: str
    angle: int
    noise: float
    h: float
    status: Status
    saccades_count: int
    threshold: float
    X: array
    Y: array
    V0: array
    Y0: array

    def __str__(self):
        return f'Record for file: {self.filename}'

    def downsampled(self, factor: int) -> 'Record':
        return Record(
            filename=self.filename,
            angle=self.angle,
            noise=self.noise,
            h=self.h * factor,
            status=self.status,
            saccades_count=self.saccades_count,
            threshold=self.threshold,
            X=decimate(self.X, factor),
            Y=decimate(self.Y, factor),
            V0=decimate(self.V0, factor),
            Y0=decimate(self.Y0, factor)
        )

    @property
    def sampling_frequency(self) -> float:
        return 1.0 / self.h

    def velocities(self, method: str) -> array:
        return differentiate(self.Y, self.h, method)

    def saccades(self, velocities: array, min_duration: float = None) -> Iterable[tuple[int, int]]:
        velocities = abs(velocities)
        last = len(velocities) - 1
        index = 0

        if min_duration is None:
            min_duration = {
                20: 0.09,
                30: 0.115,
                60: 0.175,
            }[self.angle]

        while index < last:
            if velocities[index] > self.threshold:
                onset = index
                while onset > 0 and velocities[onset - 1] >= 20:
                    onset -= 1
                offset = index
                while offset < last and velocities[offset + 1] >= 20:
                    offset += 1

                if (offset - onset) * self.h >= min_duration:
                    yield onset, offset

                index = offset + 1
            else:
                index += 1

    def mse_lines(self) -> Iterable[DFLine]:
        for method in METHODS:
            approx = self.velocities(method)

            yield DFLine(
                status=self.status,
                noise=self.noise,
                angle=self.angle,
                metric=Metric.MSE,
                value=mse(self.V0, approx),
                filename=self.filename,
                method=method
            )

    def detected_saccades_lines(self) -> Iterable[DFLine]:
        for method in METHODS:
            if method in {'cd3', 'cd5', 'cd7', 'cd9'}:
                continue
            approx = self.velocities(method)
            saccades = len(list(self.saccades(approx)))

            yield DFLine(
                status=self.status,
                noise=self.noise,
                angle=self.angle,
                metric=Metric.DetectedSaccades,
                value=saccades - self.saccades_count,
                filename=self.filename,
                method=method
            )

    def peak_velocity_lines(self) -> Iterable[DFLine]:
        for method in METHODS:
            if method in {'cd3', 'cd5', 'cd7', 'cd9'}:
                continue

            V0 = abs(self.V0)
            approx = abs(self.velocities(method))

            for onset, offset in self.saccades(V0):
                V0_part = V0[onset:offset]
                max_idx = argmax(V0_part)

                yield DFLine(
                    status=self.status,
                    noise=self.noise,
                    angle=self.angle,
                    metric=Metric.PeakVelocity,
                    value=approx[onset:offset][max_idx] - V0_part[max_idx],
                    filename=self.filename,
                    method=method
                )

    def time_lines(self) -> Iterable[DFLine]:
        saccades = list(self.saccades(self.V0))

        for method in METHODS:
            if method in {'cd3', 'cd5', 'cd7', 'cd9'}:
                continue

            pairing = {
                sacc: []
                for sacc in saccades
            }
            approx = abs(self.velocities(method))

            for onset, offset in self.saccades(approx):
                for (r_onset, r_offset), paired in pairing.items():
                    if (r_onset <= onset <= r_offset) or (r_onset <= offset <= r_offset) or (onset <= r_onset and offset >= r_offset):
                        paired.append((onset, offset))
                        break

            for (r_onset, r_offset), candidates in pairing.items():
                if len(candidates) == 1:
                    a_onset, a_offset = candidates[0]
                    r_duration = (r_offset - r_onset) * self.h
                    a_duration = (a_offset - a_onset) * self.h

                    yield DFLine(
                        status=self.status,
                        noise=self.noise,
                        angle=self.angle,
                        metric=Metric.Duration,
                        value=a_duration - r_duration,
                        filename=self.filename,
                        method=method
                    )

                    yield DFLine(
                        status=self.status,
                        noise=self.noise,
                        angle=self.angle,
                        metric=Metric.Latency,
                        value=(a_onset - r_onset) * self.h,
                        filename=self.filename,
                        method=method
                    )
