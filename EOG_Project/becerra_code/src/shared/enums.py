from enum import IntEnum


class Metric(IntEnum):
    MSE = 0
    DetectedSaccades = 1
    PeakVelocity = 2
    Duration = 3
    Latency = 4


class Status(IntEnum):
    Healthy = 0
    Sick = 1

    @classmethod
    def from_matlab(cls, value: str) -> 'Status':
        return {
            'S': Status.Healthy,
            'E': Status.Sick,
        }[value]
