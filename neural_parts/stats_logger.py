"""Stats logger provides a method for logging training stats."""

import sys


class AverageAggregator(object):
    def __init__(self):
        self._value = 0
        self._count = 0

    @property
    def value(self):
        return self._value / self._count

    @value.setter
    def value(self, val):
        self._value += val
        self._count += 1


class StatsLogger(object):
    __INSTANCE = None
    def __init__(self):
        if StatsLogger.__INSTANCE is not None:
            raise RuntimeError("StatsLogger should not be directly created")

        self._values = dict()
        self._loss = AverageAggregator()
        self._output_files = [sys.stdout]

        self.debug = dict()

    def add_output_file(self, f):
        self._output_files.append(f)

    def __getitem__(self, key):
        if key not in self._values:
            self._values[key] = AverageAggregator()
        return self._values[key]

    def clear(self):
        self._values.clear()
        self._loss = AverageAggregator()
        for f in self._output_files:
            if f.isatty():
                print(file=f, flush=True)

    def print_progress(self, epoch, batch, loss, precision="{:.5f}"):
        self._loss.value = loss
        fmt = "epoch: {} - batch: {} - loss: " + precision
        msg = fmt.format(epoch, batch, self._loss.value)
        for k,  v in self._values.items():
            msg += " - " + k + ": " + precision.format(v.value)
        for f in self._output_files:
            if f.isatty():
                print(msg + "\b"*len(msg), end="", flush=True, file=f)
            else:
                print(msg, flush=True, file=f)

    @classmethod
    def instance(cls):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = cls()
        return cls.__INSTANCE
