from time import perf_counter
from contextlib import contextmanager

class Stopwatch:

    def __init__(self, mask='Execution time: {s:0.4f} seconds', quiet=False):
        self.mask = mask
        self.quiet = quiet

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = self.mask.format(s=self.time)
        if not self.quiet:
            print(self.readout)