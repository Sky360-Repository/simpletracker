# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from time import perf_counter

###################################################################################################################
# A class to provide some sort of execution time reporting so that we can determine application bottle necks etc. #
###################################################################################################################
class Stopwatch:

    def __init__(self, mask='Execution time: {s:0.4f} seconds', enable=False):
        self.mask = mask
        self.enable = enable

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = self.mask.format(s=self.time)
        if self.enable:
            print(self.readout)