from cortado.abstractcovariate import AbstractCovariate
import numpy as np
from cortado.seq import Seq
from cortado.funcslicer import FuncSlicer
from numba import jit
from numba.typed import Dict
from numba import types


class ParseFactorCovariate(AbstractCovariate):
    def __init__(self, name, basefactor, func):
        self._name = name
        self._length = len(basefactor)
        self.basefactor = basefactor
        self.func = func

        levels = basefactor.levels
        parsed = np.empty(len(levels))
        for i, level in enumerate(levels):
            parsed[i] = func(level)

        @jit(nopython=True)
        def g(slice, buf, parsed):
            for i in range(len(slice)):
                buf[i] = parsed[slice[i]]    
            if len(buf) == len(slice):
                return buf
            else:
                return buf[:len(slice)] 


        def slice(start, length, slicelen):
            buf = np.empty(slicelen, dtype = np.float32)
            return Seq.map((lambda slice: g(slice, buf, parsed)), basefactor.slicer(start, length, slicelen))

        self._slicer = FuncSlicer(slice, np.float32)


    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._length

    @property
    def slicer(self):
        return self._slicer
