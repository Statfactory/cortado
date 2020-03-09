from cortado.abstractcovariate import AbstractCovariate
import numpy as np
from cortado.seq import Seq
from cortado.funcslicer import FuncSlicer
import types
from numba import jit

@jit(nopython=True)
def g_parse(slice, buf, parsed):
    for i in range(len(slice)):
        buf[i] = parsed[slice[i]]    
    if len(buf) == len(slice):
        return buf
    else:
        return buf[:len(slice)] 

class ParseFactorCovariate(AbstractCovariate):
    def __init__(self, name, basefactor, func):
        self._name = name
        self._length = len(basefactor)
        self.basefactor = basefactor
        if isinstance(func, types.FunctionType):
            self.func = func
        elif isinstance(func, dict):
            self.func = lambda level: np.float32(func[level]) if level in func else np.float32(np.nan)
        else:
            raise NotImplementedError()

        levels = basefactor.levels   
        parsed = np.empty(len(levels), dtype=np.float32)
        for i, level in enumerate(levels):
            parsed[i] = np.float32(self.func(level))

        def slice(start, length, slicelen):
            buf = np.empty(slicelen, dtype = np.float32)
            return Seq.map((lambda slice: g_parse(slice, buf, parsed)), basefactor.slicer(start, length, slicelen))

        self._slicer = FuncSlicer(slice, np.float32)


    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._length

    @property
    def slicer(self):
        return self._slicer
