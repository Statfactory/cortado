from cortado.abstractcovariate import AbstractCovariate
from cortado.vectorslicer import VectorSlicer
from cortado.fileslicer import FileSlicer
from cortado.seq import Seq
from cortado.parsefactorcovariate import ParseFactorCovariate
from cortado.funcslicer import FuncSlicer
import numpy as np

class Covariate(AbstractCovariate):
    def __init__(self, name, length, slicer):
        self._name = name
        self._length = length
        self._slicer = slicer

    @classmethod
    def from_array(cls, array, name = ""):
        vslicer = VectorSlicer(array)
        if vslicer.dtype == np.float32:
            return cls(name, len(array), vslicer)
        else:
            def g(slice, buf):
                if len(slice) == len(buf):
                    buf[:] = slice.astype(np.float32)
                    return buf
                else:
                    buf[:len(slice)] = slice.astype(np.float32)
                    return buf[:len(slice)]

            def f(start, length, slicelen):
                buf = np.empty(slicelen, dtype=np.float32)
                return Seq.map((lambda s: g(s, buf)), vslicer(start, length, slicelen))
            fslicer = FuncSlicer(f, np.float32)
            return cls(name, len(array), fslicer)

    @classmethod
    def from_file(cls, name, length, path, dtype):
        slicer = FileSlicer(path, dtype)
        return cls(name, length, slicer)

    @classmethod
    def from_factor(cls, factor, func):
        return ParseFactorCovariate(factor.name, factor, func)

    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._length

    @property
    def slicer(self):
        return self._slicer

    

    