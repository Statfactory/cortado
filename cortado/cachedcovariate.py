import cortado.chunk as ch
from cortado.seq import Seq
import numpy as np
from cortado.abstractcovariate import AbstractCovariate
from cortado.funcslicer import FuncSlicer
from cortado.vectorslicer import VectorSlicer
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL

class CachedCovariate(AbstractCovariate):
    def __init__(self, covariate):
        self.covariate = covariate
        self.cache = None

    @property
    def name(self):
        return self.covariate.name

    def __len__(self):
        return len(self.covariate)

    @property
    def slicer(self):
        if self.cache is None:
            self.cache = np.empty(len(self), dtype=np.float32)
            slices = self.covariate.slicer(0, len(self.covariate), SLICELEN)

            def f(offset, slice):
                n = len(slice)
                self.cache[offset:(offset + n)] = slice
                return offset + n

            Seq.reduce(f, 0, slices)
        return VectorSlicer(self.cache)