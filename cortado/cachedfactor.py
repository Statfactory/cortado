import cortado.chunk as ch
from cortado.seq import Seq
import numpy as np
from cortado.abstractfactor import AbstractFactor
from cortado.funcslicer import FuncSlicer
from cortado.vectorslicer import VectorSlicer
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL

class CachedFactor(AbstractFactor):
    def __init__(self, factor):
        self.factor = factor
        self.cache = None

    @property
    def name(self):
        return self.factor.name

    def __len__(self):
        return len(self.factor)

    @property
    def levels(self):
        return self.factor.levels

    @property
    def isordinal(self):
        return self.factor.isordinal

    @property
    def slicer(self):
        if self.cache is None:
            self.cache = np.empty(len(self), dtype=self.factor.slicer.dtype)
            slices = self.factor.slicer(0, len(self.factor), SLICELEN)

            def f(offset, slice):
                n = len(slice)
                self.cache[offset:(offset + n)] = slice
                return offset + n

            Seq.reduce(f, 0, slices)
        return VectorSlicer(self.cache)