from abc import ABC, abstractmethod
from cortado.seq import Seq
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL
import numpy as np
from numba import jit

class AbstractFactor(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def isordinal(self):
        return False

    @property
    @abstractmethod
    def levels(self):
        pass

    @property
    @abstractmethod
    def slicer(self):
        pass

    def cached(self):
        from cortado.cachedfactor import CachedFactor
        if isinstance(self, CachedFactor):
            return self
        else:
            return CachedFactor(self)

    def get_freq(self):
        slices = self.slicer(0, len(self), SLICELEN)
        counts0 = np.zeros(len(self.levels), dtype=np.int32)

        @jit(nopython=True, cache=True)
        def f(acc, slice):
            for v in slice:
                acc[v] = acc[v] + 1
            return acc
        return Seq.reduce(f, counts0, slices)

    def __repr__(self):
        k = min(HEADLENGTH, len(self))
        slices = self.slicer(0, k, HEADLENGTH)
        levels = self.levels
        def f(acc, slice):
            return acc + ' '.join([levels[i] for i in slice]) + " "
        datahead = Seq.reduce(f, "", slices)
        s = "" if k == len(self) else "..."
        return "Factor {f} with {len} obs and {n} levels: {head}{s}".format(f= self.name, len= len(self), head= datahead, n = len(levels) - 1, s=s)

    def __str__(self):
        return self.__repr__()

