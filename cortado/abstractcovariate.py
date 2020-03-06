from abc import ABC, abstractmethod
from cortado.seq import Seq
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL
import numpy as np
from numba import jit
from numba.typed import Dict
from numba import types


class AbstractCovariate(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def slicer(self):
        pass

    def to_array(self):
        slices = self.slicer(0, len(self), len(self))
        res, _ = Seq.try_read(slices)
        return res

    def unique(self):
        slices = self.slicer(0, len(self), SLICELEN)
        # v, tail = Seq.try_read(slices)
        # set0 = set(v)

        # @jit(nopython=True, cache=True)
        # def f(acc, slice):
        #     for v in slice:
        #         if not np.isnan(v):
        #             acc.add(v)
        #     return acc
        # res = Seq.reduce(f, set0, tail)
        dt = types.float32 if self.slicer.dtype == np.float32 else types.float64
        set0 = Dict.empty(key_type=dt, value_type=dt)
        
        @jit(nopython=True, cache=True)
        def f(acc, slice):
            for v in slice:
                if not np.isnan(v) and not v in acc:
                    acc[v] = v
            return acc
        res = Seq.reduce(f, set0, slices)

        arr = np.array(list(res.keys()), dtype=self.slicer.dtype)
        arr.sort()
        return arr

    def __repr__(self):
        slices = self.slicer(0, min(HEADLENGTH, len(self)), HEADLENGTH)
        def f(acc, slice):
            return acc + ' '.join(["." if np.isnan(v) else str(v) for v in slice]) + " "
        datahead = Seq.reduce(f, "", slices)
        return "Covariate {cov} with {len} obs: {head}...".format(cov= self.name, len= len(self), head= datahead)

    def __str__(self):
        return self.__repr__()
