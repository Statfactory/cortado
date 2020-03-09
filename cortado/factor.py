from cortado import chunk as ch
from cortado.seq import Seq
import numpy as np
from numba import jit
from datetime import datetime
from cortado.abstractfactor import AbstractFactor
from cortado.abstractcovariate import AbstractCovariate
from cortado.funcslicer import FuncSlicer
from cortado.cutcovfactor import CutCovFactor
from cortado.vectorslicer import VectorSlicer
from cortado.fileslicer import FileSlicer

class Factor(AbstractFactor):
    def __init__(self, name, length, levels, slicer):
        self._name = name
        self._length = length
        self._levels = levels
        self._slicer = slicer

    @classmethod
    def from_array(cls, name, levels, array):
        slicer = VectorSlicer(array)
        return cls(name, len(array), levels, slicer)

    @classmethod
    def from_file(cls, name, length, levels, path, dtype):
        slicer = FileSlicer(path, dtype)
        return cls(name, length, levels, slicer)

    @classmethod
    def from_covariate(cls, covariate, cuts = None, rightclosed = False):
        assert isinstance(covariate, AbstractCovariate)
        if cuts is not None:
            cuts = np.array(cuts, dtype=np.float32)
            if cuts[0] != np.NINF:
                cuts = np.insert(cuts, 0, np.NINF)
            if cuts[len(cuts)] != np.PINF:
                cuts = np.append(cuts, np.PINF)
            return CutCovFactor(covariate, cuts=cuts, rightclosed=rightclosed)
        else:
            unique_ = covariate.unique()
            n = len(unique_)
            cuts = np.empty(n + 1, dtype=np.float32)
            cuts[0] = np.NINF
            for i in range(n):
                if i == n - 1:
                    cuts[i + 1] = np.PINF
                else:
                    cuts[i + 1] = unique_[i] + 0.5 * (unique_[i + 1] - unique_[i])
            return CutCovFactor(covariate, cuts=cuts, rightclosed=rightclosed)

    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._length

    @property
    def levels(self):
        return self._levels

    @property
    def slicer(self):
        return self._slicer

    