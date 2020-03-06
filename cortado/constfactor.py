import cortado.chunk as ch
from cortado.seq import Seq
import numpy as np
from cortado.abstractfactor import AbstractFactor
from cortado.funcslicer import FuncSlicer
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL

class ConstFactor(AbstractFactor):
    def __init__(self, length):
        self._name = "Intercept"
        self._length = length
        self._levels = [MISSINGLEVEL, "Intercept"]

        def slicer(start, length, slicelen):
            length = min(self._length - start, length)
            slicelen = min(length, slicelen)
            buf = np.ones(slicelen, dtype = np.uint8)
            return Seq.map((lambda x: buf[x[0]:x[1]]), Seq.from_next((start, length, slicelen), ch.next_slice_indices))
        self._slicer = FuncSlicer(slicer, np.uint8)
      
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