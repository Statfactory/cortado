from cortado.seq import Seq
import numpy as np
from cortado.chunk import next_slice_indices
from cortado.abstractslicer import AbstractSlicer

class VectorSlicer(AbstractSlicer):
    def __init__(self, vector):
        self.vector = vector
        self._dtype = vector.dtype

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, start, length, slicelen):
        assert start >= 0 and start < len(self.vector)
        length = min(length, len(self.vector) - start)
        slicelen = min(length, slicelen) 
        state = start, length, slicelen
        
        def f(from_to):
            return self.vector[from_to[0] : from_to[1]]

        return Seq.map(f, Seq.from_next(state, next_slice_indices))