from cortado.chunk import next_data_chunk
from cortado.seq import Seq
import numpy as np
from cortado.abstractslicer import AbstractSlicer

class FileSlicer(AbstractSlicer):
    def __init__(self, path, dtype):
        self.path = path
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, start, length, slicelen):
        assert start >= 0
        slicelen = min(length, slicelen) 
        iostream = open(self.path, "rb")
        buf = np.empty(slicelen, dtype = self.dtype) 
        itemsize = buf.dtype.itemsize
        iostream.seek(itemsize * start)
        state = (buf, iostream, start, length, slicelen)
        return Seq.from_next(state, next_data_chunk) 
