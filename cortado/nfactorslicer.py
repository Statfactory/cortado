import numpy as np
from cortado.seq import Seq

class NFactorSlicer():
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, start, length, slicelen):
        dtypes = [f.slicer.dtype for f in self.factors]
        alluint8 = all([d == np.uint8 for d in dtypes])
        dtype = np.uint8 if alluint8 else np.uint16
        k = len(self.factors)
        buf = np.empty((k, slicelen), dtype=dtype)

        def f(slices):
            n = len(slices[0])
            for i in range(k):
                if dtypes[i] == dtype:
                    buf[i, :n] = slices[i]
                else:
                    buf[i, :n] = slices[i].astype(dtype)
            if n == slicelen:
                return buf
            else:
                return buf[:, :n]

        return Seq.map(f, Seq.zip(*[f.slicer(start, length, slicelen) for f in self.factors]))