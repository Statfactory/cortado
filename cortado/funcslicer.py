from cortado.abstractslicer import AbstractSlicer

class FuncSlicer(AbstractSlicer):

    def __init__(self, f, dtype):
        self._dtype = dtype
        self.f = f

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, start, length, slicelen):
        return self.f(start, length,slicelen)
