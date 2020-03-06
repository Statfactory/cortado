from cortado.abstractboolvariate import AbstractBoolVariate
from cortado.vectorslicer import VectorSlicer

class BoolVariate(AbstractBoolVariate):
    def __init__(self, name, length, slicer):
        self._name = name
        self._length = length
        self._slicer = slicer

    @classmethod
    def from_array(cls, name, array):
        slicer = VectorSlicer(array)
        return cls(name, len(array), slicer)

    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._length

    @property
    def slicer(self):
        return self._slicer