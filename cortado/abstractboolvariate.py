from abc import ABC, abstractmethod
from cortado.seq import Seq
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL

class AbstractBoolVariate(ABC):

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

    def __repr__(self):
        slices = self.slicer(0, min(HEADLENGTH, len(self)), HEADLENGTH)
        def f(acc, slice):
            return acc + ' '.join([str(v) for v in slice]) + " "
        datahead = Seq.reduce(f, "", slices)
        return "BoolVariate {var} with {len} obs: {head}".format(var= self.name, len= len(self), head= datahead)

    def __str__(self):
        return self.__repr__()

