from abc import ABC, abstractmethod

class AbstractSlicer(ABC):

    @abstractmethod
    def __call__(self, start, length, slicelen):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

