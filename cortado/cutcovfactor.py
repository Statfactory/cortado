from cortado.abstractfactor import AbstractFactor
import numpy as np
from cortado.seq import Seq
from cortado.funcslicer import FuncSlicer
from cortado.consts import HEADLENGTH, SLICELEN, MISSINGLEVEL
from numba import jit
from numba.typed import Dict
from numba import types

class CutCovFactor(AbstractFactor):

    def __init__(self, covariate, cuts, rightclosed = False):
        self.covariate = covariate
        self.cuts = cuts
     
        assert cuts[0] == np.NINF and cuts[-1] == np.PINF
        levelcount = len(cuts) - 1
        if rightclosed:
            levels = [MISSINGLEVEL] + ["{z}{x},{y}]".format(x=str(cuts[i]), y=str(cuts[i + 1]), z="[" if i == 0 else "(") for i in range(levelcount)]
        else:
            levels = [MISSINGLEVEL] + ["[{x},{y}{z}".format(x=str(cuts[i]), y=str(cuts[i + 1]), z="]" if i == (levelcount - 1) else ")") for i in range(levelcount)]
        dtype = np.uint8 if levelcount <= 256 else np.uint16

        @jit(nopython=True, cache=True)
        def g_leftclosed(slice, buf, cuts, k):
            def f(x):
                if np.isnan(x):
                    return 0
                if x == np.PINF:
                    return k
                else:
                    i = np.searchsorted(cuts, x, side='right') 
                    return i

            for i in range(len(slice)):
                buf[i] = f(slice[i])
            if len(buf) == len(slice):
                return buf
            else:
                return buf[:len(slice)]

        @jit(nopython=True, cache=True)
        def g_rightclosed(slice, buf, cuts):
            def f(x):
                if np.isnan(x):
                    return 0
                if x == np.NINF:
                    return 1
                else:
                    i = np.searchsorted(cuts, x, side='left') 
                    return i

            for i in range(len(slice)):
                buf[i] = f(slice[i])
            if len(buf) == len(slice):
                return buf
            else:
                return buf[:len(slice)]

        def slicer(start, length, slicelen):
            length = min(len(self) - start, length)
            slicelen = min(length, slicelen)
            buf = np.empty(slicelen, dtype = dtype)
            if rightclosed:
                return Seq.map((lambda s: g_rightclosed(s, buf, cuts)), covariate.slicer(start, length, slicelen))
            else:
                return Seq.map((lambda s: g_leftclosed(s, buf, cuts, levelcount - 1)), covariate.slicer(start, length, slicelen))

        self._levels = levels
        self._slicer = FuncSlicer(slicer, dtype)

    @property
    def name(self):
        return self.covariate.name

    def __len__(self):
        return len(self.covariate)

    @property
    def isordinal(self):
        return True

    @property
    def levels(self):
        return self._levels

    @property
    def slicer(self):
        return self._slicer