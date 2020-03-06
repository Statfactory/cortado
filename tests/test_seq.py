from context import cortado
from cortado.seq import Seq
import pytest

def test_from_gen():
    gen = (i for i in range(5))
    s = Seq.from_gen(gen)
    v0, tail0 = Seq.try_read(s)
    assert v0 == 0
    v1, tail1 = Seq.try_read(tail0)
    assert v1 == 1

def test_tolist():
    gen = (i for i in range(5))
    s = Seq.from_gen(gen)
    lst = Seq.tolist(s)
    assert lst == [0, 1, 2, 3, 4]

def test_seq_zip():
    gen1 = Seq.from_gen((i for i in range(5)))
    gen2 = Seq.from_gen((i for i in range(1,6)))
    zipgen = Seq.zip(*(gen1, gen2))
    v = Seq.tolist(zipgen)
    assert v == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

