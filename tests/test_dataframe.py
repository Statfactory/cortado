from context import cortado
from cortado.seq import Seq
from cortado.dataframe import DataFrame
from cortado.factor import factor
import numpy as np
from datetime import datetime
from numba import jit
from numba.targets.registry import CPUDispatcher
from inspect import Signature
from cortado.logistic import logit_h
from cortado.covariate import Covariate
import tempfile
from pathlib import Path
import pandas as pd

def test_dataframe_convert():
    tmpdir = tempfile.TemporaryDirectory()
    pathdata = Path() / "data"
    train_csvpath = pathdata / "airlinetrain1m.csv"
    df = pd.read_csv(train_csvpath.resolve())
    DataFrame.convert(df, tmpdir.name)
    train_df = DataFrame.from_path(tmpdir.name)
    assert len(train_df.factors) == 7
    assert len(train_df.covariates) == 2

def test_dataframe_from_pandas():
    pathdata = Path() / "data"
    train_csvpath = pathdata / "airlinetrain1m.csv"
    df = pd.read_csv(train_csvpath.resolve())
    train_df = DataFrame.from_pandas(df)
    assert len(train_df.factors) == 7
    assert len(train_df.covariates) == 2

def test_dataframe_from_pandas_chunked():
    pathdata = Path() / "data"
    train_csvpath = pathdata / "airlinetrain1m.csv"
    df = pd.read_csv(train_csvpath.resolve(), chunksize=250)
    train_df = DataFrame.from_pandas(df)
    assert len(train_df.factors) == 7
    assert len(train_df.covariates) == 2
    