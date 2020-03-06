import json
import os
from cortado.consts import MISSINGLEVEL 
from cortado.covariate import Covariate
from cortado.factor import Factor
from cortado.fileslicer import FileSlicer
from cortado.vectorslicer import VectorSlicer
from cortado.chunk import read_binary
from cortado.abstractfactor import AbstractFactor
from cortado.abstractcovariate import AbstractCovariate
from cortado.consts import MISSINGLEVEL
from numba import jit
import pandas as pd
from pandas.api.types import union_categoricals
from pandas.io.parsers import TextFileReader
import numpy as np
from pathlib import Path
import random
import string

@jit(nopython=True, cache=True)
def incr_(x, y):
    for i in range(len(x)):
        y[i] = x[i] + 1

class ColImporter():
    def __init__(self, colname, iscategorical, dirpath):
        self.colname = colname
        self.iscategorical = iscategorical
        self.iscat_uint8 = True
        self.levels = []
        self.length = 0
        self.dirpath = dirpath
        self.rndname = "".join(random.choices(string.ascii_letters, k=10)) + ".dat"
        self.filepath = (Path(dirpath) / self.rndname).resolve()

    def importnumeric(self, data):
        with open(self.filepath, "ab") as f:
            mem_view = memoryview(data)
            f.write(mem_view.tobytes())
        self.length += len(data) 
        
    def importcategorical(self, data, levels):
        if len(self.levels) <= 256 and len(levels) > 256 and self.length > 0:
            rndname = "".join(random.choices(string.ascii_letters, k=10)) + ".dat"
            newfilepath = (Path(self.dirpath) / rndname).resolve()
            with open(self.filepath, "rb") as fin:
                with open(newfilepath, "wb") as fout:
                    buf = np.empty(self.length, dtype=np.uint8)
                    mem_view = memoryview(buf)
                    n = fin.readinto(mem_view)
                    assert n == self.length
                    v = buf.astype(np.uint16)
                    mem_view = memoryview(v)
                    fout.write(mem_view.tobytes())
                    self.filepath = newfilepath
                    self.rndname = rndname
                   
        self.iscat_uint8 = len(levels) <= 256

        with open(self.filepath, "ab") as f:
            if self.iscat_uint8:
                buf = np.empty(len(data), np.uint8)
                incr_(data, buf)
                mem_view = memoryview(buf)
                f.write(mem_view.tobytes())
            else:
                buf = np.empty(len(data), np.uint16)
                incr_(data, buf)
                mem_view = memoryview(buf)
                f.write(mem_view.tobytes())

        self.length += len(data)
        self.levels = levels

    def asdict(self):
        dtype = "float32" if not self.iscategorical else ("uint8" if self.iscat_uint8 else "uint16")
        return {"name": self.colname, "length": self.length, "filename": self.rndname, "datatype": dtype, "levels": self.levels}

class DataFrame():
    def __init__(self, columns):
        self.cols = {col.name : col for col in columns}

    @classmethod
    def from_path(cls, path, preload = True):
        res = []
        path = os.path.abspath(path)
        header_path = path if os.path.isfile(path) else os.path.join(path, "header.json")
        with open(header_path) as header_file:
            header = json.load(header_file)
            data_cols = header["datacolumns"]
            for data_col in data_cols:
                data_type = data_col["datatype"]
                length = data_col["length"]
                name = data_col["name"]
                path = os.path.join(os.path.dirname(header_path), data_col["filename"])

                if data_type == "float32":
                    if preload:
                        buf = read_binary(path, length, np.float32)
                        res.append(Covariate.from_array(buf, name))           
                    else:
                        res.append(Covariate.from_file(name, length, path, np.float32))
                        

                if data_type == "uint8":
                    levels = data_col["levels"]
                    if preload:
                        buf = read_binary(path, length, np.uint8)
                        res.append(Factor.from_array(name, levels, buf))
                    else:
                        res.append(Factor.from_file(name, length, levels, path, np.uint8))

                if data_type == "uint16":
                    levels = data_col["levels"]
                    if preload:
                        buf = read_binary(path, length, np.uint16)
                        res.append(Factor.from_array(name, levels, buf))
                    else:
                        res.append(Factor.from_file(name, length, levels, path, np.uint16))
        return cls(res)

    def __getitem__(self, name):
        return self.cols[name]

    def info(self):
        for _, v in self.cols.items():
            print(v)

    @property
    def factors(self):
        return [v for _, v in self.cols.items() if isinstance(v, AbstractFactor)]

    @property
    def covariates(self):
        return [v for _, v in self.cols.items() if isinstance(v, AbstractCovariate)]

    @classmethod
    def from_pandas(cls, df):

        df = df if isinstance(df, TextFileReader) else [df]
        res = []
        for chunk in df:
            for col in chunk.columns:
                dtype = chunk[col].dtype
                if dtype == "object" or dtype == "str" or dtype == "bool":
                    chunk[col] = chunk[col].astype("category")

            if len(res) == 0:
                res = chunk
            else:
                for col in chunk.columns:
                    if chunk[col].dtype.name == "category":
                        uc = union_categoricals([res[col], chunk[col]])
                        res[col] = pd.Categorical(res[col], categories=uc.categories)
                        chunk[col] = pd.Categorical(chunk[col], categories=uc.categories)
                res = pd.concat([res, chunk])

        datacols = []
        for col in res.columns:
            dtype = res[col].dtype
            if dtype == np.float32 or dtype == np.float64 or dtype == np.int8 or dtype == np.int16 or dtype == np.int32 or dtype == np.int64 or dtype == np.uint8 or dtype == np.uint16 or dtype == np.uint32 or dtype == np.uint64:
                datacols.append(Covariate.from_array(res[col].values, col))
            elif dtype.name == "category":
                factor = res[col]
                codes = factor.cat.codes.values
                levels = [MISSINGLEVEL] +  [str(v) for v in factor.cat.categories.values]
                data = np.empty(len(codes), dtype=np.uint8) if len(levels) <= 256 else np.empty(len(codes), dtype=np.uint16)
                incr_(codes, data)
                datacols.append(Factor.from_array(col, levels, data))
            else:
                pass
        return cls(datacols)

    @classmethod
    def convert(cls, df, outdir):
        path = Path(outdir)
        assert path.is_dir()
        if not path.exists():
            path.mkdir()

        df = df if isinstance(df, TextFileReader) else [df]
        colimporters = {}
        v = []
        for chunk in df:
            for col in chunk.columns:
                dtype = chunk[col].dtype
                if dtype == "object" or dtype == "str" or dtype == "bool":
                    chunk[col] = chunk[col].astype("category")
                elif dtype == np.float32 or dtype == np.float64 or dtype == np.int8 or dtype == np.int16 or dtype == np.int32 or dtype == np.int64 or dtype == np.uint8 or dtype == np.uint16 or dtype == np.uint32 or dtype == np.uint64:
                    chunk[col] = chunk[col].astype(np.float32)
                else:
                    pass
               
            if len(v) == 0:
                v = chunk
            else:
                for col in chunk.columns:
                    if chunk[col].dtype.name == "category":
                        uc = union_categoricals([v[col], chunk[col]])
                        v[col] = pd.Categorical(v[col], categories=uc.categories)
                        chunk[col] = pd.Categorical(chunk[col], categories=uc.categories)

            for col in chunk.columns:
                if not col in colimporters:
                    iscat = chunk[col].dtype.name == "category"
                    colimporters[col] = ColImporter(col, iscat, path)
                colimporter = colimporters[col]
                if colimporter.iscategorical:
                    factor = chunk[col]
                    levels = [MISSINGLEVEL] + [str(v) for v in factor.cat.categories.values]
                    colimporter.importcategorical(chunk[col].cat.codes.values, levels)
                else:
                    colimporter.importnumeric(chunk[col].values)

        d = {"datacolumns": [c.asdict() for _, c in colimporters.items()]}
        jsonstr = json.dumps(d)
        with open((path / "header.json").resolve(), "w") as f:
            f.write(jsonstr)

        








