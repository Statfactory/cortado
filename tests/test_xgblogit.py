from context import cortado
from cortado.seq import Seq
from cortado.dataframe import DataFrame
from cortado.factor import factor
from cortado.covariate import Covariate
from cortado.parsefactorcovariate import ParseFactorCovariate
from cortado.boolvariate import BoolVariate
from cortado.cachedfactor import CachedFactor
from cortado.logistic import xgblogit
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, hstack
import numpy as np
import xgboost as xgb
import pandas as pd
from pathlib import Path
from pandas.api.types import union_categoricals

pathdata = Path() / "data"
train_csvpath = pathdata / "airlinetrain1m.csv"
df = pd.read_csv(train_csvpath.resolve())
traintest_df = DataFrame.from_pandas(df)

label = df["dep_delayed_15min"].map({"N": 0, "Y": 1})
deptime = df["DepTime"].astype(pd.SparseDtype("float32", 0.0))
distance = df["Distance"].astype(pd.SparseDtype("float32", 0.0))

month = pd.get_dummies(df["Month"], prefix="month", sparse=True, dtype=np.float32)
dayofmonth = pd.get_dummies(df["DayofMonth"], prefix="dayofmonth", sparse=True, dtype=np.float32)
dayofweek = pd.get_dummies(df["DayOfWeek"], prefix="dayofweek", sparse=True, dtype=np.float32)

uniqcarrier = pd.get_dummies(df["UniqueCarrier"], prefix="carrier", sparse=True, dtype=np.float32)
origin = pd.get_dummies(df["Origin"], prefix="origin", sparse=True, dtype=np.float32)
dest = pd.get_dummies(df["Dest"], prefix="dest", sparse=True, dtype=np.float32)

data = pd.concat([month, dayofmonth, dayofweek, uniqcarrier, origin, dest, deptime, distance], axis=1)
spdata = coo_matrix(data.sparse.to_coo()).tocsr()

# featmappath = pathdata / "featmap.txt"
# with open(featmappath.resolve(), "w") as f: 
#     lines = ["{fid} {fname} {ftype}\n".format(fid=i, fname=col, ftype="int" if col.startswith(("deptime", "distance")) else "i") for (i, col) in enumerate(data.columns)]
#     f.writelines(lines)

eta = 0.1
nrounds = 100
max_depth = 8

#rawdumppath = pathdata / "dump.raw.txt"

start = datetime.now()
model = xgb.XGBClassifier(max_depth=max_depth, nthread=1, learning_rate=eta, tree_method="exact", n_estimators=nrounds, verbosity=3)
model.fit(spdata, label, verbose=True, eval_set=[(spdata, label)], eval_metric="auc")
p = model.predict_proba(spdata)
end = datetime.now()
print("xgboost elapsed: {e}".format(e=(end - start)))

distance = traintest_df["Distance"]
deptime = traintest_df["DepTime"]
dep_delayed_15min = traintest_df["dep_delayed_15min"]
label = Covariate.from_factor(dep_delayed_15min, lambda level: 1.0 if level == "Y" else 0.0)
deptime = CachedFactor(factor(traintest_df["DepTime"]))
distance = CachedFactor(factor(traintest_df["Distance"]))

factors = traintest_df.factors + [deptime, distance]
factors.remove(dep_delayed_15min)

trainsel = np.array(np.arange(1000000) < 1000000)

def test_1():
    start = datetime.now()
    trees, pred = xgblogit(label, factors, trainsel,  eta = eta, lambda_ = 1.0,
                           gamma = 0.0, minh = 1.0, nrounds = nrounds, maxdepth = max_depth)
    end = datetime.now()
    print("cortado elapsed: {e}".format(e=(end - start)))
    
    y = label.to_array()
    auc = roc_auc_score(y, pred)
    print("cortado auc")
    print(auc)
    print("max pred diff")
    diff = np.max(np.abs(p[:, 1] - pred))
    print(diff)
    assert diff < 1e-6


