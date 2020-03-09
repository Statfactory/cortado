from context import cortado
from cortado import Seq, DataFrame, Covariate, Factor, ParseFactorCovariate, BoolVariate, xgblogit
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
csvpath = pathdata / "airlinetrain1m.csv"
df = pd.read_csv(csvpath.resolve())

label = df["dep_delayed_15min"].map({"N": 0, "Y": 1})

covariates = ["DepTime", "Distance"]
factors = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

sp_covariates = list(map(lambda col: df[col].astype(pd.SparseDtype("float32", 0.0)), covariates))
sp_factors = list(map(lambda col: pd.get_dummies(df[col], prefix=col, sparse=True, dtype=np.float32), factors))

data = pd.concat(sp_factors + sp_covariates, axis=1)
spdata = coo_matrix(data.sparse.to_coo()).tocsr()

# featmappath = pathdata / "featmap.txt"
# with open(featmappath.resolve(), "w") as f: 
#     lines = ["{fid} {fname} {ftype}\n".format(fid=i, fname=col, ftype="int" if col.startswith(("deptime", "distance")) else "i") for (i, col) in enumerate(data.columns)]
#     f.writelines(lines)

eta = 0.1
nrounds = 100
max_depth = 8

start = datetime.now()
model = xgb.XGBClassifier(max_depth=max_depth, nthread=1, learning_rate=eta, tree_method="exact", n_estimators=nrounds, verbosity=3)
model.fit(spdata, label, verbose=True, eval_set=[(spdata, label)], eval_metric="auc")
p = model.predict_proba(spdata)
end = datetime.now()
print("xgboost elapsed: {e}".format(e=(end - start)))

train_df = DataFrame.from_pandas(df)

distance = train_df["Distance"]
deptime = train_df["DepTime"]
dep_delayed_15min = train_df["dep_delayed_15min"]
label = Covariate.from_factor(dep_delayed_15min, lambda level: level == "Y")
deptime = Factor.from_covariate(train_df["DepTime"]).cached()
distance = Factor.from_covariate(train_df["Distance"]).cached()

factors = train_df.factors + [deptime, distance]
factors.remove(dep_delayed_15min)

def test_1():
    start = datetime.now()
    trees, pred = xgblogit(label, factors,  eta = eta, lambda_ = 1.0,
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


