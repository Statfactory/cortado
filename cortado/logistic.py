from cortado.seq import Seq
from cortado.covariate import Covariate
from cortado.split import growtree
import numpy as np
from numba import jit
from numba import generated_jit
from datetime import datetime

@jit(nopython=True, cache=True)
def logit_g(yhat, y):
    return yhat - y

@jit(nopython=True, cache=True)
def logit_h(yhat):
    eps = np.finfo(np.float32).eps
    return max(yhat * (1 - yhat), eps)

@jit(nopython=True, cache=True)
def logitraw(p):
    return -np.log(1.0 / p - 1.0)

@jit(nopython=True, cache=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xgblogit(label, factors, trainselector, mu = 0.5, eta = 0.1, lambda_ = 1.0, gamma = 0.0,
             maxdepth = 2, nrounds = 2, minh = 1.0, slicelen = 10000):

    f0 = np.empty(len(label), dtype=np.float32)
    f0.fill(logitraw(mu))
    label = label.to_array()
    g = np.zeros(len(f0), dtype=np.float32)
    h = np.zeros(len(f0), dtype=np.float32)

    @jit(nopython=True, cache=True)
    def get_gh(b, yhatraw, label, g, h):
        for i in range(len(g)):
            if b[i]:
                yhat = sigmoid(yhatraw[i])
                y = label[i]
                g[i] = logit_g(yhat, y)
                h[i] = logit_h(yhat)

    @jit(nopython=True, cache=True)
    def get_pred(fm):
        for i in range(len(fm)):
            fm[i] = sigmoid(fm[i])

    def step(x, m):
        fm, trees = x
        get_gh(trainselector, fm, label, g, h)
        g_cov = Covariate.from_array(g)
        h_cov = Covariate.from_array(h)
        tree, predraw = growtree(factors, g_cov, h_cov, fm, eta, maxdepth, lambda_, gamma, minh, slicelen)
        
        trees.append(tree)
        return (predraw, trees)

    fm, trees = Seq.reduce(step, (f0, []), Seq.from_gen((i for i in range(nrounds)))) 
    get_pred(fm)
    return trees, fm

