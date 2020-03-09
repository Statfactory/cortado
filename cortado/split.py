from cortado.seq import Seq
from cortado.constfactor import ConstFactor
from cortado.tree import LeafNode, SplitNode, TreeGrowState
from cortado.vectorslicer import VectorSlicer
from cortado.nfactorslicer import NFactorSlicer
import numpy as np
from numba import jit
from functools import reduce
from datetime import datetime

@jit(nopython=True, cache=False)
def f_splitnode(nodeslice, fslices, issplitnode, leftpartitions, factorindex, nodemap):
    for i in range(len(nodeslice)):
        nodeid = nodeslice[i]
        if issplitnode[nodeid]:
            levelindex = fslices[factorindex[nodeid], i]
            nodeslice[i] = nodemap[nodeslice[i]] if leftpartitions[nodeid, levelindex] else nodemap[nodeslice[i]] + 1
        else:
            nodeslice[i] = nodemap[nodeslice[i]]

@jit(nopython=True, cache=False)
def f_hist(acc0, zipslice):
    nodeslice, factorslice, gslice, hslice = zipslice
    gsum, hsum, nodecansplit = acc0
    for i in range(len(nodeslice)):
        nodeid = nodeslice[i]
        if nodecansplit[nodeid]:
            levelindex = factorslice[i]
            gsum[nodeid, levelindex] += gslice[i]
            hsum[nodeid, levelindex] += hslice[i]
    return (gsum, hsum, nodecansplit)

@jit(nopython=True, cache=False)
def f_weights(nodeids, weights, fm, eta):
    for i in range(len(nodeids)):
        fm[i] = eta * weights[nodeids[i]] + fm[i]

@jit(nopython=True, cache=False)
def get_weight(g, h, lambda_): 
    return -g / (h + lambda_)

@jit(nopython=True, cache=False)
def getloss(g, h, lambda_): 
    return -(g * g) / (h + lambda_)

@jit(nopython=True, cache=False)
def get_best_stump_split(g_hist, h_hist, partition, lambda_, minh):
    gsum = np.sum(g_hist)
    hsum = np.sum(h_hist)
    currloss = getloss(gsum, hsum, lambda_) 
    bestloss = currloss
    stump_index = -1
    
    for i in range(len(partition)):
        if not partition[i]:
            continue
        loss = getloss(g_hist[i], h_hist[i], lambda_) + getloss(gsum - g_hist[i], hsum - h_hist[i], lambda_)
        if loss < bestloss and (h_hist[i] >= minh) and (hsum - h_hist[i] >= minh):
            bestloss = loss
            stump_index = i
    
    if stump_index >= 0:
        leftpartition = np.copy(partition)
        rightpartition = np.copy(partition)
        for i in range(len(partition)):
            if partition[i]:
                if i == stump_index:
                    rightpartition[i] = False
                else:
                    leftpartition[i] = False
        leftgsum = g_hist[stump_index]
        lefthsum = h_hist[stump_index]
        rightgsum = gsum - leftgsum
        righthsum = hsum - lefthsum
        return (currloss, bestloss, leftpartition, rightpartition, leftgsum, lefthsum, rightgsum, righthsum)
    else:
        return (currloss, bestloss, partition, partition, 0.0, 0.0, 0.0, 0.0)

@jit(nopython=True, cache=False)
def get_best_range_split(g_hist, h_hist, partition, lambda_, minh):
    gsum = np.sum(g_hist)
    hsum = np.sum(h_hist)
    currloss = getloss(gsum, hsum, lambda_)
    bestloss = currloss
    split_index = -1
    miss_left = True
    miss_g = g_hist[0]
    miss_h = h_hist[0]
    miss_active = partition[0] and (miss_g + miss_h > 0.0) and miss_h >= minh

    gcumsum = 0.0
    hcumsum = 0.0
    if miss_active:
        for i in range(len(partition) - 1):
            if not partition[i]:
                continue
            gcumsum += g_hist[i]
            hcumsum += h_hist[i]
            loss_miss_left = getloss(gcumsum, hcumsum, lambda_) + getloss(gsum - gcumsum, hsum - hcumsum, lambda_)
            loss_miss_right = loss_miss_left if i == 0 else getloss(gcumsum - miss_g, hcumsum - miss_h, lambda_) + getloss(gsum - gcumsum + miss_g, hsum - hcumsum + miss_h, lambda_)
            if loss_miss_left < bestloss and (hcumsum >= minh) and (hsum - hcumsum >= minh):
                bestloss = loss_miss_left
                split_index = i
                miss_left = True
                leftgsum = gcumsum
                lefthsum = hcumsum
                rightgsum = gsum - gcumsum
                righthsum = hsum - hcumsum

            if loss_miss_right < bestloss and (hcumsum - miss_h >= minh) and (hsum - hcumsum + miss_h >= minh):
                bestloss = loss_miss_right
                split_index = i
                miss_left = False
                leftgsum = gcumsum if i == 0 else gcumsum - miss_g
                lefthsum = hcumsum if i == 0 else hcumsum - miss_h
                rightgsum = gsum - gcumsum if i == 0 else gsum - gcumsum + miss_g
                righthsum = hsum - hcumsum if i == 0 else hsum - hcumsum + miss_h

    else:
        for i in range(0, len(partition) - 1):
            if not partition[i]:
                continue
            gcumsum += g_hist[i]
            hcumsum += h_hist[i]
            loss = getloss(gcumsum, hcumsum, lambda_) + getloss(gsum - gcumsum, hsum - hcumsum, lambda_)
            if loss < bestloss and (hcumsum >= minh) and (hsum - hcumsum >= minh):
                bestloss = loss
                split_index = i
                miss_left = True
                leftgsum = gcumsum
                lefthsum = hcumsum
                rightgsum = gsum - gcumsum
                righthsum = hsum - hcumsum

    if split_index >= 0:
        leftpartition = np.copy(partition)
        rightpartition = np.copy(partition)
        for i in range(len(partition)):
            if partition[i]:
                if i == 0 and miss_active:
                    leftpartition[i] = miss_left
                    rightpartition[i] = not miss_left
                else:
                    leftpartition[i] = i <= split_index
                    rightpartition[i] = i > split_index
        return (currloss, bestloss, leftpartition, rightpartition, leftgsum, lefthsum, rightgsum, righthsum)
    else:
        return (currloss, bestloss, partition, partition, 0.0, 0.0, 0.0, 0.0)

def get_hist_slice(gsum0, hsum0, nodeids, nodecansplit, factor, gcovariate, hcovariate,
                   start, length, slicelen): 

    nodeslices = VectorSlicer(nodeids)(start, length, slicelen)
    factorslices = factor.slicer(start, length, slicelen)
    gslices = gcovariate.slicer(start, length, slicelen)
    hslices = hcovariate.slicer(start, length, slicelen)
    zipslices = Seq.zip(nodeslices, factorslices, gslices, hslices)

    return Seq.reduce(f_hist, (gsum0, hsum0, nodecansplit), zipslices)

def get_histogram(nodeids, nodecansplit, factor, gcovariate, hcovariate, slicelen):
    
    nodecount = len(nodecansplit)
    levelcount = len(factor.levels)
    start = 0
    length = len(nodeids)

    gsum = np.zeros((nodecount, levelcount), dtype=np.float64) #np.array([np.zeros(levelcounts[node], dtype=np.float32) if nodecansplit[node] else np.empty(0, dtype=np.float32) for node in range(nodecount)])
    hsum = np.zeros((nodecount, levelcount), dtype=np.float64) #np.array([np.zeros(levelcounts[node], dtype=np.float32) if nodecansplit[node] else np.empty(0, dtype=np.float32) for node in range(nodecount)])

    get_hist_slice(gsum, hsum, nodeids, nodecansplit, factor, gcovariate, hcovariate, start, length, slicelen)
    return (gsum, hsum)

def splitnodeidsslice(nodeids, factors, issplitnode, nodemap, leftpartitions, factorindex,
                      start, length, slicelength):
    if len(factors) > 0:
        factorslices = NFactorSlicer(factors)(start, length, slicelength) 
        nodeslices = VectorSlicer(nodeids)(start, length, slicelength)

        Seq.foreach(lambda x: f_splitnode(x[0], x[1], issplitnode, leftpartitions, factorindex, nodemap), Seq.zip(nodeslices, factorslices)) 
        
def splitnodeids(nodeids, nodes, slicelength):
    nodecount = len(nodes)
    length = len(nodeids)
    start = 0
    issplitnode = [isinstance(n, SplitNode) for n in nodes]
    nodemap = []
    splitnodecount = 0
    for (i, x) in enumerate(issplitnode):
        nodemap.append(i + splitnodecount) 
        if x:
            splitnodecount += 1
    nodemap = np.array(nodemap)
    issplitnode = np.array(issplitnode)

    factors = []
    factorindex = np.empty(nodecount, dtype=np.int32)
    for i in range(nodecount):
        if issplitnode[i]:
            factor = nodes[i].factor
            try:
                _ = factors.index(factor)
            except ValueError:
                factors.append(factor)
            factorindex[i] = factors.index(factor)
        else:
            factorindex[i] = -1

    maxlevelcount = max([len(n.leftnode.partitions[n.factor]) if isinstance(n, SplitNode) else 0 for n in nodes])   
    leftpartitions = np.empty((nodecount, maxlevelcount), dtype=np.bool)
    for i in range(nodecount):
        if issplitnode[i]:
            n = nodes[i]
            leftpart = n.leftnode.partitions[n.factor]

            leftpartitions[i, :len(leftpart)] = leftpart

    splitnodeidsslice(nodeids, factors, issplitnode, nodemap, leftpartitions, factorindex,
                      start, length, slicelength)
    return nodeids

def get_splitnode(factor, leafnode, histogram, lambda_, minh):

    g_hist, h_hist = histogram
    partition = leafnode.partitions[factor]

    if factor.isordinal:
        currloss, bestloss, leftpartition, rightpartition, leftgsum, lefthsum, rightgsum, righthsum = get_best_range_split(g_hist, h_hist, partition, lambda_, minh)
    else:
        currloss, bestloss, leftpartition, rightpartition, leftgsum, lefthsum, rightgsum, righthsum = get_best_stump_split(g_hist, h_hist, partition, lambda_, minh)

    if bestloss < currloss:
        leftpartitions = {k : v for k, v in leafnode.partitions.items()}
        rightpartitions = {k : v for k, v in leafnode.partitions.items()}
        leftpartitions[factor] = leftpartition
        rightpartitions[factor] = rightpartition
        leftcansplit = any([np.sum(v) > 1 for k, v in leftpartitions.items()]) 
        rightcansplit = any([np.sum(v) > 1 for k, v in rightpartitions.items()])
        left_gh = leftgsum, lefthsum
        right_gh = rightgsum, righthsum
        leftloss = getloss(leftgsum, lefthsum, lambda_)
        rightloss = getloss(rightgsum, righthsum, lambda_)
        leftnode = LeafNode(left_gh, leftcansplit, leftpartitions, leftloss)
        rightnode = LeafNode(right_gh, rightcansplit, rightpartitions, rightloss)        
        return SplitNode(factor, leftnode, rightnode, bestloss, currloss - bestloss)
    else:
        return None

def getnewsplit(histograms, nodes, factor, lambda_, gamma, minh):
    newsplit = [None for i in range(len(nodes))]
    for i in range(len(nodes)):
        hist = (histograms[0][i,:], histograms[1][i,:])
        if isinstance(nodes[i], LeafNode) and nodes[i].cansplit:
            newsplit[i] = get_splitnode(factor, nodes[i], hist, lambda_, minh)
    return newsplit

def findbestsplit(state):

    nodecansplit = [isinstance(n, LeafNode) and n.cansplit for n in state.nodes]
    mingain = state.gamma

    def f(currsplit, factor):
        histograms = get_histogram(state.nodeids, nodecansplit, factor, state.gcovariate, state.hcovariate, state.slicelength)

        newsplit = getnewsplit(histograms, state.nodes, factor, state.lambda_, state.gamma, state.minh)

        res = [None for i in range(len(newsplit))]
        for i in range(len(newsplit)):
            if newsplit[i] is not None:
               newloss = newsplit[i].loss
               newgain = newsplit[i].gain
               if newgain > mingain and newloss < currsplit[i].loss:
                   res[i] = newsplit[i]
               else:
                   res[i] = currsplit[i] 
            else:
               res[i] = currsplit[i] 
        return res

    return reduce(f, state.factors, state.nodes)

def updatestate(state, layernodes):
    splitnodeids(state.nodeids, layernodes, state.slicelength)  

    newnodes = []
    for n in layernodes:
        if isinstance(n, SplitNode):
            newnodes.append(n.leftnode)
            newnodes.append(n.rightnode)
        else:
            newnodes.append(n)
    state.nodes = newnodes
    return state

def nextlayer(state):
    layernodes = findbestsplit(state)
    updatestate(state, layernodes)
    return layernodes, state      

def predict(treenodes, nodeids, fm, eta, lambda_):
    weights = np.empty(len(treenodes), dtype=np.float32)
    for (i, node) in enumerate(treenodes):
        weights[i] = -node.ghsum[0] / (node.ghsum[1] + lambda_)

    f_weights(nodeids, weights, fm, eta)

def growtree(factors, gcovariate, hcovariate, fm, eta, maxdepth, lambda_, gamma, minh, slicelen):

    length = len(gcovariate)
    maxnodecount = 2 ** maxdepth
    nodeids = np.zeros(length, dtype=np.uint8) if maxnodecount <= np.iinfo(np.uint8).max else np.zeros(length, dtype=np.uint16)

    loss0 = np.finfo(np.float32).max
    nodes0 = [LeafNode((0.0, 0.0), True, {f : np.full(len(f.levels), True) for f in factors}, loss0)]
    
    state0 = TreeGrowState(nodeids, nodes0, factors, gcovariate, hcovariate, gamma, lambda_, minh, slicelen)
    layers = Seq.tolist(Seq.take(Seq.from_next(state0, nextlayer), maxdepth))
    predict(state0.nodes, nodeids, fm, eta, lambda_)
    return layers, fm