from abc import ABC, abstractmethod

class AbstractTreeNode(ABC):
    pass

class LeafNode(AbstractTreeNode):
    def __init__(self, ghsum, cansplit, partitions, loss):
        self.ghsum = ghsum
        self.cansplit = cansplit
        self.partitions = partitions
        self.loss = loss

class SplitNode(AbstractTreeNode):
    def __init__(self, factor, leftnode, rightnode, loss, gain):
        self.factor = factor
        self.leftnode = leftnode
        self.rightnode = rightnode
        self.loss = loss
        self.gain = gain

class TreeGrowState():
    def __init__(self, nodeids, nodes, factors, gcovariate, hcovariate, gamma, lambda_, minh, slicelength):
        self.nodeids = nodeids
        self.nodes = nodes
        self.factors = factors
        self.gcovariate = gcovariate
        self.hcovariate = hcovariate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.minh = minh
        self.slicelength = slicelength