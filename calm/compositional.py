#coding:utf-8

from numpy import array, log, exp

def sum_normalize(a):
    arr = array(a)
    return arr/arr.sum()

def clr(p):
    """
    centered log-ratio transformation; inverse is softmax.
    p is a matrix with discrete probability distributions in the rows
    """
    logp = log(p)
    return (logp.T - logp.mean(-1)).T

def softmax(logp):
    """
    softmax transform; inverse of clr when clr is restricted to the unit simplex
    """
    p = exp(logp)
    return ((p.T)/p.sum(-1)).T

def closure(a):
    """
    closure operation on compositional data;
    project rows of a onto the unit simplex
    """
    sums = array(a).sum(-1)
    return (a.T/sums).T

def distributional_center(p, weights=None):
    """
    p is a matrix with discrete probability distributions in the rows.
    perform clr, take standard feature-wise mean, then sofmax
    """
    p = array(p)
    if len(p.shape) == 1:
        return p
    logp = clr(p)
    if weights is not None:
        logmean = (logp.T*weights).sum(-1)/weights.sum()
    else:
        logmean = logp.mean(0)
    return softmax(logmean)
