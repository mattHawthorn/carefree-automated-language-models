#coding: utf-8

from numpy import dot, exp, square, sqrt

# Operations for vectors/arrays

def unitNormalize(a):
    norms = sqrt(square(a).sum(axis=-1))
    if len(norms.shape) > 0:
        norms[norms <= 0.0] = 1.0
        return (a.T/norms).T
    else:
        if norms <= 0:
            norms = 1.0
        return a/norms

# Metrics for vectors/arrays

def cosine(v1,v2):
    return dot(v1,v2)/(norm(v1)*norm(v2))

def norm(v):
    return sqrt((square(v)).sum(axis=-1))

def normsq(v):
    return square(v).sum(axis=-1)

def euc(v1,v2):
    return norm(v1-v2)

def gauss_kernel(v1,v2,sigma=1.0):
    return exp(-normsq(v1-v2)/(sigma**2))

def vmf_kernel(v1,v2,kappa=1.0):
    return exp(kappa*dot(v1/norm(v1),v2/norm(v2)))

