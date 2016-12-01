#coding:utf-8

from numpy import where, sort, cumsum, argsort, array, bincount, full, log2, outer

def head(dist, p=1.0, n=None, eps=None):
    """
    :param dist: a numpy array of numeric values representing frequencies or probabilities (any shape)
    :param p: a float specifying the amount of the distribution to keep as a proportion of the total
    :return: head - a tuple (of length the dimension of dist) of numpy arrays of indices giving locations of the
     head distribution items according to dist.
    """
    """
    returns positions in a array (dist) of non-negative numbers accounting for
    at least p of the total sum of all entries in dist.
    """
    dist = array(dist)
    if eps is not None:
        total = dist.sum()
        indices = where(dist > eps*total)
    elif n is not None or p is not None:
        sorted_dist = sort(dist.flatten())
        if n is not None:
            cutoff = sorted_dist[-1*n]
        else: # p is not None:
            cum_dist = cumsum(sorted_dist)
            cutoff = sorted_dist[searchsorted(cum_dist, (1.0 - p)*cum_dist[-1], side='right')]
        indices = where(dist >= cutoff)
    
    order = argsort(dist[indices])[::-1]
    return tuple(i[order] for i in indices)

############################################################
# Information-theoretic functions ##########################
############################################################

def KL(p,q,normalize=False):
    """
    KL-divergence from q to p;
    p is typically thought of as ground truth
    """
    if normalize:
        p = p/p.sum()
        q = q/q.sum()
    
    kl = ((log2(p) - log2(q))*p)
    kl[p == 0.0] = 0.0
    return kl.sum()


def H(p,q=None,normalize=False):
    """
    entropy of a discrete distribution, 
    or cross-entropy w.r.t another discrete dist
    """
    if q is None:
        q = p
    if normalize:
        p = p/p.sum()
    h = log2(q)*p
    h[p==0.0] = 0.0
    return -1.0*h.sum()


############################################################
# Useful things for clustering evaluation ##################
############################################################

def distributions(labels1,labels2,smoothing=0.0):
    """
    marginal and joint distributions for a sequence of observations
    from a pair of disrete random variables, with additive smoothing on the
    joint distribution and the marginals in such a way that 
    marginal(smooth(conditional)) = smooth(marginal)
    """
    if len(labels1) != len(labels2):
        raise ValueError("label lists must have the same length")
    
    set1 = set(labels1)
    set2 = set(labels2)
    n1 = len(set1)
    n2 = len(set2)
    l2i1 = dict(zip(set1,range(len(set1))))
    l2i2 = dict(zip(set2,range(len(set2))))
    l1 = array([l2i1[l] for l in labels1])
    l2 = array([l2i2[l] for l in labels2])
    
    d1 = bincount(l1) + smoothing*n2
    d1 = d1/d1.sum()
    d2 = bincount(l2) + smoothing*n1
    d2 = d2/d2.sum()
    
    a_true = full((n1,n2),smoothing)
    for i,j in zip(l1,l2):
        a_true[i,j] += 1.0
    a_true = a_true/a_true.sum()
    
    return d1,d2,a_true
    

def MI(labels1,labels2,smoothing = 0.0):
    d1,d2,a_true = distributions(labels1,labels2,smoothing)
    
    a_ind = outer(d1,d2)
    
    # these should be equal
    #print(H(d1) + H(d2) - H(a_true))
    #print(KL(a_true,a_ind))
    return KL(a_true,a_ind)


def NMI(labels1,labels2,smoothing = 0.0):
    d1,d2,a_true = distributions(labels1,labels2,smoothing)
    
    a_ind = outer(d1,d2)
    
    return 2.0*KL(a_true,a_ind)/(H(d1) + H(d2))

