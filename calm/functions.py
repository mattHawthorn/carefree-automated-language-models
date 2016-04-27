#coding: utf-8
from numpy import sqrt, log, square, array, sum, ones, zeros

########################################
# FUNCTIONS ############################
########################################

def IDF(df,docCount,offset=1.0):
    # takes a numpy array and applies IDF
    return offset + log(docCount/df)


def sublinearTF(tf):
    # takes a numpy array and applies sublinear tf
    # adding 1 ensures no problems with 0 counts
    return log(tf + 1.0)


def tfidfVector(bagOfWords,DF,docCount,dfweighting=IDF,tfweighting = None,normalize=True):
    keys = list(set(bagOfWords).intersection(DF))
    v = array([bagOfWords[k] for k in keys],dtype='float')
    if tfweighting:
        v = tfweighting(v)
    idf = dfweighting(array([DF[k] for k in keys]),docCount=docCount)
    v = idf*v
    
    if normalize:
        norm = sqrt(sum(square(v)))
        if norm == 0.0:
            v = zeros(len(v),dtype='float')
        else:
            v = v/norm
    
    v = dict(zip(keys,v))
    
    return v
    

def cosineSimilarity(bagOfWords1,bagOfWords2,DF=None,docCount=None,dfweighting=IDF,tfweighting=None):
    """
    First three arguments are hashmaps from token ID's to counts.
    weights can be computed in a customized way by putting any function of in for weighting;
    the default is the standard IDF.
    Weighting functions should take a numpy array and return a numpy array.
    """
    # relevant terms for each BOW
    if DF:
        keys1 = set(bagOfWords1).intersection(DF)
        keys2 = set(bagOfWords2).intersection(DF)
    else:
        keys1 = set(bagOfWords1)
        keys2 = set(bagOfWords2)
    
    # relevant terms common to both BOWs
    keys = keys1.union(keys2)
    v1=array([bagOfWords1[k] if k in bagOfWords1 else 0.0 for k in keys],dtype='float')
    v2=array([bagOfWords2[k] if k in bagOfWords2 else 0.0 for k in keys],dtype='float')
    if tfweighting:
        v1 = tfweighting(v1)
        v2 = tfweighting(v2)
    if DF:
        idf=dfweighting(array([DF[k] for k in keys]),docCount=docCount)
        v1=idf*v1
        v2=idf*v2
    
    norm1 = sqrt(sum(square(v1)))
    norm2 = sqrt(sum(square(v2)))
    if norm1*norm2 == 0.0:
        return 0.0
    
    # only need the product over the intersection
    keys2=keys1.intersection(keys2)
    index=array([key in keys2 for key in keys])
    
    return sum(v1[index]*v2[index])/(norm1*norm2)
    
    
class Memoizer:
    __slots__=('cache','f')
    def __init__(self,f):
        self.cache = {}
        self.f = f
        
    def __call__(self,x):
        try:
            return self.cache[x]
        except:
            value = self.f(x)
            self.cache[x] = value
            return value
        
