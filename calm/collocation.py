#coding:utf-8

from itertools import tee, chain

# Collocation

def basic_ngram_joiner(joinchar):
    def join(ngram):
        return joinchar.join(ngram)
    return join

def join_bigrams(tokenlist, bigramset, joinfunc=basic_ngram_joiner("_")):
    bigrams = window(tokenlist,2)
    joins = [(bigram in bigramset) for bigram in bigrams]
    joinedtokens = [tokenlist[0]]
    for i,join in enumerate(joins):
        if join:
            tail = joinedtokens[-1]
            joinedtokens[-1] = (*tail, tokenlist[i+1])
        else:
            joinedtokens.append((tokenlist[i+1],))
    
    return list(map(joinfunc, joinedtokens))

# for collecting ngrams
def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)
