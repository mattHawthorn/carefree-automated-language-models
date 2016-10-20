#coding:utf-8
import sys
from functools import partial
from itertools import chain
from numbers import Number
from functools import partial
from collections import Set, Mapping, deque


zero_depth_bases = (str, bytes, range, bytearray)
iteritems = 'items'

def getsize(obj_0):
    """
    Recursively iterate into sub-objects to determine total memory usage of an object.
    From Stack Exchange answer of Aaron Hall on May 19 '15 at 4:26
    """
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)
    
    
class Memoized:
    __slots__=('cache','f')
    def __init__(self,f):
        self.cache = {}
        self.f = f
        
    def __call__(self,*x):
        try:
            return self.cache[x]
        except:
            value = self.f(x)
            self.cache[x] = value
            return value


class MemoizedMethod(object):
    """cache the return value of a method
        
       This class is meant to be used as a decorator of methods. The return value
       from a given method invocation will be cached on the instance whose method
       was invoked. All arguments passed to a method decorated with memoize must
       be hashable.
               
       If a memoized method is invoked directly on its class the result will not
       be cached. Instead the method will be invoked like a static method:
       class Obj(object):
           @memoize
           def add_to(self, arg):
               return self + arg
       Obj.add_to(1) # not enough arguments
       Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
        
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
        
    def __call__(self, *args):
        obj = args[0]
        cache = obj.__cache__
        key = (self.func, args[1:])
        try:
            result = cache[key]
        except KeyError:
            result = cache[key] = self.func(*args)
        return result
    

def substitute_keys(d,subs,check=False):
    """
    take a dict d and a dict subs and return the result of applying subs to every key in d
    while keeping the values constant
    """
    if check:
        if len(set(subs).intersection(d)) < len(d):
            raise ValueError("Substitution map does not contain all keys in d")
    return {subs[key]:value for key,value in d.items()}

def compose_maps(d1,d2,check=False):
    """
    take a dict d and a dict subs and return the result of applying subs to every key in d
    while keeping the values constant
    """
    if check:
        if len(set(d2).intersection(d1.values)) < len(d1):
            raise ValueError("d1 has values not in the keys of d2")
    return {key:d2[value] for key,value in d1.items()}

    
def binarySearch(x,listlike):
    # return the index of the first item in listlike which exceeds x. Returns None if not found.
    # Could also do np.searchsorted(listlike,x,side='left'), and check the return against len(listlike)
    # since this returns max index of listlike plus 1 when no item is found.
    # Useful for sampling from discrete probability distributions
    if len(listlike) == 0:
        return None
    
    low = 0
    high = len(listlike) - 1
    i = int((low + high)/2)
    
    if x > listlike[-1]:
        return None
    
    while low != i and high != i:
        if x < listlike[i]:
            high = i
        else:
            low = i
        
        i = int((low + high)/2)
    
    if listlike[i] >= x:
        return i
    else:
        return i+1


def ngramIter(tokens,n,start=None,end=None,head=False):
    ngram = deque()
    tokens = iter(tokens)
    i=1
    if start:
        i+=1
        ngram.append(start)
        if head:
            yield tuple(ngram)
        if n == 1:
            ngram.popleft()
    while i < n:
        i+=1
        ngram.append(next(tokens))
        if head:
            yield tuple(ngram)

    for token in tokens:
        ngram.append(token)
        yield tuple(ngram)
        ngram.popleft()
    if end:
        ngram.append(end)
        yield tuple(ngram)


def charNgramIter(s,n,start=None,end=None,head=False):
    if end:
        s = s + end
    if start:
        s = start + s
    if head:
        return map(lambda i: s[i:(i+n)],range(len(s) - n + 1))
    else:
        return chain(map(lambda i: s[0:i],range(0,n-1)), map(lambda i: s[i:(i+n)],range(len(s) - n + 1)))
    

def rollSum(indicators,n,start=None,end=None):
    ngram = deque()
    tokens = iter(indicators)
    i=0
    s = 0
    if start is not None:
        i+=1
        ngram.append(start)
        s += start
    while i < n:
        i+=1
        t = next(tokens)
        s += t
        ngram.append(t)
    yield s
    for token in tokens:
        ngram.append(token)
        s += token
        s -= ngram.popleft()
        yield s
    if end is not None:
        yield s + end - ngram.popleft()

