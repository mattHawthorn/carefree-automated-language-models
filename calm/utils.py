#coding:utf-8
import sys
from numbers import Number
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
        
    def __call__(self,x):
        try:
            return self.cache[x]
        except:
            value = self.f(x)
            self.cache[x] = value
            return value


def binarySearch(x,listlike):
    # return the index of the first item in listlike which exceeds x. Returns None if not found.
    # Could also do np.searchsorted(listlike,x,side='left'), and check the return against len(listlike)
    # since this returns max index of listlike plus 1 when no item is found.
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

