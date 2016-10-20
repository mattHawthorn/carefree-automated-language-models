from numpy import int as np_int
from numpy import array,where,log,log2,dot,arange,floor_divide,stack,hsplit,trim_zeros,concatenate
from struct import pack, pack_into, unpack, unpack_from

def to_base(n,b,n_digits=None):
    """
    n can be an unsigned int, or list or ndarray of unsigned ints
    return type has shape n.shape + (n_digits,), where n_digits is inferred if not passed;
    thus putting d in the last index gives the digit at position d for the int referenced 
    by the first indices in the original array
    """
    if len(n) == 0:
        return array((),dtype=np_int)
    n = array(n)
    if n_digits is None:
        m = n.max()
        if m > 0:
            n_digits = np_int(log(m)/log(b) + 1)
        else:
            n_digits = 1
    places = np_int(b) ** arange(n_digits)
    return floor_divide.outer(n,places) % np_int(b)
    
def from_base(a,b):
#     n = 0
#     for i in reversed(a):
#         n = b*n + i
#     return n
    places = np_int(b) ** arange(len(a))
    return dot(a,places)

def compressInts(a):
    """
    a is assumed to be an int-indexable containing unsigned ints.
    """
    if len(a) == 0:
        return bytes()
    digits = to_base(a,127)
    digits[:,0] *= -1
    digits[:,0] -= 1
    compressed = bytearray()
    for i in range(digits.shape[0]):
        ds = trim_zeros(digits[i,:],'b')
        if len(ds) == 0:
            ds = (0,)
        compressed.extend(pack('{}b'.format(len(ds)),*ds))
    return bytes(compressed)
    
def decompressInts(a):
    """
    a is assumed to be an int-indexable or iterable of bytes
    """
    if len(a) == 0:
        return array((),dtype=np_int)
    a = array(bytearray(a),dtype='int8')
    starts = where(a<0)[0]
    a[starts] += 1
    a[starts] *= -1
    nums = hsplit(a, starts[1:])
    return array([from_base(n,127) for n in nums])

