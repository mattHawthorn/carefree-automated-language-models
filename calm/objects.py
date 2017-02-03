#coding: utf-8
from .utils import Memoized,ngramIter
from itertools import repeat

########################################
# BAGS OF WORDS AND NGRAMS #############
########################################


class BagOfWords(dict):
    # inherits from dict (hashmap), with added drop and add methods as well as counters.
    # useful not only for document representations but also for corpus-wide term frequency dictionaries (TTF/DF)
    __slots__=('total')
    def __init__(self,tokens=None):
        self.total = 0
        if tokens:
            self._addmany(tokens)

    def add(self,token,count=1):
        if token not in self:
            self[token] = count
        else:
            self[token] += count
        self.total += count

    def addMany(self,tokens,counts=None):
        if counts is None:
            self._addmany(tokens)
        else:
            self._addmanyCounts(zip(tokens,counts))
    
    def addBagOfWords(self,bagOfWords):
        # note that the syntax here allows bagOfWords to be any indexed object with numeric values;
        # it need not be a BagOfWords object.
        # Use count = 1 for a DF dictionary
        self._addmanyCounts(bagOfWords.items())

    def subtractBagOfWords(self,bagOfWords):
        self._subtractmanyCounts(bagOfWords.items())
    
    def _addmany(self,tokens):
        total = 0
        for i,token in enumerate(tokens):
            total += 1
            if token not in self:
                self[token] = 1
            else:
                self[token] += 1
        self.total += total
    
    def _addmanyCounts(self,tokencounts):
        for token,count in tokencounts:
            if token not in self:
                self[token] = count
            else:
                self[token] += count
            self.total += count
    
    def _subtractmanyCounts(self,tokencounts, decrement=True):
        drops = []
        for token,count in tokencounts:
            if token in self:
                cur_count = self[token]
                sub_count = min(cur_count, count)                
                cur_count -= sub_count
                if cur_count == 0:
                    drops.append(token)
                else:
                    self[token] = cur_count
                    self.total -= sub_count
        for token in drops:
            self.drop(token, decrement)
    
    def __iadd__(self,other):
        # enable use of self += BagOfWords (or BagOfNgrams)
        self._addmanyCounts(other.items())
    
    def __isub__(self,other):
        # enable use of self += BagOfWords (or BagOfNgrams)
        self._subtractmanyCounts(other.items())
            
    def drop(self,token,decrement=False):
        if not decrement:
            if token in self:
                del self[token]
        else:
            if token in self:
                count = self.pop(token)
                self.total -= count

    def dropMany(self,tokens,decrement=False):
        # else a list or other iterable of ngrmas/tokens
        for token in tokens:
            self.drop(token,decrement)
    
    addManyCounts = _addmanyCounts
    subtractManyCounts = _subtractmanyCounts


class MultiSet(BagOfWords):
    pass


@Memoized
def joined_ngram_counter(joinchar):
    def f(ngram):
        return ngram.count(joinchar) + 1
    return f

class BagOfNgrams:
    """
    simply a list of BagOfWords objects containing counts only for ngrams of length the index.
    indexing methods defined to take this into account
    this particular implementation assumes that ngrams take the form ("token_1","token_2",...,"token_n")
    """
    __slots__=['max_n','ngrams','total','joinchar','_nfunc']
    def __init__(self,max_n,joinchar=None):
        self.total = 0
        self.max_n = max_n
        self.joinchar = joinchar
        self._nfunc = (len if self.joinchar is None else joined_ngram_counter(joinchar))
        self.ngrams = tuple([()] + [BagOfWords() for i in range(max_n)])
        
    def __copy__(self):
        bag = BagOfNgrams(max_n=self.max_n,joinchar=self.joinchar)
        bag.total = self.total
        bag.ngrams = tuple(copy(bow) for bow in self.ngrams)
        return bag
    
    def __getitem__(self,ngram):
        try:
            return self.ngrams[self._nfunc(ngram)][ngram]
        except (KeyError,IndexError):
            return 0

    def __setitem__(self,ngram,count):
        try:
            self.ngrams[self._nfunc(ngram)][ngram] = count
        except IndexError as e:
            raise e("This bag of ngrams contains only ngrams of length {} or less".format(self.max_n))

    def __contains__(self,ngram):
        n = self._nfunc(ngram)
        return n <= len(self.ngrams) and ngram in self.ngrams[n]
        
    def get(self,ngram,default):
        if ngram in self:
            return self[ngram]
        else:
            return default

    def __len__(self):
        return sum(map(len,self.ngrams))
        
    def items(self):
        for bow in self.ngrams[1:]:
            for token,count in bow.items():
                yield token,count
                
    def keys(self):
        for bow in self.ngrams:
            for token in bow:
                yield token
                
    def __iter__(self):
        return self.keys()
        
    def add(self,ngram,count=1):
        try:
            ngrams = self.ngrams[self._nfunc(ngram)]
        except IndexError as e:
            raise e("This bag of ngrams contains only ngrams of length {} or less".format(self.max_n))
        
        if ngram in ngrams:
            ngrams[ngram] += count
        else:
            ngrams[ngram] = count
        self.total += 1
    
    def addMany(self,ngrams,counts=None):
        if counts is None:
            self._addmany(ngrams)
        else:
            self._addmanyCounts(zip(ngrams,counts))
    
    def addBagOfWords(self,bagOfWords):
        # bagOfwords needs to have an items() method and numeric values (i.e. a BagOfWords object)
        self._addmanyCounts(bagOfWords.items())
            
    def _addmany(self,ngrams):
        for i,ngram in enumerate(ngrams):
            if ngram not in self:
                self[ngram] = 1
            else:
                self[ngram] += 1
        self.total += (i+1)
    
    def _addmanyCounts(self,ngramcounts):
        for ngram,count in ngramcounts:
            if ngram not in self:
                self[ngram] = count
            else:
                self[ngram] += count
            self.total += count
    
    def __iadd__(self,other):
        # enable use of self += BagOfWords (or BagOfNgrams)
        self._addmanyCounts(other.items())
    
    def __isub__(self,other):
        # enable use of self += BagOfWords (or BagOfNgrams)
        self._addmanyCounts((k, -v) for k, v in other.items())

    def drop(self,ngram):
        if ngram in self:
            del self.ngrams[self._nfunc(ngram)][ngram]
            
    def dropMany(self,ngrams):
        for ngram in ngrams:
            if ngram in self:
                del self.ngrams[self._nfunc(ngram)][ngram]

    def distinctNgrams(self,n):
        return len(self.ngrams[n])
        
    def totalNgrams(self,n):
        return self.ngrams[n].total
    
    addManyCounts = _addmanyCounts



##################################################
# TRIE FOR NGRAM LANGUAGE MODELS #################
##################################################

class FrequencyTrie(dict):
    """
    A nested hashmap (i.e. a tree) storing counts at the nodes.
    All nodes except leaves are instances of this object, in a recursive manner.
    This can be either a standard (forward) or reversed trie, which may be faster for certain
    NLP tasks such as counting contexts of a token or backing off to a lower-order distribution.
    """
    __slots__ = ('total','distinct','max_depth','parent','order','_first','_second')
    def __init__(self,max_depth=2,order=1,parent=None):
        self.order = -1 if order < 0 else 1
        self._first = (self.order - 1)//2
        self._second = self._first+self.order
        self.total = 0
        self.distinct = 0
        self.max_depth = max_depth
        self.parent = parent
        
    def node(self,ngram):
        """
        ngram should be an indexable of keys.
        """
        if len(ngram) == 0:
            return self
        elif len(ngram) > self.max_depth:
            raise IndexError("ngram of length {} exceeds max_depth of {}".format(len(ngram),self.max_depth))
        return self._node(ngram)
        
    def _node(self,ngram):
        node = self
        for t in ngram[self._first::self.order]:
            try:
                newnode = node[t]
            except:
                return None
            else:
                node = newnode
        return node
    
    def count(self,ngram,distinct=False):
        if len(ngram) > self.max_depth:
            raise IndexError("ngram of length {} exceeds max_depth of {}".format(len(ngram),self.max_depth))
        
        node = self._node(ngram)
        if node:
            return node.total if not distinct else (1 if type(node) is FrequencyTrieLeaf else node.distinct)
        else:
            return 0
    
    def add(self,ngram,count=1):
        if len(ngram) == 0:
            return
        elif len(ngram) > self.max_depth:
            raise IndexError("ngram of length {} exceeds max_depth of {}".format(len(ngram),self.max_depth))
        # non-recursive version; messy but fast
        self._add(((ngram,count),))
    
    def addMany(self,ngrams,counts=repeat(1)):
        # warning: no length check is performed here!
        self._add(zip(ngrams,counts))
    
    addManyCounts = _add
    
    def addBagOfNgrams(self,bag):
        # warning: no check is done here!
        self._add(bag.items())
    
    def addTokens(self,tokens,n=None,start=None,end=None):
        if n is None:
            n = self.max_depth
        if n > self.max_depth:
            raise ArgumentError("trie of max_depth {} will not accomodate length-{} ngrams".format(self.max_depth,n))
        ngrams = ngramIter(tokens,n,start=start,end=end)
        self.addMany(ngrams)
        
    def _add(self,ngram_counts):
        # this is assumed safe (every ngram of actual length self.max_depth)
        # we add multiple ngrams in a loop to avoid the overhead of many function calls
        for (ngram,count) in ngram_counts:
            node = self
            new = False
            for d,t in zip(range(len(ngram) -1),ngram[self._first::self.order]):
                try:
                    newnode = node[t]
                except:
                    node[t] = FrequencyTrie(max_depth=self.max_depth-d-1,order=self.order,parent=node)
                    newnode = node[t]
                    new = True
                finally:
                    newnode.total += count
                    node = newnode
            if len(ngram) == self.max_depth:
                t = ngram[-1 - self._first]
                try:
                    newnode = node[t]
                except:
                    node[t] = FrequencyTrieLeaf(parent=node)
                    newnode = node[t]
                    new = True
                finally:
                    newnode.total += count
                    #node = newnode
            if new:
                node.incrementDistinct()
            
            self.total += count
    
    def incrementDistinct(self,count=1):
        self.distinct += count
        if self.parent:
            self.parent.incrementDistinct(count)
        
    def backoff(self,ngram):
        return (self.parent if self.order < 0 else self.node(ngram[self._second::self.order]))
    
    def contexts(self,ngram):
        if self.order < 0:
            node = self.node(ngram)
            return 0 if not node else node.distinct
        else:
            return sum(node._node(ngram) is not None for ng,node in self._nodes(self.max_depth - len(ngram)))
            
    def nodes(self,depth):
        if depth > self.max_depth:
            raise AttributeError("no nodes of depth {} in trie of max_depth {}".format(depth,self.max_depth))
        else:
            for ngram,node in self._nodes(depth):
                yield ngram,node
    
    def _nodes(self,depth,address=()):
        if depth == 1:
            for key,node in self.items():
                yield (address+(key,))[::self.order], node
        else:
            for key,node in self.items():
                subaddress = (address+(key,))
                for ngram,node in node._nodes(depth-1,subaddress):
                    yield ngram, node
    
    def leaves(self):
        return self.nodes(self.max_depth)
    
    def ngramCounts(self,n=None):
        if n is None:
            n = self.max_depth
        for ngram,leaf in self.nodes(n):
            yield ngram,leaf.total
        
    def __iadd__(self,trie):
        if trie.max_depth > self.max_depth:
            raise ArgumentError("trie of max_depth {} will not accomodate length-{} ngrams".format(self.max_depth,trie.max_depth))
        
        self._add(trie.ngramCounts())
        return self
    
    def __contains__(self,ngram):
        return self._node(ngram) is not None
    
    def memSize(self):
        return getsize(self)
    

class FrequencyTrieLeaf:
    __slots__=('total','parent','max_depth','distinct')
    def __init__(self,parent=None):
        self.total = 0
        self.distinct = 1
        self.max_depth = 0
        self.parent = parent
        
    def add(self,ngram=()):
        if len(ngram) > 0:
            raise IndexError("ngram of length {} exceeds max_depth of 0".format(len(ngram)))
        self.total += 1
        
    def count(self,ngram=()):
        if len(ngram) > 0:
            raise IndexError("ngram of length {} exceeds max_depth of 0".format(len(ngram)))
        return self.total
    
    def node(self,ngram=()):
        if len(ngram) > 0:
            raise IndexError("ngram of length {} exceeds max_depth of 0".format(len(ngram)))
        return self

