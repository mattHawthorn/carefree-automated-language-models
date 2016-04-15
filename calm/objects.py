
########################################
# BAGS OF WORDS AND NGRAMS #############
########################################


class BagOfWords(dict):
    # inherits from dict (hashmap), with added drop and add methods as well as counters.
    # useful not only for document representations but also for corpus-wide term frequency dictionaries (TTF/DF)
    __slots__=()
    def __init__(self,ngrams=None):
        if ngrams:
            [self.add(ngram) for ngram in ngrams]

    def add(self,token, count=1):
        if token not in self:
            self[token] = count
        else:
            self[token] += count

    def addList(self,tokens,counts=None):
        if not counts:
            for token in tokens:
                if token not in self:
                    self[token] = 1
                else:
                    self[token] += 1
        else:
            for token, count in zip(tokens,counts):
                if token not in self:
                    self[token] = count
                else:
                    self[token] += count
        
    def addBagOfWords(self,bagOfWords,count=None):
        # note that the syntax here allows bagOfWords to be any indexed object with numeric values;
        # it need not be a BagOfWords object.
        # Use count = 1 for a DF dictionary
        if not count:
            for ngram in bagOfWords:
                self.add(ngram,bagOfWords[ngram])
        else:
            for ngram in bagOfWords:
                self.add(ngram,count)
            
    def drop(self,tokens):
        # assume a single ngram/token in these cases
        if type(tokens) in {str,int,tuple}:
            if token in self:
                del self[token]
            return
        
        # else a list or other iterable of ngrmas/tokens
        for token in tokens:
            if token not in self:
                continue
            del self[token]
            

class BagOfJoinedNgrams:
    """
    simply a list of BagOfWords objects containing counts only for ngrams of length the index.
    indexing methods defined to take this into account
    this particular implementation assumes that ngrams take the form ("joined_ngram_string",n)
    """
    __slots__=['max_n','ngrams']
    def __init__(self,max_n):
        self.max_n = max_n
        self.ngrams = [None]
        for i in range(max_n):
            self.ngrams.append(BagOfWords())
                
    def __getitem__(self,ngram):
        try:
            return self.ngrams[ngram[1]][ngram[0]]
        except:
            return 0

    def __setitem__(self,ngram,count):
        self.ngrams[ngram[1]][ngram[0]] = count

    def __contains__(self,ngram):
        return ngram[0] in self.ngrams[ngram[1]]
        
    def add(self,ngram,count=1):
        if ngram[0] in self.ngrams[ngram[1]]:
            self.ngrams[ngram[1]][ngram[0]] += count
        else:
            self.ngrams[ngram[1]][ngram[0]] = count
        
    def addList(self,ngrams,counts=None):
        if not counts:
            for ngram in ngrams:
                if ngram not in self:
                    self[ngram] = 1
                else:
                    self[ngram] += 1
        else:
            for ngram, count in zip(ngrams,counts):
                if ngram not in self:
                    self[ngram] = count
                else:
                    self[ngram] += count
                    
    def addBagOfWords(self,bagOfWords):
        # note that the syntax here allows bagOfWords to be any indexed object with numeric values;
        # it need not be a BagOfWords object.
        for token in bagOfWords:
            self.add(token,bagOfWords[ngram])
    
    def drop(self,ngram):
        if ngram in self:
            del self.ngrams[ngram[1]][ngram[0]]
            
    def dropList(self,ngrams):
        for ngram in ngrams:
            self.drop(ngram)

class BagOfNgrams:
    """
    simply a list of BagOfWords objects containing counts only for ngrams of length the index.
    indexing methods defined to take this into account
    this particular implementation assumes that ngrams take the form ("token_1","token_2",...,"token_n")
    """
    __slots__=['max_n','ngrams']
    def __init__(self,max_n):
        self.max_n = max_n
        self.ngrams = [None]
        for i in range(max_n):
            self.ngrams.append(BagOfWords())
                
    def __getitem__(self,ngram):
        try:
            return self.ngrams[len(ngram)][ngram]
        except:
            return 0

    def __setitem__(self,ngram,count):
        self.ngrams[len(ngram)][ngram] = count

    def __contains__(self,ngram):
        return ngram in self.ngrams[len(ngram)]
        
    def add(self,ngram,count=1):
        if ngram in self.ngrams[len(ngram)]:
            self.ngrams[len(ngram)][ngram] += count
        else:
            self.ngrams[len(ngram)][ngram] = count
        
    def addList(self,ngrams,counts=None):
        if not counts:
            for ngram in ngrams:
                if ngram not in self:
                    self[ngram] = 1
                else:
                    self[ngram] += 1
        else:
            for ngram, count in zip(ngrams,counts):
                if ngram not in self:
                    self[ngram] = count
                else:
                    self[ngram] += count
            
    def addBagOfWords(self,bagOfWords):
        # note that the syntax here allows bagOfWords to be any indexed object with numeric values;
        # it need not be a BagOfWords object.
        for token in bagOfWords:
            self.add(token,bagOfWords[ngram])
            
    def drop(self,ngram):
        if ngram in self:
            del self.ngrams[len(ngram)][ngram]
            
    def dropList(self,ngrams):
        for ngram in ngrams:
            self.drop(ngram)
            
            
class NgramIter:
    # Iterator to return a sequence of ngrams from an iterable of tokens
    def __init__(self,tokens,n):
        self.i = 0
        self.n = n
        self.maxIndex = len(tokens) - n
        self.tokens = tokens
        
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i > self.maxIndex:
            raise StopIteration
        
        ngram = self.tokens[self.i:(self.i + self.n)]
        self.i += 1
        
        return tuple(ngram)

