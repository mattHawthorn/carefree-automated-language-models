#coding: utf-8from numpy import arrayimport numpy as npimport osimport randomimport sysfrom math import log, expfrom random import random, sample                class FreqTrieNode(dict):    # node in a frequencie trie.  Hashes token ID's to child nodes.    def __init__(self, depth=0):        # total token count        self.count = 0        # total unique tokens        self.unique = 0        # parent node        self.parent = None        # token ID hashing to this node for use in determining prefixes        self.address = None        # depth (for potential future use with depth-sensitive smoothing; default 0)        self.depth = depth        # probability for use in language model; null until set by model        self.p = None        # generative model: list of token IDs and corresponding transition probabilities;        # null until set by language model        self.model = None            def add(self,token,count=1):        # if the token is already hashed        if token in self:            # add token count to the count of the child node            self[token].count += count        else:            # otherwise, establish a new child node at the token hash            self[token] = FreqTrieNode(depth=self.depth + 1)            # establish the child's parent as self            self[token].parent = self            # store the ID hashing to this node for future use in determining prefixes            self[token].address = token            # augment the child's count by count            self[token].count += count            # and increment the unique count            self.unique += 1        # if root, add count; no one else will!        if not self.parent:            self.count += count                # get counts of children    def counts(self):        counts = [(ID,self[ID].count) for ID in self]        return counts    # get probabilities of children, given self    def probs(self):        probs = [(ID,self[ID].p) for ID in self]        return probs            # suffix for this node; the (n-1)-gram suffixing the n-gram addressing this node    def suffix(self):        # initialize empty list        length = self.depth - 1        suffix = [0]*(length)                node = self                for i in range(0,length):            suffix[i] = node.address            node = node.parent                suffix.reverse()        return suffix            def node(self,IDs):        node = self        for ID in IDs:            node = node[ID]        return node                                class FreqTrie:    # A trie made up of FreqTrieNodes, with methods to intuitively access the nodes corresponding to n-grams    def __init__(self,n,vocab=None):        self.n = n        self.root = FreqTrieNode()                if vocab:            self.vocab = vocab        else:            self.vocab = Vocabulary()            # add an n-gram all the way down the trie    def add(self,ngram,count=1):        # but not if the n-gram exceeds n in length; assume an error and pass        if len(ngram) > self.n:            raise ValueError("This trie only accomodates ngrams of length {0} or less".format(self.n))                # start at the root        node = self.root                for token in ngram:            # add the token to the vocab in case it's not there            self.vocab.add([token])            # get the ID of the token            ID = self.vocab.ID[token]            # augment the node child counts with the ngram count            node.add(ID,count)            # descend to the child            node = node[ID]                def counts(self,ngram):        node = self[ngram]        counts = node.counts()        counts = [(self.vocab.token[x[0]],x[1]) for x in counts]                return counts        # method for intuitive indexing by ngrams to get their counts; appropriate-length token iterables (tuples, lists etc)    def __getitem__(self,ngram):        node = self.root        if len(ngram) > self.n:            raise ValueError("This trie only accomodates ngrams of length {0} or less".format(self.n))                for token in ngram:            try:                ID = self.vocab.ID[token]            except:                return None            try:                node = node[ID]            except:                return None                    return node                class NodeProbModel:    """    Probability model for a single node of an Ngram language model probability trie.    No local information regarding smoothing is stored here, but rather in the parent model, which passes that information for    initialization.    Has a list of cumsum'ed probabilities for fast sampling via binary search, and a correspoding list of token IDs to sample from.    Also keeps track of reserved probability for interpolation (in the case of linear and absolute discount) or unseen tokens     (in the case of additive smoothing)    """        def __init__(self,freqTrieNode,V,smoothing,param,keyDtype='uint32'):        # total unique tokens seen at the node        unique = freqTrieNode.unique        # total token count seen at the node        count = freqTrieNode.count        # total unique unseen tokens at the node        unseen = V - unique                # ID's of all tokens seen at the node stored as an array (uint32 used to save space over int64 default,         # 16 will do for most corpora)        IDs = np.array(list(freqTrieNode.keys()),dtype=keyDtype)        # stored as an easily searchable (sorted) array for purposes of lookup        IDs = np.sort(IDs)                # intialize probabilities as counts        probs = np.array([freqTrieNode[int(ID)].count for ID in IDs],dtype='float32')                # apply smoothing depending on type        if smoothing == 'additive':            # total smoothed count of tokens            total = count + param*V            # add the constant delta to the counts and normalize            probs = (probs + param)/total            # probability mass reserved for unseen tokens            self.reserved = param*unseen/total            if unseen == 0:                self.unseen = 0            else:                self.unseen = self.reserved/unseen                    elif smoothing == 'linear interpolation':            # simply take the ML estimate, multiplied by lambda            probs = probs*param/count            # then set the reserved probability to 1-lambda            self.reserved = 1 - param                    elif smoothing == 'absolute discount':            # subtract the constant delta from the counts and normalize            probs = (probs - param)/V            # the probability mass reserved for unseen tokens            self.reserved = param*unique/V                    # store the probabilities and IDs        self.probs = probs        self.ID = IDs            def sample(self):        # take the cumsum for sampling purposes; this can be binary-searched for efficient sampling        probs = np.cumsum(self.probs)                # sample from Unif(0,1)        rand = random()        # get the index of the entry just greater than rand in the cumsum probs array using binary search        index = binarySearch(rand,probs)                # if this returned an index, then get the ID of the token        if index:            ID = self.ID[index]        # otherwise, rand was greater than the greatest cumsum prob; this occurs with a probability of self.reserved        # I return None and defer to the parent trie for the appropriate sampling procedure, which depends on the        # global vocab and the smoothing type        else:            ID = None                    return IDclass NgramModel:    """    Ngram language model.  Modelled as a trie with transition probabilities at each node.    Methods for computation of probability and perplexity of a token sequence, as well as    random text generation.  Training is assumed to be in batch mode on a complete previously    generated frequency trie.  In this way, probability calculations and sampling can be    performed quickly.    """        def __init__(self,n,smoothing='additive',param=1,newTokens=1,keyDtype='uint32'):        """        Smoothing is one of: {"absolute discount","linear interpolation","additive"}        param is the single parameter supplied to each of these methods.  Traditionally called delta in the additive         and discount cases, and lambda in the linear case.  lambda is a keyword in python, so I choose simply 'param'         to capture all cases.        newTokens is a number which allows allocation of probability mass to tokens not in the vocab, i.e. for        calculating perplexity on documents outside the training set.  This is added to the vocab size self.V, and         propagates to all computed probabilites in interpolation calculations.        """                # smoothing type        self.smoothing = smoothing        # the smoothing parameter        self.param = param        # the numpy datatype of the token keys, for use in the generative models.          # This can save considerable space over the standard 64-bit int; an unsigned 16-bit         # int can for example index a vocublary of size 65535 or less.        self.key_dtype=keyDtype        # empty root node        self.root = None        self.vocab = None        self.V = None        self.n = n        self.new = newTokens            def train(self,freqTrie):        # freqTrie is of the FreqTrie class and is appended at the root node of the probability        # model.  nodes are recursively assigned probabilities and generative models.                if freqTrie.n < self.n:            raise ValueError("This model must be trained on a trie of depth {0}.".format(self.n))            return                # the vocab        self.vocab = freqTrie.vocab        self.V = self.vocab.size + self.new        # establish the root node        self.root = freqTrie.root        # build the sampling models recursively on the root node        # these do not take interpolation into account, but allows sampling random         # sequences of tokens via tracking of reserved probability mass for         # unseen tokens or allocation to the suffix (n-1)-gram in interpolation.        # self.__conditional(ngram) is computed on the fly and takes interpolation into account        self.__makeModel(self.root)                    def __makeModel(self,node):        # make the sampling model for the node        node.model = NodeProbModel(freqTrieNode=node,V=self.vocab.size,smoothing=self.smoothing,                                   param=self.param,keyDtype=self.key_dtype)        # this model has a 'reserved' probability, which in the case of absolute discount or        # linear interpolation, is the probability mass given to the distribution on the prefix        # node, i.e. p(w3|w1w2) = p(w3 at node w1w2 from counts at node w1w2) + reserved*p(w2|w1).        # in the case of additive smoothing, this is just the probability distributed evenly over        # unseen tokens.                for child in node:            self.__makeModel(node=node[child])                                def p(self,tokens,logarithm=False):        # return probability of an iterable of tokens        # use the log-sum-exp to avoid overflow, optionally returning log-probability                tokens = self.IDs(tokens)        logp = 0                # first, the log probability of the head of the list (1-gram, 2-gram, ... (n-1)-gram)        l = min(self.n - 1, len(tokens))                for i in range(1,l+1):            ngram = tokens[0:i]            logp += log(self.__conditional(ngram))                # now, the rest of the tokens        if len(tokens) >= self.n:            for ngram in NgramIter(tokens,self.n):                logp += log(self.__conditional(ngram))                        if logarithm:            return logp        else:            return exp(logp)        def perplexity(self,tokens,logarithm=False):        # get the perplexity of a sequence of tokens        logp = self.p(tokens,logarithm=True)        logp = -1*logp/len(tokens)                if logarithm:            return logp        else:            return(exp(logp))            def conditional(self,ngram):        # for the user interface; can give raw tokens rather than their int IDs        IDs = self.IDs(ngram)        return self.__conditional(IDs)            def __conditional(self,ngram):        # __conditional probability of the last word in an ngram given the prior word        # all tokens are assumed to be their int ID's here.                # token we want conditional p for is the last one:        token = ngram[-1]                # handle the edge case: 1 token        # we could have handled only the 0-gram case, but we've allocated some mass        # to new out-of-vocab tokens, so we must know what the token is to determine        # how much is allocated        if len(ngram) == 1:            if token in self.root:                if self.smoothing == 'additive':                    p = (self.root[token].count + self.param)/(self.root.count + self.param*self.V)                elif self.smoothing == 'linear interpolation':                    p = self.param*self.root[token].count/self.root.count + (1-self.param)/self.V                elif self.smoothing == 'absolute discount':                    reserved = self.param*self.root.unique/(self.root.count)                    p = (self.root[token].count - self.param)/self.root.count + reserved/self.V            elif token in self.vocab.token:                if self.smoothing == 'additive':                    p = self.param/(self.root.count + self.param*self.V)                elif self.smoothing == 'linear interpolation':                    p = (1-self.param)/self.V                elif self.smoothing == 'absolute discount':                    reserved = (self.param*self.root.unique/self.root.count)                    p = reserved/self.V            else:                # token is new to the vocab; allocate self.new rather than 1                if self.smoothing == 'additive':                    p = self.param*self.new/(self.root.count + self.param*self.V)                elif self.smoothing == 'linear interpolation':                    p = (1-self.param)*self.new/self.V                elif self.smoothing == 'absolute discount':                    reserved = (self.param*self.root.unique/self.root.count)                    p = reserved*self.new/self.V                        return p                if self.smoothing == 'additive':            # try to get the the node corresponding to the (n-1)-gram            try:                node = self.node(ngram[0:-1])            # if not; the ngram wasn't seen; under additive smoothing            # this would imply a probability of 1/V for all tokens from the             # first unseen token down (or self.new/V for an out-of-vocab new             # token; assign this to p            except:                if token in self.vocab.token:                    p = 1/self.V                else:                    p = self.new/self.V            else:                # if the (n-1)-gram was found, get the smoothed MLE of token there                # param is delta here                if token in node:                    p = (node[token].count + self.param)/(self.V*self.param + node.count)                # otherwise, count was 0; check if it's in the vocab                elif token in self.vocab.token:                    # and assign the p you would get from a 0 count                    p = self.param/(self.V*self.param + node.count)                else:                    # token is new, not in the vocab; use the reserved mass for new tokens                    p = self.param*self.new/(self.V*self.param + node.count)        elif self.smoothing in 'linear interpolation':            # try to get the the node corresponding to the (n-1)-gram            try:                node = self.node(ngram[0:-1])            # if an n-gram hasn't been seen, then there is no information on which to            # estimate an MLE beyond the first unseen word.  Interpreting the math             # strictly, the MLE for every word would be 0, but this is not a pdf.            # Strictly speaking the MLE's would all be equal so we could just set them            # to 1/V, but I assume we get better results by allocating all the mass to             # the suffix interpolation, i.e. just let lambda = 1, which is the other             # obvious solution to normalization.  In this way we better leverage prior            # knowledge for estimation.            except:                p = self.__conditional(ngram[1:len(ngram)])            else:                # get the mle and mix with the (n-1)-gram suffix's mle linearly.                # this will recurse up as far as needed, possibly to the root                if token in node:                   mle = node[token].count/node.count                else:                   mle = 0                # mix with the suffix (n-1)-gram.  param here is lambda                p = self.param*mle + (1-self.param)*self.__conditional(ngram[1:len(ngram)])        elif self.smoothing == 'absolute discount':            try:                node = self.node(ngram[0:-1])            except:                # ngram is unseen.                # same argument as above for linear interpolation                p = self.__conditional(ngram[1:len(ngram)])            else:                if token in node:                    discounted = (node[token].count - self.param)/node.count                else:                    discounted = 0                p = discounted + (node.unique*self.param)*self.__conditional(ngram[1:len(ngram)])/node.count                        return p                def node(self,IDs):        node = self.root        for ID in IDs:            node = node[ID]        return node        def __getitem__(self,ngram):        IDs = self.IDs(ngram)        if None in IDs:            raise KeyError("node does't exist")        else:            return self.node(IDs)        def topTokens(self,ngram,k=10):        node = self[ngram]        IDs = self.IDs(ngram)        probs = [(ID,self.__conditional(IDs + [ID])) for ID in node]        probs = sorted(probs,key= (lambda x: x[1]), reverse=True)        probs = probs[0:k]        topk = [(self.vocab.token[ID],p) for ID,p in probs]        return topk        def IDs(self,tokens):        # return a list of token IDs from a list of tokens (filling None where missing from the vocab)        length = len(tokens)        # initialize a list        IDs = [0]*length        # get the token IDs        for i in range(0,length):            try:                IDs[i] = self.vocab.ID[tokens[i]]            except KeyError:                IDs[i] = None                return IDs            def generate(self,maxLength,beginToken=None,endToken=None):        tokens = list()        length = 0                if beginToken:            prior = self.IDs([beginToken])        else:            prior = []                while length < maxLength:            # sample an ID            ID = None            newprior = prior                        while not ID:                try:                    node = self.node(newprior)                except:                    # prior ngram doesn't exist; in the case of additive, we would                    # sample from the vocab evenly:                    if self.smoothing == 'additive':                        ID = sample(list(self.vocab.token),1)[0]                        print(ID)                    # in the interpolation case, sample from the suffix node.                    # this happens with the correct probability assured by the                     # construction of the model.                    elif self.smoothing in ['absolute discount','linear interpolation']:                        if len(newprior) == 1:                            newprior == []                        else:                            newprior = newprior[1:len(newprior)]                else:                    ID = node.model.sample()                    if not ID:                        if self.smoothing == 'additive':                            # sample from the unseen tokens evenly in the additive case                            ID = sample(set(self.vocab.token).difference(set(node)),1)[0]                            print(ID)                            break                        if len(newprior) == 0:                            # we've sampled from probability not seen at the root;                            if self.smoothing in ['absolute discount','linear interpolation']:                                # sample evenly from the vocab in the interpolation case (0-gram distribution)                                ID = sample(list(self.vocab.token),1)[0]                                print(ID)                        elif len(newprior) == 1:                            newprior == []                        else:                            newprior = newprior[1:len(newprior)]                        token = self.vocab.token[ID]            tokens.append(ID)            l = min(self.n,len(tokens))            prior = tokens[-l:len(tokens)]            length += 1                        if endToken and token == endToken:                break                    tokens = [self.vocab.token[ID] for ID in tokens]        return tokens                def binarySearch(x,listlike):    # return the index of the first item in listlike which exceeds x. Returns None if not found.    # Could also do np.searchsorted(listlike,x,side='left'), and check the return against len(listlike)    # since this returns max index of listlike plus 1 when no item is found.    if len(listlike) == 0:        return None        low = 0    high = len(listlike) - 1    i = int((low + high)/2)        if x > listlike[-1]:        return None        while low != i and high != i:        if x < listlike[i]:            high = i        else:            low = i                i = int((low + high)/2)        if listlike[i] >= x:        return i    else:        return i+1        class NgramIter:    # Iterator to return a sequence of ngrams from an iterable of tokens    def __init__(self,tokens,n):        self.i = 0        self.n = n        self.maxIndex = len(tokens) - n        self.tokens = tokens            def __iter__(self):        return self        def __next__(self):        if self.i > self.maxIndex:            raise StopIteration                ngram = self.tokens[self.i:(self.i + self.n)]        self.i += 1                return tuple(ngram)