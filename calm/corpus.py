#coding:utf-8
from .processor import Processor
from .tfidf import *
from .objects import *
from .compression import *
from .utils import substitute_keys
from copy import copy, deepcopy
from numpy import array, uint32
from .utils import getsize
from operator import itemgetter
#from collections import namedtuple


##################################################
# CORPUS, VOCABULARY, AND DOCUMENT ###############
##################################################

class Document:
    """
    Just a simple container for doc attributes in class form.
    Use list of (key,value) pairs for corpus-specific attributes and __slots__
    for generic properties to avoid overhead of creating 1000's of dicts.
    """
    __slots__=['bagOfWords','tokens','_len','keys','attributes','text','ID']
    
    def __init__(self,record,textAttribute=None,IDAttribute=None,docAttributes=None):
        if not docAttributes:
            if hasattr(record,"keys"):
                self.docAttributes = tuple(record.keys())
            elif type(record) in [list,tuple]:
                self.docAttributes = tuple(range(len(record)))
            else:
                raise ValueError("Unsupported document data structure: {}".format(type(record)))
        
        self.ID = (None if IDAttribute is None else record[IDAttribute])
        self.text = (None if textAttribute is None else record[textAttribute])
        
        # unpack attributes into a namedtuple (without the overhead of the namedtuple type)
        attributes = []
        for attr in docAttributes:
            try:
                attributes.append(record[attr])
            except:
                attributes.append(None)
        
        self._len = len(docAttributes)
        # we assign the passed in keys directly; these are then shared in memory across a corpus
        self.keys = docAttributes
        self.attributes = tuple(attributes)
        
        
    # this allows for dict-like access to corpus-specific features in addition to dot notation for the
    # standard features (BOW, text, ID, tokens)
    def __getitem__(self,key):
        i = 0
        for k in self.keys:
            if k == key:
                break
            i += 1
        if i == self._len:
            return None
        return self.attributes[i]
    
    def __setitem__(self,key,value):
        self.keys = tuple(self.keys) + (key,)
        self.attributes = tuple(self.attributes) + (value,)
        self._len += 1
    
    def items(self):
        return (tup for tup in zip(self.keys,self.attributes))
    
    def __hasattr__(self,key):
        return (self[key] is not None)
        
    def __contains__(self,key):
        return key in self.bagOfWords


class Vocabulary:
    """
    A hashtable mapping ngrams to int IDs and a reverse hash mapping IDs to ngrams.
    (helpful for generating random text from a language model),
    together with total size, ID generator, and update methods
    """
    def __init__(self):
        # hash unigram --> ID
        self.ID = dict()
        # hash ID --> unigram
        self.token = dict()
        
        self.size = 0
        self.maxID = -1
        
    def add(self,token):
        self.addMany((token,))
    
    def addMany(self,tokens):
        for token in tokens:
            if token not in self.ID:
                # increment the maxID and vocabSize
                self.maxID += 1
                self.size += 1
                # set both mappings
                self.ID[token] = self.maxID
                self.token[self.maxID] = token
     
    def dropMany(self,tokens):
        # This can be used to prune a list of stopwords from the vocab.
        # It will in general be more time-efficient to do this after corpus processing,
        # rather than for each document as it comes in.
        # Also allows removing custom lists generated from, e.g. DF statistics.
        for token in tokens:
            if token in self.ID:
                # get the ID
                ID = self.ID[token]
                # remove from both hashmaps
                del self.ID[token]
                del self.token[ID]
                # decrement the count
                self.size -= 1
                # if that was the maxID, get a new one
                if ID == self.maxID:
                    self.maxID = max(self.token.keys())
                    
    def drop(self,token):
        self.dropMany((token,))
                    
    def __len__(self):
        return len(self.ID)
                    
                                        

class BagOfWordsCorpus:
    """
    a hashtable mapping docID's to docs.
    docs have a bag-of-words representation, along with configurable attributes: author, date, location, etc.
    processor is of class calm.processor.Processor.
    TTF and DF are dicts of total term frequencies and document frequencies respectively.
    If you wish to train a language model on this corpus, set keepTokens=True.
    At this point you will likely prefer some kind of token compression, which is performed by default but may
        be adjusted to another type (larger or smaller numpy unsigned int types currently) if you suspect a 
        larger/smaller vocab.  Current default of uint32 is more than adequate for all NLP tasks.
    """
    def __init__(self,processor,textAttribute=None,IDAttribute=None,docAttributes=None,keepText=False,keepTokens=False,keepNgrams=False,compressTokens=True):
        # the hastable of docs; initialize empty
        self.docs = (list() if IDAttribute is None else dict())
        self.docCount = 0
        
        # the string processor/tokenizer
        self.processor = processor

        # these are the fields that each doc should have (list). These are shared across all docs 
        # (unless None, then all attributes are pulled from each doc as docs are added)
        self.docAttributes = docAttributes
        # this is the field where the raw text is found in incoming records (string)
        self.textAttribute = textAttribute
        # this is the field where the document ID is found in incoming records. If None, an integer index is used
        self.IDAttribute = IDAttribute
        # The vocab of the corpus; initialize empty
        self.vocab = Vocabulary()
        # which parts of a doc are stored
        self.keepText = keepText
        self.keepTokens = keepTokens
        self.compressTokens = compressTokens
        
        self.keepNgrams = keepNgrams
        if keepNgrams:
            if max(self.processor.n) <= 1:
                print("Warning: max processor n is 1 but keepNgrams=True; only unigrams will be kept as singleton tuples")
            self._bowFunc = self.processor.bagOfNgrams
            self.joinchar = self.processor.joinchar
            self.totalNgrams = tuple([0]*(max(self.processor.n) + 1))
            self.setNFunc()
        else:
            self._bowFunc = self.processor.bagOfWords
            
        # Hashtable to keep track of TTF; initialize empty
        self.TTF = BagOfWords()
        # Hashtable to keep track of TTF; initialize empty
        self.DF = BagOfWords()
            
        # Token count for the whole corpus; useful for probability estimates
        self.totalTokens = 0
    
    def setNFunc(self):
        if self.joinchar is None:
            self._nfunc = lambda i: len(self.vocab.token[i])
        else:
            self._nfunc = lambda i: self.vocab.token[i].count(self.joinchar)

    def addDoc(self,record):
        # record is assumed to be a dict, list, or tuple
        # Initialize a new doc from the record:
        newDoc = Document(record=record,textAttribute=(self.textAttribute if self.keepText else None),
                          IDAttribute=self.IDAttribute,docAttributes=self.docAttributes)
        
        tokens = self.processor.process(record[self.textAttribute])
        bagOfWords = self._bowFunc(tokens)
        
        # add the unique tokens to the vocabulary, generating ID's for them
        self.vocab.addMany(bagOfWords.keys())
        
        # Increment the corpus total term count
        self.totalTokens += len(tokens)
        
        # convert tokens to ID's now for space efficency, and update TTF and DF
        bagOfIDs = self.bagOfIDs(bagOfWords)
        
        # add count to the TTF and DF
        self.TTF.addBagOfWords(bagOfIDs)
        self.DF.addMany(bagOfIDs.keys())
        
        # store the bag of words in efficient ID form
        newDoc.bagOfWords = bagOfIDs
        
        if self.keepNgrams:
            for i,bag in enumerate(bagOfIDs.ngrams):
                self.totalNgrams[i] += bag.total
        
        # store the tokens if specified; for training ngram models
        if self.keepTokens:
            ids = self.getIDs(tokens)
            if self.compressTokens:
                ids = compressInts(ids)
            newDoc.tokens = ids
        
        # Get an ID
        if self.IDAttribute:
            docID = record[self.IDAttribute]
        else:
            docID = self.docCount
        
        # and finally store the document record in the corpus
        self.docs[docID] = newDoc
        self.docCount += 1
    
    def docIter(self, return_ids=False, transform=None):
        class CorpusDocIter:
            def __init__(self,corpus,return_ids=False, transform=None):
                self.docs = corpus.docs
                self.return_ids = return_ids
                self.transform = transform
                
                if type(self.docs) is list:
                    if return_ids:
                        self.enumerator = lambda docs: enumerate(docs)
                    else:
                        self.enumerator = lambda docs: iter(docs)
                elif type(self.docs) is dict:
                    if return_ids:
                        self.enumerator = lambda docs: iter(docs.items())
                    else:
                        self.enumerator = lambda docs: iter(docs.values())

            def __iter__(self):
                if self.transform is None:
                    self.iterator = self.enumerator(self.docs)
                else:
                    if self.return_ids:
                        self.iterator = map(lambda tup: (tup[0], self.transform(tup[1])), self.enumerator(self.docs))
                    else:
                        self.iterator = map(self.transform, self.enumerator(self.docs))
                return self

            def __next__(self):
                try:
                    return next(self.iterator)
                except StopIteration:
                    raise StopIteration
                    
        return CorpusDocIter(self, return_ids, transform)
    
    def bagOfIDs(self,bagOfWords):
        # this returns a bag of IDs *only for ngrams/tokens in the dictionary*; no updates
        if not self.keepNgrams:
            bagOfIDs = BagOfWords()
            tokenCounts = ((self.vocab.ID[token],count) for token,count in bagOfWords.items() if token in self.vocab.ID)
            bagOfIDs._addmanyCounts(tokenCounts)
        else:
            bagOfIDs = BagOfNgrams(max_n = bagOfWords.max_n, joinchar = None)
            # this ensures the bagofngrams knows where to put incoming id's if any are ever added
            bagOfIDs._nfunc = self._nfunc
            for i,bow in enumerate(bagOfWords.ngrams):
                bow2 = BagOfWords()
                bow2._addmanyCounts((self.vocab.ID[ngram], count) for ngram,count in bow.items() if ngram in self.vocab.ID)
                bagOfIDs.ngrams[i] = bow2
                bagOfIDs.total += bow2.total
        return bagOfIDs
        
    def getTokens(self,ids=None):
        return [self.vocab.token[i] for i in map(int,ids)]
        
    def getIDs(self,tokens):
        return array([self.vocab.ID[token] for token in tokens], dtype=uint32)
        
    def getDocTokens(self,docID):
        ids = self[docID].tokens
        if self.compressTokens:
            ids = decompressInts(ids)
        return self.getTokens(ids)
        
    def getDocTokenIDs(self,docID):
        ids = self[docID].tokens
        if self.compressTokens:
            ids = decompressInts(ids)
        return ids
        
    def ttf(self,token):
        return self.TTF[self.vocab.ID[token]]
        
    def df(self,token):
        return self.DF[self.vocab.ID[token]]
    
    def tfidf(self,docID,normalize=True):
        return tfidfVector(self[docID].bagOfWords,DF=self.DF,docCount=self.docCount,dfweighting=IDF,tfweighting = None,normalize=normalize)
    
    def cosine(self, docID, bagOfIDs, dfweighting=IDF, tfweighting=sublinearTF):
        vector = self[docID].bagOfWords
        
        return cosineSimilarity(vector, bagOfIDs, DF=self.DF, docCount=self.docCount,
                                dfweighting=dfweighting, tfweighting=tfweighting)
    
    def query(self, string, n, dfweighting=IDF, tfweighting=sublinearTF):
        bagOfIDs = self.bagOfIDs(self.processor.bagOfWords(string))
        sims = [(docID, self.cosine(docID,bagOfIDs,dfweighting,tfweighting)) for docID in self.docs]
        sims = sorted(sims, key=itemgetter(1),reverse=True)
        return sims[0:n]
    
    
    def lowDFTerms(self,atMost=None, bottomN=None):
        """Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs"""
        return self._extremeTerms(kind='df',limit=atMost,number=bottomN,how='low')
        
    def highDFTerms(self,atLeast=None, topN=None):
        """Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs"""
        return self._extremeTerms(kind='df',limit=atLeast,number=topN,how='high')
        
    def lowTTFTerms(self,atMost=None, bottomN=None):
        """Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs"""
        return self._extremeTerms(kind='ttf',limit=atMost,number=bottomN,how='low')
        
    def highTTFTerms(self,atLeast=None, topN=None):
        """Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs"""
        return self._extremeTerms(kind='ttf',limit=atLeast,number=topN,how='high')
        
    def _extremeTerms(self,kind='ttf',limit=None,number=None,how='high'):
        if kind=='ttf':
            total = self.totalTokens
            bow = self.TTF
        elif kind=='df':
            total = self.docCount
            bow = self.DF
        if limit is not None:
            if float(limit) < 1.0:
                limit = int(limit*total)
            else:
                limit = int(limit)
            if how=='high':
                compare = int.__gt__
            elif how=='low':
                compare = int.__le__
            tokens = [self.vocab.token[ID] for ID,count in bow.items() if compare(count,limit)]
        elif number is not None:
            if how=='high':
                reverse = True
            elif how=='low':
                reverse = False
            sorted_bow = sorted(bow.items(), key=itemgetter(1), reverse=reverse)
            tokens = [self.vocab.token[ID] for ID,count in sorted_bow[0:number]]
        return tokens
    
    def removeTerms(self,terms,docs=True,vocab=False):
        """remove an iterable of ngrams/tokens from the corpus, including each document's bagOfWords if indicated"""
        # get the ngram IDS from the vocab for dropping them in all the other structures
        terms = set(terms)
        ngramIDs = [self.vocab.ID[term] for term in terms if term in self.vocab.ID]
        
        if vocab:
            self.vocab.dropMany(terms)
            self.DF.dropMany(ngramIDs)
            self.TTF.dropMany(ngramIDs)
        
        if docs:
            if type(self.docs) is list:
                keys = range(0,len(self.docs))
            elif type(self.docs) is dict:
                keys = self.docs.keys()
            for key in keys:
                delkeys = set(self.docs[key].bagOfWords).difference(self.vocab.token)
                self.docs[key].bagOfWords.dropMany(delkeys)
    
    def reduceTokenIDs(self):
        """
        If tokens have been removed from the vocab, this will make sure their int ID's solidly fill in
        a range from 0 to len(vocab) - 1.  These are sorted descending by TTF, which should aid in
        compression somewhat.
        """
        oldTTF = self.TTF
        oldDF = self.DF
        oldVocab = self.vocab
        
        newOrder = sorted(oldTTF.items(), key=itemgetter(1), reverse=True)
        
        newVocab = Vocabulary()
        newVocab.addMany(oldVocab.token[i] for i,count in newOrder)
        
        oldToNew = {i:newVocab.ID[token] for i,token in oldVocab.token.items()}
        newTTF = BagOfWords()
        newTTF._addmanyCounts((oldToNew[i], count) for i,count in oldTTF.items())
        newDF = BagOfWords()
        newDF._addmanyCounts((oldToNew[i], count) for i,count in oldDF.items())
        
        self.TTF = newTTF
        self.DF = newDF
        self.vocab = newVocab
        
        if self.keepTokens:
            for docID, doc in self.docIter(return_ids=True):
                ids = [oldToNew[i] for i in self.getDocTokenIDs(docID)]
                if self.compressTokens:
                    ids = compressInts(ids)
                doc.tokens = ids
                
        if self.keepNgrams:
            self.setNFunc()
            for doc in self.docIter(return_ids=False):
                # really a bag of ngrams
                oldNgrams = doc.bagOfWords.ngrams
                # entry k stores counts for ngrams of length k for k > 0
                newNgrams = ((),) + tuple(BagOfWords() for i in range(len(oldNgrams) - 1))
                for i, bag in enumerate(newNgrams[1:]):
                    bag._addmanyCounts((oldToNew[i],count) for i,count in oldNgrams[i].items())
                doc.bagOfWords.ngrams = newNgrams
                doc.bagOfWords._nfunc = self._nfunc
        else:
            for doc in self.docIter(return_ids=False):
                newBag = BagOfWords()
                newBag._addmanyCounts((oldToNew[i], count) for i,count in doc.bagOfWords.items())
                doc.bagOfWords = newBag
        
    def __getitem__(self,docID):
        """Allows direct access to docs as Corpus[docID]"""
        try:
            doc = self.docs[docID]
        except:
            return None
        else:
            return doc
    
    # Allows access to the "key in item" syntax
    def __contains__(self,docID):
        if type(self.docs) is dict:
            return (docID in self.docs)
        else:
            return 0 <= docID < len(self.docs)

    # allow for iteration over doc contents
    def __iter__(self):
        if type(self.docs) is dict:
            return iter(self.docs.values())
        else:
            return enumerate(self.docs)
            
    # Allows access to the dictionary method keys()
    def keys(self):
        if type(self.docs) is dict:
            return self.docs.keys()
        else:
            return range(len(self.docs))
            
    def items(self):
        if type(self.docs) is dict:
            return self.docs.items()
        else:
            return enumerate(self.docs)

