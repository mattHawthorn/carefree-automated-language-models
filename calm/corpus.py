#coding:utf-8
from .processor import Processor
from .functions import *
from .objects import *
from .utils import getsize
from operator import itemgetter
from numpy import uint8,uint16,uint32,uint64,array
#from collections import namedtuple

COMPRESSED_TOKEN_TYPES = (uint8,uint16,uint32,uint64)


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
     
    def drop(self,tokens):
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
    def __init__(self,processor,textAttribute=None,IDAttribute=None,docAttributes=None,keepText=False,keepTokens=False,keepNgrams=False,compress=True,keyDtype=uint32):
        # the hastable of docs; initialize empty
        if keyDtype is not None and keyDtype not in COMPRESSED_TOKEN_TYPES:
            raise ArgumentError("Unsupported type {} for compressed tokens.  Must be one of {}".format(keyDtype,COMPRESSED_TOKEN_TYPES))
        
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
        
        self.keepNgrams = keepNgrams
        if keepNgrams and max(self.processor.n) > 1:
            self._bowFunc = self.processor.BagOfNgrams
        else:
            if keepNgrams:
                print("Warning: max processor n is 1 but keepNgrams=True; only unigrams will be kept")
            def bowFunc(tokens):
                return self.processor.BagOfWords(tokens,self.keepNgrams)
            self._bowFunc = bowFunc
            self.totalNgrams = tuple([0]*(max(self.processor.n) + 1))
            
        self.keyDtype = keyDtype
        self.compress = compress
        
        # Hashtable to keep track of TTF; initialize empty
        self.TTF = BagOfWords()
        # Hashtable to keep track of TTF; initialize empty
        self.DF = BagOfWords()
            
        # Token count for the whole corpus; useful for probability estimates
        self.totalTokens = 0
        
        
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
            for i,bag in enumerate(bagOfIDs):
                self.totalNgrams[i] += bag.total
        
        # store the tokens if specified; for training ngram models
        if self.keepTokens:
            if self.compress:
                tokens = self.compressTokens(tokens)
            newDoc.tokens = tokens
        
        # Get an ID
        if self.IDAttribute:
            docID = record[self.IDAttribute]
        else:
            docID = self.docCount
        
        # and finally store the document record in the corpus
        self.docs[docID] = newDoc
        self.docCount += 1

    def bagOfIDs(self,bagOfWords):
        # this returns a bag of IDs *only for ngrams/tokens in the dictionary*; no updates
        if not self.keepNgrams:
            bagOfIDs = BagOfWords()
            # consume with popitem()
            while bagOfWords:
                ngram, count = bagOfWords.popitem()
                if ngram in self.vocab.ID:
                    ID = self.vocab.ID[ngram]
                    bagOfIDs[ID] = count
        else:
            for i,bow in enumerate(bagOfWords.ngrams):
                bow2 = {self.vocab.ID[token]:count for token,count in bow.items()}
                bagOfWords.ngrams[i] = bow2
                bagOfIDs = bagOfWords
        
        return bagOfIDs
        
    def compressTokens(self,tokens):
        # for speed and lack of a safe missing value default, assumes all tokens are in the vocab, 
        # which is satisfied upon doc ingest
        return array([self.vocab.ID[token] for token in tokens], dtype=self.keyDtype)
        
    def getTokens(self,ids=None):
        return [self.vocab.token[i] for i in map(int,ids)]
    
    def tfidf(self,docID,normalize=True):
        return tfidfVector(self[docID].bagOfWords,DF=self.DF,docCount=self.docCount,dfweighting=IDF,tfweighting = None,normalize=normalize)
    
    def cosine(self,docID,bagOfIDs):
        vector = self[docID].bagOfWords
        
        return cosineSimilarity(vector,bagOfIDs,DF=self.DF,docCount=self.docCount,
                                dfweighting=IDF,tfweighting=sublinearTF)
    
    def query(self,string,n):
        bagOfIDs = self.bagOfIDs(self.processor.bagOfWords(string))
        sims = [(docID,self.cosine(docID,bagOfIDs)) for docID in self.docs]
        sims = sorted(sims,key=itemgetter(1),reverse=True)
        return sims[0:n]
    
    
    # Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs
    def lowDFTerms(self,atMost=None, bottomN=None):
        if atMost:
            if float(atMost) < 1.0:
                atMost = int(atMost*len(self.docs))
            tokens = [self.vocab.token[ID] for ID in self.DF if self.DF[ID] <= atMost]
        elif bottomN:
            sortedDF = sorted(self.DF.items(), key=itemgetter(1), reverse=False)
            IDs = next(zip(*sortedDF[0:bottomN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
        
    # Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs
    def highDFTerms(self,atLeast=None, topN=None):
        if atLeast:
            if float(atLeast) < 1.0:
                atLeast = int(atLeast*len(self.docs)) + 1
            tokens = [self.vocab.token[ID] for ID in self.DF if self.DF[ID] >= atLeast]
        elif topN:
            sortedDF = sorted(self.DF.items(), key=itemgetter(1), reverse=True)
            IDs = next(zip(*sortedDF[0:topN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
    
        # Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs
    def lowTTFTerms(self,atMost=None, bottomN=None):
        if atMost:
            if float(atMost) < 1.0:
                    atMost = int(atMost*len(self.docs))
            tokens = [self.vocab.token[ID] for ID in self.TTF if self.TTF[ID] <= atMost]
        elif bottomN:
            sortedDF = sorted(self.TTF.items(), key=itemgetter(1), reverse=False)
            IDs = next(zip(*sortedDF[0:bottomN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
        
    # Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs
    def highTTFTerms(self,atLeast=None, topN=None):
        if atLeast:
            if float(atLeast) < 1.0:
                atLeast = int(atLeast*len(self.docs)) + 1
            tokens = [self.vocab.token[ID] for ID in self.TTF if self.TTF[ID] >= atLeast]
        elif topN:
            sortedDF = sorted(self.TTF.items(), key=itemgetter(1), reverse=True)
            IDs = next(zip(*sortedDF[0:topN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
    
    # remove an iterable of ngrams from the corpus, including each document's bagOfWords if indicated
    def removeTerms(self,terms,docs=True):
        # get the ngram IDS from the vocab for dropping them in all the other structures
        terms = set(terms)
        ngramIDs = [self.vocab.ID[term] for term in terms if term in self.vocab.ID]
        self.vocab.drop(terms)
        self.DF.drop(ngramIDs)
        self.TTF.drop(ngramIDs)
        
        if docs:
            if type(self.docs) is list:
                keys = range(0,len(self.docs))
            elif type(self.docs) is dict:
                keys = self.docs.keys()
            for key in keys:
                delkeys = set(self.docs[key].bagOfWords).difference(self.vocab.token)
                self.docs[key].bagOfWords.drop(delkeys)
        
        
    # Allows direct access to docs as Corpus[docID]
    def __getitem__(self,docID):
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

