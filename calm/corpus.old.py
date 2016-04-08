from .processor import Processor
from .functions import *
from .objects import *
from operator import itemgetter

##################################################
# CORPUS, VOCABULARY, AND DOCUMENT ###############
##################################################


class Document:
    """
    Just a simple container for doc attributes in class form.
    Really just a dict with dot notation for indexing.
    Initialized from a dict, with a specified set of keys.
    """
    # TODO: implement some kind of wrapper around this that defines __slots__ dynamically
    # to save space on the dict while still giving flexibility of attributes at init time
    def __init__(self,record,attributes=None):
        if not attributes:
            if hasattr(record,"keys"):
                attributes = record.keys()
            if type(record) in {list,tuple,set}:
                attributes = range(len(record))
            else:
                raise ValueError("Unsupported document data structure: {}".format(type(record)))
        for attribute in attributes:
            if attribute in record:
                # add the record attribute as a class attribute of Document
                self.__dict__[attribute] = record[attribute]
            else:
                # if absent, place a null value to avoid later attribute errors
                self.__dict__[attribute] = None
    
    # this allows for dict-like access to features in addition to dot notation
    def __getitem__(self,key):
        return self.__dict__[key]


class Vocabulary:
    """
    A hashtable mapping ngrams to IDs and a reverse hash mapping IDs to ngrams 
    (helpful for generating random text from a language model),
    together with total size, ID generator, and update methods
    """
    def __init__(self):
        # hash unigram --> ID
        self.ID = dict()
        # hash ID --> unigram
        self.token = dict()
        
        self.size = 0
        self.__maxID = -1
        
    def add(self,tokenlist):
        for token in tokenlist:
            if token not in self.ID:
                # increment the maxID and vocabSize
                self.__maxID += 1
                self.size += 1
                # set both mappings
                self.ID[token] = self.__maxID
                self.token[self.__maxID] = token
     
    def drop(self,tokenlist):
        # This can be used to prune a list of stopwords from the vocab.
        # It will in general be more time-efficient to do this after corpus processing,
        # rather than for each document as it comes in.
        # Also allows removing custom lists generated from, e.g. DF statistics.
        for token in tokenlist:
            if token in self.ID:
                # get the ID
                ID = self.ID[token]
                # remove from both hashmaps
                del self.ID[token]
                del self.token[ID]
                # decrement the count
                self.size -= 1
                # if that was the maxID, get a new one
                if ID == self.__maxID:
                    self.__maxID = max(self.token.keys())
                    
                                        

class BagOfWordsCorpus:
    """
    a hashtable mapping docID's to docs.
    docs have a bag-of-words representation, along with configurable attributes: author, date, location, etc.
    processor is of class Processor
    TTF and DF are dicts of total term frequencies and document frequencies respectively.
    """
    def __init__(self,processor,docAttributes,textAttribute,IDAttribute=None,DF=True,TTF=True):
        # the hastable of docs; initialize empty
        self.docs = dict()
        self.docCount = 0
        
        # the string processor/tokenizer
        self.processor = processor

        # these are the fields that each doc should have (list)
        self.docAttributes = docAttributes
        # this is the field where the raw text is found in incoming records (string)
        self.__textAttribute = textAttribute
        # this is the field where the document ID is found in incoming records. If None, an integer index is used
        self.__IDAttribute = IDAttribute
        # The vocab of the corpus; initialize empty
        self.vocab = Vocabulary()
        
        max_n = max(processor.n)

        # Hashtable to keep track of TTF; initialize empty
        if TTF:
            #self.TTF = BagOfJoinedNgrams(max_n) if self.processor.joinChar else BagOfNgrams(max_n)
            self.TTF = BagOfWords()
        # Hashtable to keep track of TTF; initialize empty
        if DF:
            #self.DF = BagOfJoinedNgrams(max_n) if self.processor.joinChar else BagOfNgrams(max_n)
            self.DF = BagOfWords()
        # Token count for the whole corpus; useful for probability estimates
        self.totalTokens = 0
        
    def addDoc(self,record):
        # record is assumed to be a dict, list, or tuple
        # Initialize a new doc from the record:
        newDoc = Document(record,self.docAttributes)
        # Get an ID
        if self.__IDAttribute:
            docID = record[self.__IDAttribute]
        else:
            docID = self.docCount
        # Now add the appropriately processed and tokenized text:
        
        bagOfWords = self.processor.bagOfWords(record[self.__textAttribute])
        
        # add the unique tokens to the vocabulary, generating ID's for them
        self.vocab.add(bagOfWords.keys())
        
        # convert tokens to ID's now for space efficency, and update TTF and DF
        bagOfIDs = dict()
        # consume as you go with pop()
        while bagOfWords:
            ngram, count = bagOfWords.popitem()
            if ngram in self.vocab.ID:
                ID = self.vocab.ID[ngram]
                bagOfIDs[ID] = count

                # add count to the TTF and DF
                self.TTF.add(ID,count)
                self.DF.add(ID,1)
            
        # store the bag of words in efficient ID form
        newDoc.__setattr__('bagOfWords',bagOfIDs)
        # and finally store the document record
        self.docs[docID] = newDoc
        self.docCount += 1

    def bagOfIDs(self,bagOfWords):
        # this returns a bag of IDs *only for ngrams/tokens in the dictionary*; no updates
        bagOfIDs = dict()
        while bagOfWords:
            ngram, count = bagOfWords.popitem()
            if ngram in self.vocab.ID:
                ID = self.vocab.ID[ngram]
                bagOfIDs[ID] = count

        return bagOfIDs

    def cosine(self,docID,bagOfIDs):
        vector = self[docID].bagOfWords
        
        return(cosineSimilarity(vector,bagOfIDs,self.DF,self.docCount))

    def query(self,string,n):
        bagOfIDs = self.bagOfIDs(self.processor.bagOfWords(string))
        sims = [(docID,self.cosine(docID,bagOfIDs)) for docID in self.docs]
        sims = sorted(sims,key=itemgetter(1),reverse=True)
        return sims[0:n]
    
    # Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs
    def lowDFTerms(self,atMost=None, bottomN=None):
        if atMost:
            tokens = [self.vocab.token[ID] for ID in self.DF if self.DF[ID] <= atMost]
        elif bottomN:
            sortedDF = sorted(self.DF.items(), key=itemgetter(1), reverse=False)
            IDs = next(zip(*sortedDF[0:bottomN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
        
    # Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs
    def highDFTerms(self,atLeast=None, topN=None):
        if atLeast:
            tokens = [self.vocab.token[ID] for ID in self.DF if self.DF[ID] >= atLeast]
        elif topN:
            sortedDF = sorted(self.DF.items(), key=itemgetter(1), reverse=True)
            IDs = next(zip(*sortedDF[0:topN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
    
        # Select rare terms by DF, either those occurring in at most atMost docs, or the bottom bottomN docs
    def lowTTFTerms(self,atMost=None, bottomN=None):
        if atMost:
            tokens = [self.vocab.token[ID] for ID in self.TTF if self.TTF[ID] <= atMost]
        elif bottomN:
            sortedDF = sorted(self.TTF.items(), key=itemgetter(1), reverse=False)
            IDs = next(zip(*sortedDF[0:bottomN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens
        
    # Select common terms by DF, either those occurring in at least atLeast docs, or the top topN docs
    def highTTFTerms(self,atLeast=None, topN=None):
        if atLeast:
            tokens = [self.vocab.token[ID] for ID in self.TTF if self.TTF[ID] >= atLeast]
        elif topN:
            sortedDF = sorted(self.TTF.items(), key=itemgetter(1), reverse=True)
            IDs = next(zip(*sortedDF[0:topN]))
            tokens = [self.vocab.token[ID] for ID in IDs]
        return tokens

    
    # remove an iterable of ngrams from the corpus, including each document's bagOfWords if indicated
    def removeNgrams(self,ngrams,docs=False):
        # get the ngram IDS from the vocab for dropping them in all the other structures
        ngramIDs = [self.vocab.ID[ngram] for ngram in ngrams]
        self.vocab.drop(ngrams)
        self.DF.drop(ngramIDs)
        self.TTF.drop(ngramIDs)
        
        if docs == True:
            for doc in self.docs:
                self.docs[doc].bagOfWords.drop(ngramIDs)
        
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
        return (docID in self.docs)

    # allow for iteration as in a query, e.g. [similarity(doc) for doc in self]
    def __iter__(self):
        return iter(self.docs)
            
    # Allows access to the dictionary (hashtable) method keys()
    def keys(self):
        return self.docs.keys()
        
