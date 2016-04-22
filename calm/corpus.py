from .processor import Processor
from .functions import *
from .objects import *
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
    __slots__=['bagOfWords','tokens','_len','attributes','text','ID']
    
    def __init__(self,record,textAttribute=None,IDAttribute=None,docAttributes=None):
        if not docAttributes:
            if hasattr(record,"keys"):
                docAttributes = record.keys()
            elif type(record) in [list,tuple]:
                docAttributes = range(len(record))
            else:
                raise ValueError("Unsupported document data structure: {}".format(type(record)))
        
        self.ID = None
        self.text = None
        if IDAttribute:
            self.ID = record[IDAttribute]
        if textAttribute:
            self.text = record[textAttribute]
            
        # unpack attributes into a namedtuple
        docAttributes = list(docAttributes)
        self._len = len(docAttributes)
        self.attributes = []
        for attr in docAttributes:
            try:
                self.attributes.append((attr,record[attr]))
            except (KeyError, IndexError) as e:
                self.attributes.append((attr,None))
        
        
    # this allows for dict-like access to corpus-specific features in addition to dot notation for the
    # standard features (BOW, text, ID, tokens)
    def __getitem__(self,key):
        i=0
        while self.attributes[i][0] != key:
            i += 1
            if i == self._len:
                return None
        return self.attributes[i][1]
        
    def __contains__(self,key):
        return key in self.bagOfWords


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
        self.maxID = -1
        
    def add(self,tokenlist):
        for token in tokenlist:
            if token not in self.ID:
                # increment the maxID and vocabSize
                self.maxID += 1
                self.size += 1
                # set both mappings
                self.ID[token] = self.maxID
                self.token[self.maxID] = token
     
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
                if ID == self.maxID:
                    self.maxID = max(self.token.keys())
                    
                                        

class BagOfWordsCorpus:
    """
    a hashtable mapping docID's to docs.
    docs have a bag-of-words representation, along with configurable attributes: author, date, location, etc.
    processor is of class Processor
    TTF and DF are dicts of total term frequencies and document frequencies respectively.
    """
    def __init__(self,processor,textAttribute=None,IDAttribute=None,docAttributes=None,keepText=False,keepTokens=False):
        # the hastable of docs; initialize empty
        self.docs = dict()
        self.docCount = 0
        
        # the string processor/tokenizer
        self.processor = processor

        # these are the fields that each doc should have (list)
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
        
        max_n = max(processor.n)

        # Hashtable to keep track of TTF; initialize empty
        #self.TTF = BagOfJoinedNgrams(max_n) if self.processor.joinChar else BagOfNgrams(max_n)
        self.TTF = BagOfWords()
        # Hashtable to keep track of TTF; initialize empty
        #self.DF = BagOfJoinedNgrams(max_n) if self.processor.joinChar else BagOfNgrams(max_n)
        self.DF = BagOfWords()
            
        # term weighting for similarity queries
        self.dfweighting = dfweighting
        self.tfweighting = tfweighting
        
        # Token count for the whole corpus; useful for probability estimates
        self.totalTokens = 0
        
        
    def addDoc(self,record):
        # record is assumed to be a dict, list, or tuple
        # Initialize a new doc from the record:
        newDoc = Document(record=record,textAttribute=self.textAttribute if self.keepText else None,
                          IDAttribute=self.IDAttribute,docAttributes=self.docAttributes)
        
        tokens = self.processor.process(record[self.textAttribute])
        bagOfWords = self.processor.bagOfWords(tokens)
        
        # add the unique tokens to the vocabulary, generating ID's for them
        self.vocab.add(bagOfWords.keys())
        
        # convert tokens to ID's now for space efficency, and update TTF and DF
        bagOfIDs = self.bagOfIDs(bagOfWords)
        
        # add count to the TTF and DF
        self.TTF.addBagOfWords(bagOfIDs)
        self.DF.addBagOfWords(bagOfIDs,count=1)
    
        # store the bag of words in efficient ID form
        newDoc.bagOfWords = bagOfIDs
        
        # Increment the corpus total term count
        self.totalTokens += sum(baOfIDs.values())
        
        # store the tokens if specified; for training ngram models
        if self.keepTokens:
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
        bagOfIDs = BagOfWords()
        # consume with popitem()
        while bagOfWords:
            ngram, count = bagOfWords.popitem()
            if ngram in self.vocab.ID:
                ID = self.vocab.ID[ngram]
                bagOfIDs[ID] = count

        return bagOfIDs

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
    def removeTerms(self,terms,docs=False):
        # get the ngram IDS from the vocab for dropping them in all the other structures
        terms = set(terms)
        ngramIDs = [self.vocab.ID[term] for term in terms if term in self.vocab.ID]
        self.vocab.drop(terms)
        self.DF.drop(ngramIDs)
        self.TTF.drop(ngramIDs)
        
        if docs:
            for doc in self.docs.values():
                bagOfWords = doc.bagOfWords
                delkeys = set(bagOfWords).difference(self.vocab.token)
                doc.bagOfWords.drop(delkeys)
        
        
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
            
    # Allows access to the dictionary method keys()
    def keys(self):
        if type(self.docs) is dict:
            return self.docs.keys()
        elif type(self.docs) is list:
            return range(len(self.docs))

