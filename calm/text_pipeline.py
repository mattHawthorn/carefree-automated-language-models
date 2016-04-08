#coding: utf-8

# just the basics:
import json
import os
import re
from math import sqrt, log
from operator import itemgetter # to sort our DF dictionaries by value rather than key



########################################
# TEXT PROCESSOR #######################
########################################


class Processor:
    """
    This can be initiated with a dict of keyword args, which can also be read from confilgFile if specified, a .json file
    Expected arguments are:
        replace: a list of regex duples of form [pattern,replacement]
        stopwords: a list of strings to be removed as stopwords
                   This is often long enough that it makes sense to use a hashset for fast lookup, since lookup
                   must be performed on every token
        tokenizer: a dict specifying which nltk tokenizer to use by name in the nltk.tokenize module as {'name':tokenizerName,'kwargs':keywordArgs}
                   or a regex for splitting as {'regex':regexToSplitOn}
        stemmer: a string specif which nltk stemmer to use by name in the nltk base module as {'name':stemmerName,'kwargs':keywordArgs}
        sequence: the sequence in which to apply the processing steps. A common choice might be 
                  ['replace','tokenize','stopwords','stem'], but other orderings are conceivable
    """
    
    def __init__(self,configFile=None,**kwargs):
        if configFile:
            with open(configFile,'r') as infile:
                kwargs = json.load(infile)
        
        if 'replace' in kwargs:
            # make a list of tuples of form (pattern replacement) where pattern and replacement are compiled regexes
            self.regexes = [(re.compile(x[0]), x[1]) for x in kwargs['replace']]
        else:
            self.regexes = list()    
            
        if 'stopwords' in kwargs:
            # make a hash set from the stopwords
            self.stopwords = set(kwargs['stopwords'])
        else:
            self.stopwords = set()
            
        if 'ngrams' in kwargs:
            lengths = kwargs['ngrams']['n']
            if hasattr(lengths,'__len__'):
                lengths = sorted(list(lengths))
            else:
                lengths = [int(lengths)]
            self.n = lengths
            self.max_n = max(self.n)
                
            try:
                self.maxNgramStopwordProportion = kwargs['ngrams']['maxstopwordproportion']
            except KeyError:
                self.maxNgramStopwordProportion = None
            try:
                self.maxNgramStopwords = kwargs['ngrams']['maxstopwords']
            except KeyError:
                self.maxNgramStopwords = self.max_n
            try:
                self.beginToken = kwargs['ngrams']['begin']
            except KeyError:
                self.beginToken = None
            try:
                self.endToken = kwargs['ngrams']['end']
            except KeyError:
                self.endToken = None
            try:
                self.joinChar = kwargs['ngrams']['joinchar']
            except KeyError:
                self.joinChar = None
            try:
                self.stemNgrams = kwargs['ngrams']['stem']
            except:
                self.stemNgrams = False
            
        else:
            self.n = [1]
            self.max_n = 1
            self.maxNgramStopwords = 1
            self.maxNgramStopwordProportion = None
            self.beginToken = None
            self.endToken = None
            self.joinChar = None
            self.stemNgrams = False
            
        if 'tokenizer' in kwargs:
            tokenizerconfig = kwargs['tokenizer']
            
            if 'regex' in tokenizerconfig:
                regex = re.compile(tokenizerconfig['regex'])
                self.tokenizer = lambda x: re.split(regex,x)
                self.tokenize = self.tokenizer
                
            elif 'name' in tokenizerconfig:
                tokenizername = tokenizerconfig['name']
                
                # import the nltk tokenizer names locally
                from nltk.tokenize import __dict__ as nltkTokenizers
                
                # if the name represents a tokenizer in nltk.tokenize,
                if tokenizername in nltkTokenizers:
                    # get the keyword args
                    if 'kwargs' in tokenizerconfig:
                        tokenizerkwargs = tokenizerconfig['kwargs']
                    else:
                        tokenizerkwargs = {}
                    # then set that tokenizer as the tokenizer,
                    self.tokenizer = nltkTokenizers[tokenizername](**tokenizerkwargs)
                    self.tokenize = self.tokenizer.tokenize

        if 'stemmer' in kwargs:
            stemmerconfig = kwargs['stemmer']
            
            if 'name' in stemmerconfig:
                stemmername = stemmerconfig['name']
                
                # import the nltk stemmer locally
                if stemmername == 'SnowballStemmer':
                    from nltk import SnowballStemmer as stemmer
                elif stemmername == 'PorterStemmer':
                    from nltk import PorterStemmer as stemmer
                
                if 'kwargs' in stemmerconfig:
                    stemmerkwargs = stemmerconfig['kwargs']
                else:
                    stemmerkwargs = {}
                
                # then set that stemmer as the stemmer,
                self.stemmer = stemmer(**stemmerkwargs)
                self.stem = lambda x: [self.stemmer.stem(x)]
                
        self.lower = lambda x: [x.lower()]
        
        if 'sequence' in kwargs:
            nameMap = {'lower':'lower','stem':'stem','tokenize':'tokenize','stopwords':'removeStopwords','replace':'replaceRegexes'}
            self.sequence = [nameMap[processName] for processName in kwargs['sequence']]
            
    def replaceRegexes(self,string):
        for pair in self.regexes:
            string = re.sub(pair[0],pair[1],string)
        if string == '':
            return []
        else:
            return [string]
    
    def removeStopwords(self,token):
        if token in self.stopwords:
            return []
        else:
            return [token]
        
    def tokens(self,string):

        # every processing function will be applied list-wise (allowing, e.g., regex removal after tokenizing)
        # thus the input and result of each intermediate step will be a list
        intermediate = [string]
        
        for functionName in self.sequence:
            # get the processor function
            function = self.__getattribute__(functionName)
            result = list()
            intermediate.reverse()
            while intermediate:
                string = intermediate.pop()
                #print(string)
                stringlist = function(string)
                result = result + stringlist
            
            intermediate = result
            
        return result
        
    def bagOfWords(self,tokens,lengths=None):
        if not lengths:
            lengths = self.n
        # if the argument is a raw string rather than a list of tokens, tokenize it
        if type(tokens) is str:
            tokens = self.tokens(tokens)

        # initialize a new BagOfWords object
        bag = BagOfWords()
        # The greatest-length n-grams the tokens will accomodate:
        max_n = max(self.max_n,len(tokens))
        # Max allowable stopwords in the n-grams (may be adjusted later if self.maxNgramStopWordProportion is present)
        maxStopwords = self.maxNgramStopwords
        if self.maxNgramStopwordProportion:
            maxStopwordProportion = self.maxNgramStopwordProportion
        else:
            maxStopwordProportion = 1.0
        
        # Only need to compute stopword occurrences once, and only if specified- a little more space but a lot less time.
        removeStopwords = False
        if maxStopwords < max_n or maxStopwordProportion < 1:
            removeStopwords = True
            isStopword = [token in self.stopwords for token in tokens]

        # only stem at this stage (as opposed to the processor stage) if specified
        if self.stemNgrams:
            tokens = [self.stem(token)[0] for token in tokens]
        
        # Pad head and tail of doc with begin and end tokens if specified
        if self.beginToken:
            tokens = [self.beginToken] + tokens
            isStopword = [False] + isStopword
        if self.endToken:
            tokens = tokens + [self.endToken]
            isStopword = isStopword + [False]
        #print(tokens)
        # Collect n-grams of all lengths in the list self.n
        for length in self.n:
            if length > max_n:
                break
                
            start = 0
            end = len(tokens) - length + 1
            # skip over the begin and end tokens when counting unigrams
            if length == 1 and self.beginToken:
                start += 1
            if length == 1 and self.endToken:
                end -= 1
            
            # if the stopword policy is specified as a max proportion, compare this to self.maxNgramStopwords
            if removeStopwords and self.maxNgramStopwordProportion:
                maxStopwords = min(self.maxNgramStopwords,int(self.maxNgramStopwordProportion*length))
                #print(maxStopwords)
                
            # select n-grams beginning at each allowable index
            for i in range(start,end):
                if removeStopwords and maxStopwords < length:
                    stopwordCount = sum(isStopword[i:(i+length)])
                    if stopwordCount > maxStopwords:
                        continue
                        
                ngram = tuple(tokens[i:(i+length)])
                
                # join into a string if one is specified
                if self.joinChar:
                    ngram = self.joinChar.join(ngram)
                #print(ngram)
                bag.add(ngram)
        
        return bag




########################################
# FUNCTIONS   ##########################
########################################
        

def IDF(df,docCount,offset=1):
    return offset + log(docCount/df)

def sublinearTF(tf):
    if tf==0:
        return 0
    else:
        return 1 + log(tf)


def cosineSimilarity(bagOfWords1,bagOfWords2,DF,docCount,termweighting=sublinearTF,weighting=IDF):
    """
    First three arguments are hashmaps from token ID's to counts.
    weights can be computed in a customized way by putting any function of in for weighting;
    the default is the standard IDF.
    """
    # reweight the bagOfWords vectors
    vec1 = dict()
    vec2 = dict()
    
    for key in bagOfWords1.keys():
        # only inlcuding terms which haven't been filtered out of the vocab yet
        if key in DF:
            vec1[key] = termweighting(bagOfWords1[key])*weighting(DF[key],docCount)
            
    for key in bagOfWords2.keys():
        if key in DF:
            vec1[key] = termweighting(bagOfWords2[key])*weighting(DF[key],docCount)
    
    # get the norms
    norm1 = L2norm(vec1)
    norm2 = L2norm(vec2)
    # get the common keys; these are all that is needed to compute the similarity
    keys = set(vec1.keys()).intersection(set(vec2.keys()))
    
    cosine = 0

    for key in keys:
        cosine += vec1[key]*vec2[key]
    
    cosine = cosine/(norm1*norm2)

    return cosine



def applyWeight(sparseVector,weighting):
    # apply a weighting function to a sparse vector (hashmap).
    # sparseVector is simply a dict with numeric values.
    vector = dict()
    
    for ID in sparseVector:
        vector[ID] = sparseVector[ID]*weighting(ID)
    
    return vector



def L2norm(sparseVector):
    # L2 norm of a sparse vector (hashmap).
    # sparseVector is simply a dict with numeric values.
    norm = 0
 
    for value in sparseVector.values():
        norm += value**2

    norm = sqrt(norm)

    return norm



##########################g##############
# THE OBJECTS ##########################
########################################
        

class BagOfWords(dict):
    # inherits from dict (hashmap), with added drop and add methods as well as counters.
    # useful not only for document representations but also for corpus-wide term frequency dictionaries (TTF/DF)
    def __init__(self):
        self.ngramCounts = dict()
    
    def add(self,token, count=1):
        if token not in self:
            self[token] = count
        else:
            self[token] += count
        if hasattr(token,'__len__'):
            length = len(token)
        else:
            length = 1
        if length in self.ngramCounts:
            self.ngramCounts[length] += count
        else:
            self.ngramCounts[length] = count

    def addBagOfWords(self,bagOfWords):
        for token in bagOfWords:
            self.add(token,bagOfWords[token])
            
    def drop(self,tokens):
        if not hasattr(tokens,'__len__'):
            tokens = [tokens]
            
        for token in tokens:
            if token not in self:
                continue
                
            if hasattr(token,'__len__'):
                length = len(token)
            else:
                length = 1
                
            self.ngramCounts[length] -= self[token]
            del self[token]



class Document:
    # This is implemented as Post in the example java.  I named it more generally for potential future uses.
    def __init__(self,record,attributes,):
        for attribute in attributes:
            if attribute in record:
                # add the record attribute as a class attribute of Document
                self.__dict__[attribute] = record[attribute]
            else:
                # if absent, place a null value to avoid later attribute errors
                self.__dict__[attribute] = None
                


class Vocabulary:
    """
    A hashtable mapping unigrams to ID's and a reverse hash mapping ID's to unigrams 
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
        # Also allows removing custom lists generated from DF statistics.
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
                    
                                        

class Corpus:
    """
    a hashtable mapping docID's to docs.
    docs have a bag-of-words representation, along with configurable attributes: author, date, rating, location, etc.
    processor is of class Processor
    TTF and DF are dicts of total term frequencies and document frequencies respectively.
    """
    def __init__(self,processor,docAttributes,textAttribute,IDAttribute=None,DF=True,TTF=True):
        # the hastable of docs; initialize empty
        self.__docs = dict()
        self.docCount = 0
        # the string processor/tokenizer
        self.processor = processor
        # tokenCount keeps track of counts of all 1,2,...n-grams
        self.ngramCounts = dict()
        for n in self.processor.n:
            self.ngramCounts[n] = 0
        # these are the fields that each doc should have (list)
        self.docAttributes = docAttributes
        # this is the field where the raw text is found in incoming records (string)
        self.__textAttribute = textAttribute
        # this is the field where the document ID is found in incoming records. If None, an integer index is used
        self.__IDAttribute = IDAttribute
        # The vocab of the corpus; initialize empty
        self.vocab = Vocabulary()
        # Hashtable to keep track of TTF; initialize empty
        if TTF:
            self.TTF = BagOfWords()
        # Hashtable to keep track of TTF; initialize empty
        if DF:
            self.DF = BagOfWords()
        # Token count for the whole corpus; useful for probability estimates
        self.totalTokens = 0
        
    def addDoc(self,record,update=True):
        # record is assumed to be a dict
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
        if update:
            self.vocab.add(bagOfWords.keys())
        
        # update the corpus-wide token counts with the incoming bagOfWords
        for length in bagOfWords.ngramCounts:
            if length in self.ngramCounts:
                self.ngramCounts[length] += bagOfWords.ngramCounts[length]
            else:
                self.ngramCounts[length] = bagOfWords.ngramCounts[length]
        
        # convert tokens to ID's now for space efficency, and update TTF and DF
        bagOfIDs = dict()
        # consume as you go with pop()
        while bagOfWords:
            ngram, count = bagOfWords.popitem()
            if ngram in self.vocab.ID:
                ID = self.vocab.ID[ngram]
                bagOfIDs[ID] = count

            if update:
                # add count to the TTF and DF
                self.TTF.add(ID,count)
                self.DF.add(ID,1)
            
        # store the bag of words in efficient ID form
        newDoc.__setattr__('bagOfWords',bagOfIDs)
        # and finally store the document record
        self.__docs[docID] = newDoc
        self.docCount += 1

        
    def cosine(self,docID1,docID2):
        vector1 = self[docID1].bagOfWords
        vector2 = self[docID2].bagOfWords
        
        return(cosineSimilarity(vector1,vector2,self.DF,self.docCount))
    
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
    
    # remove an iterable of ngrams from the corpus, including each document's bagOfWords if indicated
    def removeNgrams(self,ngrams,docs=False):
        # get the ngram IDS from the vocab for dropping them in all the other structures
        ngramIDs = [self.vocab.ID[ngram] for ngram in ngrams]
        self.vocab.drop(ngrams)
        self.DF.drop(ngramIDs)
        self.TTF.drop(ngramIDs)
        
        if docs == True:
            for doc in self.__docs:
                self.__docs[doc].bagOfWords.drop(ngramIDs)
        
    # Allows direct access to docs as Corpus[docID]
    def __getitem__(self,docID):
        try:
            doc = self.__docs[docID]
        except:
            return None
        else:
            return doc
    
    # Allows access to the "key in item" syntax
    def __contains__(self,docID):
        return (docID in self.__docs)
            
    # Allows access to the dictionary (hashtable) method keys()
    def keys(self):
        return self.__docs.keys()
        



########################################
# DOCUMENT GENERATOR ###################
########################################
        

class DocIter:
    """
    Iterates over docs in a directory or a file.  recordIter is any iterable returning docs as dicts of attribute:value pairs.
    you can supply any class here that will iterate over docs/records in a file, given a path to the file. 
    In this case the docs are records within json files in the directory,
    or (if only a json file is supplied), records within a single json file.
    By using this iterator, you never need to have more than the contents of one input file in memory at a time during processing/analysis.
    """
    
    def __init__(self,path,recordIter,recursive=False,extensions=['.json']):
        # what kind of files to read?
        self.extensions = extensions
        self.recordIter = recordIter
        
        # if path exists,
        if os.path.exists(path):
            # and path is a directory
            if os.path.isdir(path):
                # set up a recursive path walk iterator
                self.dirIter = os.walk(path)
                # but if not recursive, only take the contents of path as dirIter
                if not recursive:
                    self.dirIter = iter([next(self.dirIter)])
                    
            # otherwise assume path is a file.
            else:
                # no iteration on directories or files; dirIter will just have one parentDir, subDir, and filename
                directory, filename = os.path.split(path)
                subDir = os.path.split(directory)[1]
                # structure these as in the output from os.walk()
                self.dirIter = iter([(directory,[subDir],[filename])])
                    
        # else, path doesn't exist
        else:
            print(path+" not found. Iterator not initiated")
        
        # start with file and doc/record iterators empty; these will be initialized from dirIter
        self.fileIter = iter([])
        self.docIter = iter([])
        
    
    def __iter__(self):
        return(self)
    
    
    def __next__(self):
        
        while True:
            # try to get a doc
            try:
                nextDoc = next(self.docIter)
            # if not, docIter is exhausted or uninitiated; get new file from fileIter
            except:
                try:
                    nextFile = next(self.fileIter)

                    # if not, fileIter is exhausted or uninitiated; get new dir from dirIter
                except:
                    try:
                        nextDir = next(self.dirIter)
                    # if not, dirIter is exhausted; end
                    except:
                        raise StopIteration
                    # if that worked, make a fileIter from nextDir
                    else:
                        self.fileIter = FileIter(directory=nextDir[0], files=nextDir[2], extensions=self.extensions)
                # if that worked, make a docIter from nextFile
                else:
                    self.docIter = self.recordIter(filepath=nextFile)
            # if so, return the doc
            else:
                return nextDoc
                break


class FileIter:
    # Iterates over files in directory, returning complete paths, as long as the file extension is in extensions
    def __init__(self, directory, files, extensions):
        self.directory = directory
        files = [filename for filename in files if os.path.splitext(filename)[1] in extensions]
        self.files = iter(files)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            nextfile = next(self.files)
        except StopIteration:
            raise StopIteration
        else:
            path = os.path.join(self.directory,nextfile)
            return path
        
# Example record iterator: this one returns records from a .json of yelp reviews 
#class RecordIter:
#    # iterates over individual records in a file, in this case a json with records in a slot labelled 'Reviews'
#    def __init__(self, filepath, encodings=['iso-8859-1','utf-16le']):
#        with open(filepath,'r') as readfile:
#            data = json.load(readfile)
#            self.data = data['Reviews']
#            self.docs = iter(self.data)
#            
#    def __iter__(self):
#        return self
#    
#    def __next__(self):
#        try:
#            nextdoc = next(self.docs)
#        except:
#            raise StopIteration
#        else:
#            return nextdoc

