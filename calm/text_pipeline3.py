#coding: utf-8

import json
import os
import re
from math import sqrt, log
from operator import itemgetter # to sort our DF dictionaries by value rather than key
import config_loader

########################################
# TEXT PROCESSOR #######################
########################################


class Processor:
    """
    This can be initiated with a dict of keyword args, which can also be read from confilgFile if specified, either .json or .yml
    Expected arguments are:
        sequence: a list of operations to apply. Each takes the form: 
                {"operation":name_of_operation, "args":{arg1:value1,arg2:value2, ...}}

        ngrams: a dict specifying the behavior of ngram collection. Has the form:
                {"n":[1,2,...] OR 2, "maxstopwords": 1,"maxstopwordproportion": 0.5, "joinchar":"_"}
                
                Notes:
                If n is a list, ngrams of every length in the list, and only those lengths, are kept. This is useful for
                vector-space models.
                
                If n is an int, only ngrams of that length are kept. This is useful for n-gram language models.
                If joinchar is omitted, ngrams are stored as tuples (useful for n-gram language models), else they are joined into
                a string.
                
                Either of maxstopwords and maxstopwordproportion may be included or omitted.  There are no conflicts between them.
                Of course, certain combinations will result in an less efficent processor, since checks will be performed on both
                parameters. E.g. with maxStopwords=1 and maxStopwordProportion=.5 and n=2, there is some redundancy; the proportion
                could be omitted.
                
    Possible operations included in the above sequence, together with their result and specification are as follows:
        replace: apply re.sub() to replace a regex with a string. Ex.: 
                {"operation":"replace","args":{"pattern":"[A-Z]{2,}","replacement":"ACRONYM"}}

        lower: apply str.lower() to push an entire string to lowercase. Ex.: 
                {"operation":"lower"}

        split: apply re.split() to split a string on a regex. Ex.: 
                {"operation":"split","args":{"pattern":"[\s]+"}}
                Note: use capturing parentheses on the pattern to keep the split groups.

        filter: remove tokens (input is assumed to be in tokenized form) according to a full match with a regex. Ex.: 
                {"operation":"filter","args":{"pattern":"[0-9]+"}}

        stopwords: remove stopwords.
                {"operation":"stopwords","args":{"file":"stopwords.txt"}}

        tokenize: apply a tokenizer from the nltk.tokenize module. Must specify the name as it appears in that module and give 
                keyword args for instantiation of an instance of that tokenizer. Ex.: 
                "args":{"name":"RegexpTokenizer","kwargs":{"pattern":"\\s+","gaps":true,"discard_empty":true}}
                Notes: this is optional, since many simple tokenization tasks can be accomplished with a regex using re.split().

        stem: apply a stemmer from the nltk module. Ex.: 
                {"operation":"stem","args":{"name":"SnowballStemmer","kwargs":{"language":"english"}}}
    """
    
    def __init__(self,configFile=None,**kwargs):
        if configFile:
            kwargs = cl.load_config(configFile)
        
        if 'sequence' not in kwargs:
            raise ValueError("No processing sequence specified. See docstring for details.")
        else:
            sequence = kwargs['sequence']
        

        ngramDefaults = {"n":[1],"max_n":1,"maxNgramStopwords":None,"maxNgramStopwordProportion":None,
                "beginToken":None,"endToken":None,"joinChar":None,"stemNgrams":False}
        
        if 'ngrams' in kwargs:
            ngramConfig = kwargs['ngrams']
            if "n" in ngramConfig and not ngramConfig["n"].hasattr("len"):
                ngramConfig["n"]=[int(ngramConfig["n"])]
            ngramConfig["max_n"]=max(ngramConfig["n"])
            ngramDefaults.update(ngramConfig)
            
        ngramConfig = ngramDefaults
        
        self.sequence = list()

        for op in sequence:
            operation = get_name(op['operation'],opThesaurus)
            args = clean_args(op['args'],argThesaurus[operation])

            # every function takes a list of strings and returns a list of strings
            # I prefer to unpack these as list comprehensions to avoid extra function calls (lambda)
            # and checks to determine whether the result needs to be unlisted
            if operation=='replace':
                pattern = re.compile(args['pattern'])
                f = lambda tokens: [re.sub(pattern,args['repl'],s) for s in tokens]
                self.sequence.append(f)
                        elif op====ion=='split':
                pattern = re.compile(args['pattern'])
                if 'keep' in args:
                    args['pattern']='('+pattern+')'
                self.sequence.append(lambda s: re.split(args['pattern'],s))
            elif operation=='filter':
                pattern = re.compile(args['pattern'])
                if 'keep' in args and args['keep']:
                    self.sequence.append(lambda s: [s] if re.match(pattern,s) else [])
                else:
                    self.sequence.append(lambda s: [] if re.match(pattern,s) else [s])
            elif operation=='lower':
                if 'pattern' not in args:

            elif operation=='upper':
                if 'pattern' not in args:
                    self.squence.append((lambda s: s.upper(),False))
                else:
                    pattern = re.compile(args['pattern'])
                    self.sequence.append(lambda s: 
            elif operation=='stopwords':
            elif operation=='tokenize':
            elif operation=='stem':


        if 'tokenize' in kwargs:
            tokenizerconfig = kwargs['tokenize']['args'
            
            if 'regex' in tokenizerconfig:
                regex = re.compile(tokenizerconfig['regex'])
                self.tokenizer = lambda x: re.split(regex,x)
                self.tokenize = self.tokenizer
                
            elif 'name' in tokenizerconfig:
                tokenizername = tokenizerconfig['name']
                
                # import the nltk tokenizer names locally
                from nltk.tokenize import __dict__ as nltkTokenizers
                
                        if type(params['exclude']) is not bool:
                            raise ValueError("exclude must be boolean in {} args".format(operation))
                        else:
                            exclude = params['exclude']
                    args = (case,pattern,match,exclude)
                    f = self.__caseMatched

            elif operation=='stopwords':
                if 'stopwords' not in kwargs:
                    raise ValueError("No 'stopwords' key in the config."+
                                     "\nStopwords must be specified in order to be removed")
                args = None
                f = self.__removeStopwords

            elif operation=='tokenize':
                tokenizer = nltkTokenizers.__dict__[params['name']]
                self.tokenizer = tokenizer(**(params['kwargs']))
                self.tokenize = self.tokenizer.tokenize
                f = self.__tokenize
                args = None

            elif operation=='stem':
                # import the nltk stemmer locally
                if params['name'] == 'SnowballStemmer':
                    from nltk import SnowballStemmer as stemmer
                elif params['name'] == 'PorterStemmer':
                    from nltk import PorterStemmer as stemmer
                else:
                    raise ValueError("Unsupported stemmer: {}".format(params['name']))
                self.stemmer = stemmer( **(params['kwargs']) )
                self.stem = self.stemmer.stem
                f = self.__stem
                args = None

            self.sequence.append((f,args))


    # private functions for the heavy-lifting tasks behind the scenes
    def __replace(strings,args):
        return [re.sub(args[0],args[1],s) for s in strings]

    def __split(strings,args):
        return from_iterable([re.split(args[0],s) for s in strings])

    def __filter(strings,args):
        p = args[0] # the regex pattern
        match = args[1] # the matching function
        if args[2]: # discard/keep
            return [s for s in strings if not re.match(pattern,s)]
        else:
            return [s for s in strings if re.match(pattern,s)]

    def __case(strings,args):
        f = args[0] # the case function
        return [f(s) for s in strings]

    def __caseMatched(strings,args):
        f = args[0] # the case function
        p = args[1] # the regex pattern
        match=args[2] # the matching function
        if args[3]: # only/except
            return [f(s) if match(p,s) else s for s in strings]
        else:
            return [f(s) if not match(p,s) else s for s in strings]

    def __removeStopwords(strings,args):
        return [s for s in strings if s not in self.stopwords]

    def __stem(strings,args):
        return [self.stem(s) for s in strings]

    def __tokenize(strings,args):
        return from_iterable([self.tokenize(s) for s in strings])


    # the main processing function: take a doc as a string and return a list of tokens
    def process(self,string):
        # every processing function in self.sequence will be applied in order 
        # the the input and result of each intermediate step is a list
        intermediate = [string]

        for f,args in self.sequence:
            tokens = f(tokens,args)
        
        return tokens


    # take a list of tokens (such as would be produced by self.process, and return a bag of words
    def bagOfWords(self,tokens,lengths=None):
        # PREPARATIONS
        if not lengths:
            lengths = self.n
        # if the argument is a raw string rather than a list of tokens, tokenize it
        if type(tokens) is str:
            tokens = self.tokens(tokens)

        # The greatest-length n-grams the tokens will accomodate:
        max_n = min(max(lengths),len(tokens))
        
        # Max allowable stopwords in the n-grams (may be adjusted later if self.maxNgramStopWordProportion is present)
        maxStopwords = self.maxNgramStopwords
        if self.maxNgramStopwordProportion:
            maxStopwordProportion = self.maxNgramStopwordProportion
        else:
            maxStopwordProportion = 1.0
        # Only need to compute stopword occurrences once, and only if specified- a little more space but a lot less time.
        removeStopwords = False
        if maxStopwords < max_n or maxStopwordProportion < 1.0:
            removeStopwords = True
            isStopword = [(1 if token in self.stopwords else 0) for token in tokens]

        # only stem at this stage (as opposed to the processor stage) if specified
        if self.stemNgrams:
            tokens = [self.stem(token) for token in tokens]
        
        # Pad head and tail of doc with begin and end tokens if specified
        if self.beginToken:
            tokens = [self.beginToken] + tokens
            if removeStopwords:
                isStopword = [0] + isStopword
                max_n+=1
        if self.endToken:
            tokens = tokens + [self.endToken]
            if removeStopwords:
                isStopword = isStopword + [0]
                max_n+=1
        

        # initialize a new BagOfWords object
        bag = BagOfWords()

        # MAIN LOOP
        # Collect n-grams of all lengths in the list self.n
        # nothing that follows is pythonic, but it should be fast!
        for n in self.n:
            if n > max_n:
                break
                
            start = n
            end = len(tokens)
            # skip over the begin and end tokens when counting unigrams
            if n == 1 and self.beginToken:
                start += 1
            if n == 1 and self.endToken:
                end -= 1
            
            # initialize the ngrams
            ngrams = [None]*(end-start)

            # Main loop for the stopword removal case
            if removeStopwords:
                # if the stopword policy is specified as a max proportion, 
                # compare this to self.maxNgramStopwords
                maxStopwords = min(maxStopwords,int(maxStopwordProportion*n))
                
                # use a deque for efficiency for large n to avoid redundant list slicing
                if n>2:
                    t = deque(['']+tokens[start-n-1:start-1])
                    s = deque([0]+isStopword[start-n-1:start-1])
                    c = sum(s)
                    j = 0
                    start-=1
                    for i in range(start,end):
                        t.popleft()
                        c = c - s.popleft()
                        t.append(tokens[i])
                        s.append(isStopword[i])
                        c = c + isStopword[i]
                        if c > maxStopwords:
                            continue
                        else:
                            ngrams[j] = tuple(t)
                            j += 1
                    ngrams = ngrams[0:j]

                else:
                    ngrams = [tuple(tokens[(i-n):i]) for i in range(start,end) if sum(isStopword[(i-n):i]) <= maxStopwords]
                    # for i in range(start,end):
                    #     c = sum(isStopword[(i-n):i]
                    #     if c > maxStopwords:
                    #         continue
                    #     ngram = tuple(tokens[(i-n):i])
            else:
                # loop without considering stopword counts
                # use a deque for efficiency for large n to avoid redundant list slicing
                if n>2:
                    t = deque(['']+tokens[start-n-1:start-1])
                    j = 0
                    start-=1
                    for i in range(start,end):
                        t.popleft()
                        t.append(tokens[i])
                        ngrams[j] = tuple(t)
                        j += 1
                    ngrams = ngrams[0:j]
                else:
                    ngrams = [tuple(tokens[(i-n):i]) for i in range(start,end)]
            
            # done with the main loop
            # JOIN NGRAMS IF SPECIFIED
            if self.joinChar:
                ngrams = [self.joinChar.join(ngram) for ngram in ngrams]

            bag.add(ngram)
        
        return bag



#########################################
# THESAURI FOR PROCESSOR CONFIG FILES ###
#########################################

# filter for comparing keywords
spaces = re.compile('[-_\s]')

# thesauri for translating config
configThesaurus = {"sequence":{"sequence","functions","ops","operations",
                                 "functionsequence","operationsequence","functionorder",
                                 "operationorder","orderofoperations"},
                   "ngrams":{"ngrams","ngram","ngramconfig","ngramparameters","ngramparams"},
                   "stopwords":{"stopwords"}
                  }

opThesaurus = {"operation":{"operation","op","function"},
               "args":{"args","arguments","params","parameters","config","configuration"}
               }
operationThesaurus = {"replace":{"replace","substitute","sub","replaceregex","repl",
                            "substituteregex","subregex"},
               "split":{"split","break","splitonregex","regexsplit","splitregex"},
               "filter":{"filter","remove","filterregex","removeregex",
                         "filterpattern","removepattern"},
               "retain":{"retain","keep","retainregex","keepregex",
                         "retainpattern","keeppattern"},
               "lower":{"lower","lowercase","pushtolower","pushtolowercase"},
               "upper":{"upper","uppercase","pushtoupper","pushtouppercase"},
               "stopwords":{"stopwords","removestopwords","filterstopwords"},
               "tokenize":{"tokenize","tokens","tokenizer"},
               "stem":{"stem","stemmer","stemtoroot"}
             }

patternSynonyms = {"pattern","regex"}
kwargSynonyms = {"kwargs","keywordarguments","keywordargs","arguments","args"}

filterThesaurus = {"pattern":patternSynonyms,
                   "match":{"match","matchtype"}
                   }

caseThesaurus = {"exclude":{"exclude","skip","not","negate"}}.union(filterThesaurus)

matchThesaurus = {"full":{"full","fullmatch","completematch","totalmatch"},
                  "partial":{"partial","partialmatch","contains"}
                  }

argThesauri = {"replace":{"pattern":patternSynonyms,
                          "repl":{"repl","replacement","substitute","replacementstring"}
                          },
               "split":{"pattern":patternSynonyms,
                        "keep":{"keep","keepsplits","keepregexes"}
                        },
               "filter":filterThesaurus,
               "retain":filterThesaurus,
               "lower":caseThesaurus,
               "upper":caseThesaurus,
               "stopwords":{"file":{"file","filename","path","stopwordsfile"},
                            "list":{"list","stopwords","stopwordslist"}
                            },
               "tokenize":{"name":{"name","tokenizer","nltktokenizer","tokenizername","nltktokenizername"},
                           "kwargs":kwargSynonyms
                           },
               "stem":{"name":{"name","stemmer","nltkstemmer","stemmername","nltkstemmername"},
                       "kwargs":kwargSynonyms
                       }
               }



########################################
# FUNCTIONS ############################
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



########################################
# THE OBJECTS ##########################
########################################


class BagOfWords(dict):
    # inherits from dict (hashmap), with added drop and add methods as well as counters.
    # useful not only for document representations but also for corpus-wide term frequency dictionaries (TTF/DF)
    def __init__(self,ngrams=None):
        if ngrams:
            [self.add(ngram) for ngram in ngrams]

    def add(self,token, count=1):
        if token not in self:
            self[token] = count
        else:
            self[token] += count
        
    def addBagOfWords(self,bagOfWords):
        # note that the syntax here allows bagOfWords to be any indexed object with numeric values;
        # it need not be a BagOfWords object.
        for ngram in bagOfWords:
            self.add(ngram,bagOfWords[ngram])
            
    def drop(self,ngrams):
        if type(ngrams) is str:
            if ngram in self:
                del self[ngram]
            return

        for ngram in ngrams:
            if ngram not in self:
                continue
            del self[token]



class Document:
    # This is implemented as Post in the example java.  I named it more generally for potential future uses.
    def __init__(self,record,attributes):
        for attribute in attributes:
            if attribute in record:
                # add the record attribute as a class attribute of Document
                self.__dict__[attribute] = record[attribute]
            else:
                # if absent, place a null value to avoid later attribute errors
                self.__dict__[attribute] = None

    def __getitem__(self,key):
        return self.__dict__[key]


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

