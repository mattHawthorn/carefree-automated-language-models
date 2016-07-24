#coding: utf-8
import os
import re
from . import config_loader as cl
from .objects import BagOfWords
from itertools import chain
from collections import deque
from nltk import tokenize as nltkTokenizers


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
                
        strip: apply str.strip(chars) to a string. Ex.:
                {"operation":"strip","args":{"chars":" -_.?!\\t\\n"}}

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
    
    def __init__(self,configFile=None,stopwordsFile=None,**kwargs):
        # load the config file if specified
        if configFile:
            config = cl.load_config(configFile)

            if not kwargs:
                kwargs = config
            else:
                kwargs.update(config)
        
        # standardize the config keywords at the top level
        kwargs = cl.clean_args(kwargs,configThesaurus,remove=spaces)
        
        # a sequence of processing steps must be specified
        if 'sequence' not in kwargs:
            raise ValueError("No processing sequence specified. See docstring for details.")
        else:
            sequence = kwargs['sequence']

        # configure ngram handling
        ngramDefaults = {"n":[1],"maxNgramStopwords":None,"maxNgramStopwordProportion":None,
                "beginToken":None,"endToken":None,"join":True,"joinChar":' ',"stemNgrams":False}
        
        if 'ngrams' in kwargs:
            ngramConfig = cl.clean_args(kwargs['ngrams'],ngramThesaurus,remove=spaces)
            if "n" in ngramConfig and type(ngramConfig["n"]) not in [set,list,tuple]:
                ngramConfig["n"]=[int(ngramConfig["n"])]
            if 'join' in ngramConfig:
                if ngramConfig['join']:
                    ngramConfig['join'] = True
                    if 'joinChar' not in ngramConfig:
                        ngramConfig['joinChar'] = ' '
                else:
                    ngramConfig['join'] = False
                    ngramConfig['joinChar'] = None
            elif 'joinChar' in ngramConfig:
                ngramConfig['join'] = True
                
            ngramConfig["n"] = [int(n) for n in ngramConfig["n"]]
            ngramDefaults.update(ngramConfig)
            
        # add the ngram config params as class attributes
        self.__dict__.update(ngramDefaults)
        
                
        self.stemmer = None
        if 'stemmer' in kwargs:
            params = cl.clean_args(kwargs['stemmer'],argThesauri['stem'])
            self.stemmer = loadStemmer(params['name'],**(params['kwargs']))
            self._stem = self.stemmer.stem
        
        self.tokenizer = None
        if 'tokenizer' in kwargs:
            params = cl.clean_args(kwargs['tokenizer'],argThesauri['tokenize'])
            self.tokenizer = loadTokenizer(params['name'],**(params['kwargs']))
            self._tokenize = self.tokenizer.tokenize
        
        # read stopwords from a text file or directly as a list or set
        self.stopwords = None
        if stopwordsFile:
            if 'stopwords' in kwargs:
                print("Warning: stopwords were specified in both the config file and the init args; "+
                      "defaulting to the init args stopwords: {}".format(stopwordsFile))
            self.stopwords = set(loadStopwords(stopwordsFile))
        elif 'stopwords' in kwargs:
            if type(kwargs['stopwords']) is str:
                stopDir = os.path.split(configFile)[0]
                stopwordsFile = os.path.join(stopDir,kwargs['stopwords'])
                self.stopwords = set(loadStopwords(stopwordsFile))
            elif type(kwargs['stopwords']) not in [set,list,tuple]:
                raise ValueError("Stopwords must be given as a filename or a set-like or list-like"+
                                 " object in the config, not {}".format(type(kwargs['stopwords'])))
            else:
                self.stopwords = set(kwargs['stopwords'])

        # initialize the list of processing steps
        self.sequence = list()


        # for all operations in the processing sequence, define functions and append to the sequence
        for i, op in enumerate(sequence):
            # standardize the operation,args keywords for the operation
            op = cl.clean_args(op,opThesaurus,remove=spaces)
            # get the standard name for the operation
            operation = cl.get_name(op['operation'],operationThesaurus)
            # standardize the argument names
            params=None
            
            if operation not in {"lower","upper","strip","stopwords","tokenize","stem"}:
                if "args" not in op:
                    raise ValueError('{} must have an "args" entry'.format(op))
            if "args" in op:
                params = cl.clean_args(op['args'],argThesauri[operation],remove=spaces)
                
            # every function takes a list of strings and returns a list of strings
            # I prefer to unpack these as list comprehensions to avoid extra function calls (lambda)
            # and checks to determine whether the result needs to be unlisted.
            # in those cases where the function returns a list, itertools.chain.from_iterable is used
            # to unpack the list of lists
            if operation=='replace':
                args = (re.compile(params['pattern']),params['repl'])
                f = self.replace

            elif operation=='split':
                if 'keep' in params:
                    params['pattern']='('+params['pattern']+')'
                args = (re.compile(params['pattern']),)
                f = self.split
                
            elif operation=='strip':
                if not params:
                    args = (None,)
                else:
                    args = (params['chars'],)
                f = self.strip

            elif operation in {'filter','retain'}:
                pattern = re.compile(params['pattern'])
                match = 'full'
                if 'match' in params:
                    match = cl.get_name(params['match'],matchThesaurus)
                matcher = re.fullmatch if match=='full' else re.search
                discard = True if operation=='filter' else False
                args = (pattern,matcher,discard)
                f = self.filter

            elif operation in {'lower','upper'}:
                case = str.lower if operation=='lower' else str.upper
                if not params:
                    args = (case,)
                    f = self.case;
                else:
                    pattern = re.compile(params['pattern'])
                    match = 'full'
                    if 'match' in params:
                        match = cl.get_name(params['match'],matchThesaurus)
                        matcher = re.fullmatch if match=='full' else re.search
                    exclude = False
                    if 'exclude' in params:
                        if type(params['exclude']) is not bool:
                            raise ValueError("exclude must be boolean in {} args".format(operation))
                        else:
                            exclude = params['exclude']
                    args = (case,pattern,matcher,exclude)
                    f = self.caseMatched

            elif operation=='stopwords':
                if not self.stopwords:
                    print("Warning: stopword removal is indicated, but no stopwords are specified")
                if max(self.n) > 1:
                    print("Warning: stopword removal is inidicated prior to ngram collection; ngrams will not reflect actual proximity.")
                args = None
                f = self.removeStopwords

            elif operation=='tokenize':
                if self.tokenizer and params:
                    print("Warning: tokenizer was specified twice in the config; defaulting to the one defined in the operation sequence.")
                if params:
                    self.tokenizer = loadTokenizer(params['name'],**(params['kwargs']))
                    self._tokenize = self.tokenizer.tokenize
                if not self.tokenizer:
                    raise ValueError("Tokenization is indicated, but no tokenizer is specified")
                f = self.tokenize
                args = None

            elif operation=='stem':
                if self.stemmer and params:
                    print("Warning: stemmer was specified twice in the config; defaulting to the one defined in the operation sequence.")
                if params:
                    self.stemmer = loadStemmer(params['name'],**(params['kwargs']))
                    self._stem = self.stemmer.stem
                if not self.stemmer:
                    raise ValueError("Stemming is indicated, but no stemmer is specified")
                f = self.stem
                args = None
                
            else:
                raise KeyError("No operation exists for keyword {}".format(operation))

            self.sequence.append((f,args))


    # private functions for the heavy-lifting tasks behind the scenes
    def replace(self,strings,args):
        return [re.sub(args[0],args[1],s) for s in strings]

    def split(self,strings,args):
        return list(filter(lambda s: s is not None,chain.from_iterable([re.split(args[0],s) for s in strings])))
        
    def strip(self,strings,args):
        return [s.strip(args[0]) for s in strings]

    def filter(self,strings,args):
        p = args[0] # the regex pattern
        match = args[1] # the matching function
        if args[2]: # discard/retain
            return [s for s in strings if not match(p,s)]
        else:
            return [s for s in strings if match(p,s)]

    def case(self,strings,args):
        f = args[0] # the case function
        return [f(s) for s in strings]

    def caseMatched(self,strings,args):
        f = args[0] # the case function
        p = args[1] # the regex pattern
        match=args[2] # the matching function
        if args[3]: # only/except
            return [f(s) if not match(p,s) else s for s in strings]
        else:
            return [f(s) if match(p,s) else s for s in strings]

    def removeStopwords(self,strings,args=None):
        return [s for s in strings if s not in self.stopwords]

    def stem(self,strings,args=None):
        return [self._stem(s) for s in strings]

    def tokenize(self,strings,args=None):
        return list(chain.from_iterable([self._tokenize(s) for s in strings]))


    # the main processing function: take a doc as a string and return a list of tokens
    def process(self,string):
        # every processing function in self.sequence will be applied in order 
        # the the input and result of each intermediate step is a list
        if type(string) is str:
            tokens = [string]
        else:
            tokens = string

        for f,args in self.sequence:
            tokens = f(tokens,args)
        
        return [t for t in tokens if t!='']


    # take a list of tokens (such as would be produced by self.process, and return a bag of words
    def bagOfWords(self,tokens,lengths=None):
        # PREPARATIONS
        if not lengths:
            lengths = self.n
        else:
            if type(lengths) not in {set,list,tuple}:
                lengths = [int(lengths)]
        # if the argument is a raw string rather than a list of tokens, tokenize it
        if type(tokens) is str:
            tokens = self.process(tokens)

        # The greatest-length n-grams the tokens will accomodate:
        max_n = min(max(lengths),len(tokens))
        
        # Max allowable stopwords in the n-grams (may be adjusted later if self.maxNgramStopWordProportion is present)
        if self.maxNgramStopwords is not None:
            maxStopwords = self.maxNgramStopwords
        else:
            maxStopwords = max_n

        if self.maxNgramStopwordProportion:
            maxStopwordProportion = self.maxNgramStopwordProportion
        else:
            maxStopwordProportion = 1.0
        
        removeStopwords = False
        # Only need to compute stopword occurrences once, and only if specified- a little more space but a lot less time.
        if (maxStopwords < max_n or maxStopwordProportion < 1.0) and self.stopwords:
            removeStopwords = True
            isStopword = [(1 if token in self.stopwords else 0) for token in tokens]
        # only stem at this stage (as opposed to the processor stage) if specified
        if self.stemNgrams:
            tokens = self.stem(tokens)
        
        # Pad head and tail of doc with begin and end tokens if specified
        if self.beginToken:
            tokens = [self.beginToken] + tokens
            if removeStopwords:
                isStopword = [0] + isStopword
        
        if self.endToken:
            tokens = tokens + [self.endToken]
            if removeStopwords:
                isStopword = isStopword + [0]
        
        # update max_n in case start and end tokens were added
        max_n = min(max(lengths),len(tokens))

        # initialize a new BagOfWords object
        bag = BagOfWords()

        # MAIN LOOP
        # Collect n-grams of all lengths in the list self.n
        # nothing that follows is pythonic, but it should be fast!
        for n in lengths:
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
            ngrams = [None]*(end-start+1)
            # Main loop for the stopword removal case
            if removeStopwords:
                # if the stopword policy is specified as a max proportion, 
                # compare this to self.maxNgramStopwords
                maxStopwords = min(maxStopwords,int(maxStopwordProportion*n))
                # use a deque for efficiency for large n to avoid redundant list slicing
                if n>3:
                    start-=1
                    t = deque(['']+tokens[start-n+1:start])
                    s = deque([0]+isStopword[start-n+1:start])
                    c = sum(s)
                    j = 0
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
                    end+=1
                    ngrams = [tuple(tokens[(i-n):i]) for i in range(start,end) if sum(isStopword[(i-n):i]) <= maxStopwords]
            else:
                # loop without considering stopword counts
                # use a deque for efficiency for large n to avoid redundant list slicing
                if n>3:
                    start-=1
                    t = deque(['']+tokens[start-n+1:start])
                    j = 0
                    for i in range(start,end):
                        t.popleft()
                        t.append(tokens[i])
                        ngrams[j] = tuple(t)
                        j += 1
                    ngrams = ngrams[0:j]
                else:
                    end+=1
                    ngrams = [tuple(tokens[(i-n):i]) for i in range(start,end)]
            
            # done with the main loop
            # JOIN NGRAMS IF SPECIFIED
            if self.joinChar:
                ngrams = [self.joinChar.join(ngram) for ngram in ngrams]

            bag.addList(ngrams)
        
        return bag



def loadTokenizer(name,**kwargs):
    from nltk.tokenize import __dict__ as nltkTokenizers
    if 'Tokenizer' in name and name in nltkTokenizers:
        tokenizer = nltkTokenizers[name](**kwargs)
    else:
        raise ValueError("Unsupported tokenizer: {}. Must specify an nltk tokenizer".format(name))
    del nltkTokenizers
    return tokenizer


def loadStemmer(name,**kwargs):
    # import the nltk stemmer locally
    from nltk import __dict__ as nltkObjects
    if 'Stemmer' in name and name in nltkObjects:
        stemmer = nltkObjects[name](**kwargs)
    else:
        raise ValueError("Unsupported stemmer: {}. Must specify an nltk stemmer".format(name))
    del nltkObjects
    return stemmer
        

def loadStopwords(stopwordsFile):
    if os.path.splitext(stopwordsFile)[1] in cl.legalConfigExtensions:
        stopwords = set(cl.load_config(stopwordsFile))
    else:
        stopwords = set()
        with open(stopwordsFile,'r') as infile:
            for line in infile:
                stopwords.add(line.strip())
    return stopwords




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
                   "stopwords":{"stopwords"},
                   "stemmer":{"stemmer","stem","nltkstemmer"},
                   "tokenizer":{"tokenizer","tokenize","nltktokenizer"}
                  }
ngramThesaurus = {"n":{"lengths","ngramlengths","orders"},
                  "maxNgramStopwords":{"maxngramstopwords","maxstopwordsperngram","maxstopwords","maxstop"},
                  "maxNgramStopwordProportion":{"maxngramstopwordproportion","maxstopwordproportion",
                                                "maxstopproportion"},
                  "beginToken":{"begintoken","starttoken","begindocumenttoken","startdocumenttoken",
                                "begindoctoken","startdoctoken","begintag","starttag"},
                  "endToken":{"endtoken","stoptoken","enddocumenttoken","enddoctoken","endtag","stoptag"},
                  "joinChar":{"joinchar","joincharacter","joinon"},
                  "join":{"join","joinngrams","joinedngrams","joined"},
                  "stemNgrams":{"stemngrams","stem"}}
opThesaurus = {"operation":{"operation","op","function"},
               "args":{"args","arguments","params","parameters","config","configuration"}
               }
operationThesaurus = {"replace":{"replace","substitute","sub","replaceregex","repl",
                            "substituteregex","subregex"},
               "strip":{"strip","stripchars","peel","peeloff","peelchars"},
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

patternSynonyms = {"pattern","regex","regexp","regularexpression"}
kwargSynonyms = {"kwargs","keywordarguments","keywordargs","arguments","args"}

filterThesaurus = {"pattern":patternSynonyms,
                   "match":{"match","matchtype"}
                   }

caseThesaurus = {"exclude":{"exclude","skip","not","negate"}}
caseThesaurus.update(filterThesaurus.copy())

matchThesaurus = {"full":{"full","fullmatch","completematch","totalmatch"},
                  "partial":{"partial","partialmatch","contains"}
                  }

argThesauri = {"replace":{"pattern":patternSynonyms,
                          "repl":{"repl","replacement","substitute","replacementstring"}
                          },
               "strip":{"chars":{"chars","characters","charlist","characterlist"}},
               "split":{"pattern":patternSynonyms,
                        "keep":{"keep","keepsplits","keepregexes","keepsplit","retainsplits","retainsplit"}
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


