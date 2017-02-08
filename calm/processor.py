#coding: utf-8
import os
import re
from . import config_loader as cl
from .objects import BagOfWords,BagOfNgrams
from .utils import Memoized,MemoizedMethod,ngramIter,rollSum
from itertools import chain
from collections import deque
#from nltk import tokenize as nltkTokenizers


########################################
# TEXT PROCESSOR #######################
########################################

class Processor:
    """
    This can be initiated with a dict of keyword args, which can also be read from confilgFile if specified, either .json or .yml.
    kwargs given explicitly will override contents of the config file if both are given.
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
        stopwords: Either:
                1. A dict specifying one or both of: 1. a file or list of files containing stopwords, 2. an list of words to include.
                    ex: {'file':['foo.stop', 'bar.stop'], 'list':["dog","cat"]}
                   These are agglomerated into one global stoplist which is the default stoword set used by the 'stopwords' operation if
                   no stopwords are specified there. 
                2. A single string, which is assumed to be a local filepath with root the dir of the config file.
                3. A list of strings, which are assumed to be a list of stopwords.
        stemmer: A dict specifying a stemmer from the nltk stemmer module. Ex:
                {"name":"SnowballStemmer","kwargs":{"language":"english"}}
        tokenizer: A dict specifying a tokenizer from the nltk tokenizer module. Ex:
                {"name":"RegexpTokenizer","kwargs":{"pattern":"\\s+","gaps":true,"discard_empty":true}
                
    Possible operations included in the above sequence, together with their result and specification are as follows:
        replace: apply re.sub() to replace a regex with a string. Ex.: 
                {"operation":"replace","args":{"pattern":"[A-Z]{2,}","replacement":"ACRONYM"}}

        lower/upper: apply str.lower()/str.upper() to push an entire string to lower/uppercase., optionally only on a matched pattern
                (with a full match being the default). Ex.: 
                {"operation":"lower","args":{"pattern":"\s[A-Z]+\s","match":"partial"}}
                
        strip: apply str.strip(chars) to a string. Ex.:
                {"operation":"strip","args":{"chars":" -_.?!\\t\\n"}}

        split: apply re.split() to split a string on a regex. Optionally keep the matched split groups (discard by default)  Ex.: 
                {"operation":"split","args":{"pattern":"[\s]+","keep":True}}
                Note: use capturing parentheses on the pattern to keep the split groups.

        filter/retain: remove/keep tokens (input is assumed to be in tokenized form) according to a full/partial match with a regex
                (with a full match being the default) Ex.: 
                {"operation":"filter","args":{"pattern":"[0-9]+","match":"full"}}

        stopwords: remove stopwords.
                {"operation":"stopwords","args":{"file":"stopwords.txt"}}

        tokenize: apply a tokenizer from the nltk.tokenize module. Must specify the name as it appears in that module and give 
                keyword args for instantiation of an instance of that tokenizer. Ex.: 
                "args":{"name":"RegexpTokenizer","kwargs":{"pattern":"\\s+","gaps":true,"discard_empty":true}}
                Notes: this is optional, since many simple tokenization tasks can be accomplished with a regex using re.split().

        stem: apply a stemmer from the nltk module. Ex.: 
                {"operation":"stem","args":{"name":"SnowballStemmer","kwargs":{"language":"english"}}}
    """
    
    def __init__(self,configFile=None,stopwordsFiles=None,bow_constructor=BagOfWords,**kwargs):
        self.bow_constructor = bow_constructor
        self.__cache__ = dict() # for @MemoizedMethods
        
        # load the config file if specified
        if configFile:
            configDir = os.path.join(*os.path.split(configFile)[:-1])
            config = cl.load_config(configFile)

            if kwargs:
                config.update(kwargs)
            kwargs = config
        
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
        
        if 'ngramConfig' in kwargs:
            ngramConfig = cl.clean_args(kwargs['ngramConfig'],ngramThesaurus,remove=spaces)
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
                
            ngramConfig["n"] = sorted([int(n) for n in ngramConfig["n"]])
            ngramDefaults.update(ngramConfig)
            
        # add the ngram config params as class attributes
        self.ngramConfig = ngramDefaults
        self.__dict__.update(ngramDefaults)
        self._ngramJoiner = ngramJoiner(self.joinChar)
        
                
        self._stemmer = None
        if 'stemmer' in kwargs:
            params = cl.clean_args(kwargs['stemmer'],argThesauri['stem'])
            self.stemmer = params
            self._stemmer = loadStemmer(params['name'],**(params['kwargs']))
            self._stem = self._stemmer.stem
        
        self._tokenizer = None
        if 'tokenizer' in kwargs:
            params = cl.clean_args(kwargs['tokenizer'],argThesauri['tokenize'])
            self.tokenizer = params
            self._tokenizer = loadTokenizer(params['name'],**(params['kwargs']))
            self._tokenize = self._tokenizer.tokenize
        
        # read stopwords from a text file or directly as a list or set
        self.stopwords = None
        if stopwordsFiles:
            if 'stopwords' in kwargs:
                print("Warning: stopwords were specified in both the config file and the init args; "+
                      "defaulting to the init args stopwords: {}".format(stopwordsFile))
            self.stopwords = set(loadStopwords(files=stopwordsFiles, configDir=configDir))
        elif 'stopwords' in kwargs:
            stopconfig = kwargs['stopwords']
            if stopconfig is not None:
                if type(stopconfig) is str:
                    self.stopwords = set(loadStopwords(files=(stopconfig,), configDir=configDir))
                elif type(stopconfig) in [set,list,tuple]:
                    self.stopwords = set(stopconfig)
                elif type(stopconfig) is dict:
                    # assume a dict of the form passed to the stopwords op
                    stopconfig = cl.clean_args(stopconfig, argThesauri['stopwords'])
                    self.stopwords = set(loadStopwords(configDir=configDir, **stopconfig))
                else:
                    raise ValueError("Stopwords must be given as a filename, a list-like of strings, or a dict"+
                                     " object with 'files' and/or 'words' in the config, not {}".format(type(kwargs['stopwords'])))

        self.localStopwords = None
        # initialize the list of processing steps
        self._sequence = list()
        # track the config
        self.sequence = list(sequence)
        
        # for all operations in the processing sequence, define functions and append to the sequence
        for op in sequence:
            self.appendOp(op,configDir)
        
        
    def appendOp(self,op,configDir):
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
            if params.get('keep',False) in (True,'true','True'):
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
            if params is None:
                if not self.stopwords:
                    print("Warning: stopword removal is indicated, but no stopwords are specified either globally or locally for the operation")
                    stopwords = set()
                else:
                    stopwords = self.stopwords
            else:
                stopwords = set(loadStopwords(configDir=configDir, **params))
            
            if max(self.n) > 1:
                print("Warning: stopword removal is inidicated prior to ngram collection; ngrams will not reflect actual proximity.")
            
            if self.localStopwords is None:
                self.localStopwords = [stopwords]
            else:
                self.localStopwords.append(stopwords)
            args = (stopwords,)
            f = self.removeStopwords

        elif operation=='tokenize':
            if self._tokenizer is not None and params is not None:
                print("Warning: tokenizer was specified twice in the config; defaulting to the one defined in the operation sequence.")
            if params is not None:
                self.tokenizer = params
                self._tokenizer = loadTokenizer(params['name'],**(params['kwargs']))
                self._tokenize = self._tokenizer.tokenize
            if not self._tokenizer:
                raise ValueError("Tokenization is indicated, but no tokenizer is specified")
            f = self.tokenize
            args = ()

        elif operation=='stem':
            if self._stemmer is not None and params is not None:
                print("Warning: stemmer was specified twice in the config; defaulting to the one defined in the operation sequence.")
            if params is not None:
                self.stemmer = params
                self._stemmer = loadStemmer(params['name'],**(params['kwargs']))
                self._stem = self._stemmer.stem
            if not self.stemmer:
                raise ValueError("Stemming is indicated, but no stemmer is specified")
            f = self.stem
            args = ()
            
        else:
            raise KeyError("No operation exists for keyword {}".format(operation))

        self._sequence.append((f,args))


    # private functions for the heavy-lifting tasks behind the scenes
    def replace(self,strings,*args):
        return (re.sub(args[0],args[1],s) for s in strings)

    def split(self,strings,*args):
        return list(filter(lambda s: s is not None,chain.from_iterable([re.split(args[0],s) for s in strings])))
        
    def strip(self,strings,*args):
        return (s.strip(args[0]) for s in strings)

    def filter(self,strings,*args):
        p = args[0] # the regex pattern
        match = args[1] # the matching function
        if args[2]: # discard/retain
            return (s for s in strings if not match(p,s))
        else:
            return (s for s in strings if match(p,s))

    def case(self,strings,*args):
        f = args[0] # the case function
        return (f(s) for s in strings)

    def caseMatched(self,strings,*args):
        f = args[0] # the case function
        p = args[1] # the regex pattern
        match=args[2] # the matching function
        if args[3]: # only/except
            return (f(s) if not match(p,s) else s for s in strings)
        else:
            return (f(s) if match(p,s) else s for s in strings)

    def removeStopwords(self,strings,*args):
        stopwords = args[0]
        return (s for s in strings if s not in stopwords)

    def stem(self,strings):
        return (self._stem(s) for s in strings)

    def tokenize(self,strings):
        return chain.from_iterable((self._tokenize(s) for s in strings))


    # the main processing function: take a doc as a string and return a list of tokens
    def process(self,string,ngrams=False):
        if ngrams:
            return self.ngrams(string)
            # every processing function in self._sequence will be applied in order 
        # the the input and result of each intermediate step is a list
        if type(string) is str:
            tokens = [string]
        else:
            tokens = string

        for f,args in self._sequence:
            tokens = f(tokens,*args)
        
        return [t for t in tokens if t!='']
    
    __call__ = process

    
    def bagOfWords(self,tokens,ngrams=False):
        bag = BagOfWords()
        # if the argument is a raw string rather than a list of tokens, tokenize it
        #if type(tokens) is str:
        #    tokens = self.process(tokens,ngrams=ngrams)
        bag.addMany(tokens)
        return bag

    
    def bagOfNgrams(self,tokens):
        bag = BagOfNgrams(max_n=self.n[-1],joinchar = self.joinChar)
        # if the argument is a raw string rather than a list of tokens, tokenize it
        #if type(tokens) is str:
        #    tokens = self.process(tokens)
        bag.addMany(self.ngrams(tokens))
        return bag

    
    @MemoizedMethod
    def _ngramConfig(self,max_n):
        # Max allowable stopwords in the n-grams (may be adjusted later if self.maxNgramStopWordProportion is present)
        if self.maxNgramStopwords is not None:
            maxStopwords = self.maxNgramStopwords
        else:
            maxStopwords = max_n
        if self.maxNgramStopwordProportion is not None:
            maxStopwordProportion = self.maxNgramStopwordProportion
        else:
            maxStopwordProportion = 1.0

        discardStops = (maxStopwords < max_n or maxStopwordProportion < 1.0) and self.stopwords is not None
        beginStop = None
        endStop = None
        
        pad = (1 if self.beginToken else 0) + (1 if self.endToken else 0)
        
        return (max_n,pad,maxStopwords,maxStopwordProportion,discardStops)
    
    
    @MemoizedMethod
    def _noStopNgramGen(self,*args):
        n,max_n,maxStopwords,maxStopwordProportion = args
        maxStops = min(maxStopwords,int(maxStopwordProportion*n))
        
        if maxStops >= n:
            ngramGen = lambda tokens: (tup for tup in ngramIter(tokens,n))
        else:
            beginStop = (0 if self.beginToken else None)
            endStop = (0 if self.endToken else None)
        
            ngramGen = lambda tokens,isStopword: (tup for tup,num_stop in zip(ngramIter(tokens,n,start=self.beginToken,end=self.endToken),
                                                                          rollSum(isStopword, n, start=beginStop, end=endStop))
                                                   if num_stop <= maxStops)
        return ngramGen
        
        
    # take a string or a list of tokens (such as would be produced by self.process, and return a bag of words
    def ngrams(self,tokens,lengths=None):
        # PREPARATIONS
        if not lengths:
            lengths = self.n
        else:
            if type(lengths) not in {set,list,tuple}:
                lengths = [int(lengths)]
        
        # if the argument is a raw string rather than a list of tokens, tokenize it
        #if type(tokens) is str:
        #    tokens = self.process(tokens)
        
        # only stem at this stage (as opposed to the processor stage) if specified
        if self.stemNgrams:
            tokens = self.stem(tokens)
        
        # The greatest-length n-grams the tokens will accomodate:
        max_n = min(max(lengths),len(tokens))
        
        max_n,pad,maxStopwords,maxStopwordProportion,discardStops = self._ngramConfig(max_n)
                        
        # Only need to compute stopword occurrences once, and only if specified- a little more space but a lot less time.
        if discardStops:
            isStopword = [(1 if token in self.stopwords else 0) for token in tokens]
        
        # update max_n in case start and end tokens will be added
        max_n = min(max(lengths),len(tokens) + pad)
        
        # MAIN LOOP
        # Collect n-grams of all lengths in the list self.n
        # return types are consumable iterators/maps rather than lists since these could become large
        if not discardStops:
            if self.joinChar is None:
                return chain.from_iterable((tup for tup in ngramIter(tokens,n)) for n in lengths)
            else:
                return map(self._ngramJoiner,chain.from_iterable((tup for tup in ngramIter(tokens,n)) for n in lengths))
        else:
            if self.joinChar is None:
                return chain.from_iterable(self._noStopNgramGen(n,max_n,maxStopwords,maxStopwordProportion)(tokens,isStopword) for n in lengths)
            else:
                return map(self._ngramJoiner,chain.from_iterable(self._noStopNgramGen(n,max_n,maxStopwords,maxStopwordProportion)(tokens,isStopword) for n in lengths))
        
        
    def composeLeft(self,processor,args=None):
        """
        Prepend another tokenizer/processor to this one as an initial processing stage.
        processor: any function which takes optionally a string or an iterable of strings and returns an iterable of strings.
                   if you desire to compose a calm.processor.Processor P, then pass in P.process for 'processor'
        args: optionally, pass in a tuple of fixed args to be passed to 'processor'
        """
        if args is None:
            args = ()
        self._sequence = [(processor,args)] + self._sequence
        self.sequence = [dict(op=processor, args=args)] + self.sequence
        
        
    def composeRight(self,processor,args=None):
        """
        Append another tokenizer/processor to this one as an post-processing stage.
        processor: any function which takes optionally a string or an iterable of strings and returns, well, anything you like!
                   (though keep in mind that if processor returns something other than a string or iterable of strings, further
                   right composition with calm.processor.Processor objects will be broken)
                   If you desire to compose a calm.processor.Processor P, then pass in P.process for 'processor'
        args: optionally, pass in a tuple of fixed args to be passed to 'processor'.  Unpacked with *, as that is
            the custom with all other (f,args) tuples in self._sequence, so the processor function should be structured as such.
        """
        if args is None:
            args = ()
        self._sequence.append((processor,args))
        self.sequence.append(dict(op=processor, args=args))
        
        
    def __add__(self,processor):
        p1 = Processor(**{k:value for k,value in self.__dict__.items() if not k.startswith('_') and k not in self.ngramConfig})
        # copying p2 ensures that mutation will not affect the composed processor
        p2 = Processor(**{k:value for k,value in processor.__dict__.items() if not k.startswith('_') and k not in processor.ngramConfig})
        p1.composeRight(p2.process)
        return p1
        
        
    def __iadd__(self,processor):
        self.composeRight(processor.process)


def ngramJoiner(joinchar=None):
    if joinchar is None:
        def joinNgram(tokens):
            return tokens
    else:
        def joinNgram(tokens):
            return joinchar.join(tokens)
    return joinNgram


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
        

def loadStopwords(files=None, words=None, configDir=None):
    all_stopwords = set()
    if files is not None:
        if type(files) is str:
            files = (files,)
        for stopwordsFile in files:
            if configDir is not None and os.path.abspath(stopwordsFile) != stopwordsFile:
                stopwordsFile = os.path.join(configDir, stopwordsFile)
            if os.path.splitext(stopwordsFile)[1] in cl.legalConfigExtensions:
                stopwords = set(cl.load_config(stopwordsFile))
            else:
                stopwords = set()
                with open(stopwordsFile,'r') as infile:
                    for line in infile:
                        stopwords.add(line.strip())
            all_stopwords = all_stopwords.union(stopwords)
    if words is not None:
        all_stopwords = all_stopwords.union(words)
    return all_stopwords



#########################################
# THESAURI FOR PROCESSOR CONFIG FILES ###
#########################################

# filter for comparing keywords
spaces = re.compile('[-_\s]')

# thesauri for translating config
configThesaurus = {"sequence":{"sequence","functions","ops","operations",
                                 "functionsequence","operationsequence","functionorder",
                                 "operationorder","orderofoperations"},
                   "ngramConfig":{"ngrams","ngram","ngramconfig","ngramparameters","ngramparams"},
                   "stopwords":{"stopwords"},
                   "stemmer":{"stemmer","stem","nltkstemmer"},
                   "tokenizer":{"tokenizer","tokenize","nltktokenizer"}
                  }
ngramThesaurus = {"n":{"lengths","ngramlengths","orders","ns"},
                  "maxNgramStopwords":{"maxngramstopwords","maxstopwordsperngram","maxstopwords","maxstop"},
                  "maxNgramStopwordProportion":{"maxngramstopwordproportion","maxstopwordproportion",
                                                "maxngramstopwordsproportion","maxstopwordsproportion",
                                                "maxstopproportion"},
                  "beginToken":{"begintoken","starttoken","begindocumenttoken","startdocumenttoken",
                                "begindoctoken","startdoctoken","begintag","starttag","begin"},
                  "endToken":{"endtoken","stoptoken","enddocumenttoken","enddoctoken","endtag","stoptag","end"},
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
               "stopwords":{"files":{"file","filename","path","stopwordsfile","files","paths","filenames", "stopwordsfiles"},
                            "words":{"list","stopwords","stopwordslist","stoplist","words"}
                            },
               "tokenize":{"name":{"name","tokenizer","nltktokenizer","tokenizername","nltktokenizername"},
                           "kwargs":kwargSynonyms
                           },
               "stem":{"name":{"name","stemmer","nltkstemmer","stemmername","nltkstemmername"},
                       "kwargs":kwargSynonyms
                       }
               }


