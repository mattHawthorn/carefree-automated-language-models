from .tfidf import *
from .objects import BagOfWords
from .corpus import Document, BagOfWordsCorpus
from .models import NgramModel
from .utils import Memoized
from operator import itemgetter
from collections import defaultdict
from numpy.random import random, normal
from numpy import sqrt, log


class kNNTextClassifier:
    def __init__(self,k,hashlength,dfweighting=IDF,tfweighting=sublinearTF,
                 knnweighting=lambda x: 1,random=normal):
        self.k = k
        self.dfweighting = dfweighting
        self.tfweighting = Memoized(tfweighting)
        self.knnweighting = knnweighting
        self.docCount = None
        self.buckets = [[] for i in range(2**hashlength)]
        self.DF = None
        self.IDF = None
        self._random = random
        self._L = hashlength
        # powers of 2: dotting with the hash value generates the bucket list index
        self._decodeVec = []
        if self._L > 0:
            self._decodeVec.append(1)
            for i in range(1,self._L):
                self._decodeVec.append(2*self._decodeVec[i-1])
        
    def train(self,docs,classFunction,features=None,keepIDs=False,minFeatures=1,DF=None):
        # Determine how to iterate over docs; if list, tuple, or set, do nothing
        if type(docs) is dict:
            docs = docs.values()
        elif type(docs) is BagOfWordsCorpus:
            if type(docs.docs) is list:
                docs = docs.docs
            elif type(docs.docs) is dict:
                docs = docs.docs.values()
        # At this point if the type of docs hasn't been caught, we assume it's an iterable of
        # calm.corpus.Document objects

        self.classFunction = classFunction
        
        # allocate a list for the incoming docs
        newDocs = [None]*len(docs)
        # functions to get doc class and ID 
        # (if specified; need the ID's to pull the original docs text from the corpus later if desired)
        cls = self.classFunction
        docID = lambda doc: doc.ID if keepIDs else lambda doc: None
        
        # take in new BagOfWords objects, keeping only desired features and trimming out short docs
        i = 0
        for doc in docs:
            if features:
                bagOfWords = {ID:count for ID,count in doc.bagOfWords.items() if ID in features}
            else:
                bagOfWords = doc.bagOfWords
            # only add a doc to the model if it has more than minFeatures features
            if len(bagOfWords) >= minFeatures:
                newDocs[i] = (bagOfWords,cls(doc),docID(doc))
                i+=1
        
        # if no DF dictionary was given, build it
        if not DF:
            self.DF = BagOfWords()
            for doc in newDocs:
                self.DF.addBagOfWords(doc[0],count=1)
        else:
            self.DF = DF
        
        # keep only the non-null entries; pop off the tail end of the preallocated list
        for j in range(i,len(newDocs)):
            newDocs.pop()
        
        # need this for the numerator in IDF computations
        self.docCount = len(newDocs)
            
        # DF counting is complete; now we can compute IDF's
        idf = self.dfweighting
        self.IDF = {}
        for ID,count in self.DF.items():
            self.IDF[ID] = idf(count,self.docCount)
        
        # generate the random hash vectors
        self._hashVecs = self.randomVecs()
        
        # weight and normalize the bag of words vectors now that DF's are complete
        # store the normalized vectors with their classes in the buckets
        while newDocs:
            bagOfWords,c,ID = newDocs.pop()
            vec = self.normalize(self.weight(bagOfWords))
            # then hash them to buckets
            hashcode = self.hash(vec)
            index = self.bucketIndex(hashcode)
            self.buckets[index].append((vec,c))
            
    
    def classify(self,bagOfWords,confidence=False):
        if type(bagOfWords) is Document:
            bagOfWords = bagOfWords.bagOfWords
        
        classcounts = self.classCounts(bagOfWords)
        pred = max(classcounts.items(),key=itemgetter(1))
        
        if confidence:
            return (pred[0],pred[1]/sum(list(classcounts.values())))
        else:
            return pred[0]
            
    
    def classCounts(self,bagOfWords):
        knn = self.query(bagOfWords,self.k)
        classes = set([result[1] for result in knn])
        classcounts = {c:0 for c in classes}
        for cos,c in knn:
            classcounts[c]+=self.knnweighting(cos)
        
        return classcounts
    
    
    def query(self,bagOfWords,k):
        vec = self.normalize(self.weight(bagOfWords))
        bucket = self.buckets[self.bucketIndex(self.hash(vec))]
        # for all docs in the bucket, get cosine similarity and class
        sims = [(self.dot(vec,doc[0]),doc[1]) for doc in bucket]
        # sort by similarity descending
        sims = sorted(sims,key=itemgetter(0),reverse=True)
        k = min(k,len(sims))
        return sims[0:k]

    
    def hash(self,vec):
        return [0 if self.dot(vec,randomVec) < 0 else 1 for randomVec in self._hashVecs]

        
    def bucketIndex(self,hashcode):
        if self._L==0:
            return 0
        return sum([hashcode[i]*self._decodeVec[i] for i in range(self._L)])

        
    def dot(self,vec1,vec2):
        keys = set(vec1).intersection(vec2)
        return sum([vec1[key]*vec2[key] for key in keys])

    
    def weight(self,vec):
        return {key:self.tfweighting(count)*self.IDF[key] for key,count in vec.items() if key in self.IDF}
        
        
    def normalize(self,vec):
        norm = sqrt(self.dot(vec,vec))
        return {key:value/norm for key,value in vec.items()}
    

    def randomVecs(self):
        V = len(self.IDF)
        randomVecs = []
        for i in range(self._L):
            randomVecs.append(dict(zip(self.IDF.keys(), self._random(size=V))))
        return randomVecs
        
        
    def __repr__(self):
        name = __name__+'.knnTextClassifier'
        call = '(k={},hashlength={},dfweighting={},tfweighting={},knnweighting={},random={})'.format(self.k,
                    self._L,self.dfweighting,self.tfweighting,self.knnweighting,self._random)
        return name+call
        
        
class NgramBayesTextClassifier:
    def __init__(self,n,smoothing='additive',smoothing_param=0.1):
        self.n = n
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param
    
    
    def train(self,docs,classFunction,features=None):
        # Determine how to iterate over docs; if list, tuple, or set, do nothing
        if type(docs) is dict:
            docs = docs.values()
        elif type(docs) is BagOfWordsCorpus:
            if type(docs.docs) is list:
                docs = docs.docs
            elif type(docs.docs) is dict:
                docs = docs.docs.values()
        # At this point if the type of docs hasn't been caught, we assume it's an iterable of
        # calm.corpus.Document objects
        
        self.classFunction = classFunction
        docsByClass = defaultdict(list)
        
        for doc in docs:
            docsByClass[classFunction(doc)].append(doc)
        
        classCounts = {c:len(docList) for c,docList in docsByClass.items()}
        self.logClassProbs = {c:log(count/len(docs)) for c,count in classCounts.items()}
        
        self.classModels = {}
        for c,docList in docsByClass.items():
            model = NgramModel(n=self.n,smoothing=self.smoothing,param=self.smoothing_param,newTokens=1,keyDtype='uint32')
            model.train(docList)
            self.classModels[c] = model
        
    
    def classify(self,tokens,confidence=False):
        if type(tokens) is Document:
            tokens = tokens.tokens
            
        classProbs = [(c,self.logp(tokens,c)) for c in self.classModels]
        
        pred = max(classProbs,key=itemgetter(1))
        
        if confidence:
            return pred
        else:
            return pred[0]
        
        
    def logp(self,tokens,c):
        return self.logClassProbs[c] + self.classModels[c].p(tokens,logarithm=True)

