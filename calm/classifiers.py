from .functions import *
from .objects import BagOfWords
from operator import itemgetter
from numpy.random import random, normal


class kNNTextClassifier:
    def __init__(self,k,hashlength,dfweighting=IDF,tfweighting=sublinearTF,
                 knnweighting=lambda x: 1,random=normal):
        self.k = k
        self.dfweighting = dfweighting
        self.tfweighting = tfweighting
        self.docCount = 0
        self.buckets = [[]]*(2**hashLength)
        self.DF = BagOfWords()
        self.IDF = {}
        self._L = hashLength
        self._hashVecs = [{}]*hashlength
        self._decodeVec = [1]*hashlength
        self._random = random
        for i in range(1,len(self._decodeVec)):
            self._decodeVec[i]=2*self._decodeVec[i-1]
        
    def train(self,docs,classFunction,features=None):
        self.classFunction = classFunction
        self.docCount = len(docs)
        # take in new BagOfWords objects, keeping only desired features and building up DF's as you go
        newDocs = [None]*len(docs)
        i = 0
        for doc in docs:
            bagOfWords = {ID:count for ID,count in doc.bagOfWords.items() if ID in features}
            newDocs[i] = (bagOfWords,self.classFunction(doc))
            i+=1
            self.DF.addBagOfWords(bagOfWords,count=1)
        
        # DF counting is complete; now we can compute IDF's
        for ID,count in self.DF.items():
            self.IDF[ID] = self.dfweighting(count,self.docCount)
        
        # generate the random hash vectors
        self._hashVecs = self.randomVecs()
        
        # weight and normalize the bag of words vectors now that DF's are complete
        # store the normalized vectors with their classes in the buckets
        for bagOfWords,c in newDocs:
            vec = self.normalize(self.weight(bagOfWords))
            # then hash them to buckets
            hashcode = self.hash(vec)
            index = self.bucketIndex(hashcode)
            self.buckets[bucketIndex].append((vec,c))
            
    
    def classCounts(self,bagOfWords):
        vec = self.normalize(self.weight(bagOfWords))
        bucket = self.bucketIndex(self.hash(vec))
        knn = self.query(vec,bucket)
        classes = set([c for cos,c in knn])
        classcounts = {c:0 for c in classes}
        for cos,c in knn:
            classcounts[c]+=self.knnweighting(cos)
            
    
    def classify(self,bagOfWords):
        classcounts = self.classCounts(bagOfWords)
        max = 0
        pred = None
        for c,count in classcounts.items():
            if count > max:
                max = count
                pred = c
        return pred
    
    
    def query(self,normedVec,bucket):
        # normedVec is assumed normalized. Compute list of tuples (similarity,class) 
        # for all docs in the bucket
        sims = [(self.dot(normedVec,doc[0]),doc[1]) for doc in self.buckets[bucket]]
        # sort by similarity descending
        sims = sorted(sims,key=itemgetter(0))
        k = min(self.k,len(sims))
        return sims[0:k]

    
    def hash(self,vec):
        return [0 if self.dot(vec,randomVec) < 0 else 1 for randomVec in self.randomVecs]

        
    def bucketIndex(self,hashcode):
        return sum([hashcode[i]*self._decodeVec[i] for i in range(self._L)])

        
    def dot(self,vec1,vec2):
        keys = set(vec1).intersection(vec2)
        return sum([vec1[key]*vec2[key] for key in keys])

    
    def weight(self,vec):
        return {key:self.termweighting(count)*self.IDF[key] for key,count in vec.items()}
        
        
    def normalize(self,vec):
        norm = self.dot(vec,vec)
        return {key:value/norm for key,value in vec.items}
    

    def randomVecs(self):
        V = len(self.IDF)
        randomVecs = []
        for i in range(self._L):
            randomVecs.append(dict(zip(IDF.keys(), self._random(size=V))))
        return randomVecs
        
