from numpy import sum, log, array, zeros, nan_to_num, apply_along_axis, transpose
from scipy.stats import entropy
from scipy.stats.distributions import chi2
from functools import reduce
from operator import itemgetter
from itertools import combinations
from random import shuffle
from scipy.stats import entropy, ttest_rel
from sklearn.metrics import precision_score, recall_score, f1_score
from pandas import DataFrame, Index, MultiIndex


def IG(classFeatureMatrix):
    """
    Information gain of an attribute for a set of classes.
    Classes are indexed by rows (1st index) and feature values are indexed by columns.
    Entries are counts of the feature value given the class.
    """
    # counts of each class: sum the rows
    classcounts = classFeatureMatrix.sum(axis=1)
    
    # attribute counts: 
    valuecounts = classFeatureMatrix.sum(axis=0)
    
    # total count of observations
    total = sum(classcounts)
    
    # MLE of class probabilities; vector operation divide by total
    p_classes = classcounts/total
    # MLE of feature value probabilities; vector operation divide by total
    p_values = valuecounts/total
    
    # feature value counts conditional on classes; 
    # divide each column by the total for that class to get P(class|feature value)
    p_classes_conditional = classFeatureMatrix/valuecounts
    
    # entropy of classes (vector log operation)
    class_entropy = entropy(p_classes)
    
    # conditional entropy of classes for each feature value
    cond_entropies = apply_along_axis(entropy,axis=0,arr=p_classes_conditional)
    
    # information gain: original entropy of classes minus conditional entropy of classes 
    # dotted with feature value probabilities
    return class_entropy-sum(p_values*cond_entropies)
    


def chisquared(classFeatureMatrix):
    """
    Chi-squared statistic for a feature and a set of classes.
    Classes are indexed by rows (1st index) and feature values are indexed by columns.
    Entries are counts of the feature value given the class.
    """
    # counts of each class: sum the rows
    classcounts = classFeatureMatrix.sum(axis=1)
    
    # attribute counts: 
    valuecounts = classFeatureMatrix.sum(axis=0)
    
    # total count of observations
    total = sum(classcounts)
    
    # expected observations under the independence assumption
    expected = array([valuecounts]*len(classcounts),dtype='float')
    expected = transpose(transpose(expected)*classcounts/total)
    
    # chi-squared statistic: (obs-expected)^2/expected where obs is classFeatureMatrix
    chi = sum(((classFeatureMatrix - expected)**2)/expected)
    
    # degrees of freedom
    df = (expected.shape[0]-1)*(expected.shape[1]-1)
    
    # the p-value; 1 minus the cdf of chi-squared at the given statistic value
    return 1-chi2(df=df).cdf(chi)
    
        
    
def classFeatureArray(docs,classFunction,numClasses,featureFunction,numValues):
    """
    classFunction and attrFunction should both take a doc in corpus and return an int.
    Care must be taken to assure that these functions return ints in the ranges
    0:numClass-1 and 0:numAttr-1, since these will be used as array indices on a preallocated array
    """
    # docs is an iterable of doc objects
    A = zeros(shape=(numClasses,numValues),dtype='float')
    for doc in docs:
        A[classFunction(doc),featureFunction(doc)] += 1.0
        
    return A
    

def PrecRecCurve(truth,preds,pos=1):
    """
    preds is presumed to be an iterable of tuples (ground_truth,prediction), where ground_truth is 0/1
    and prediction is on a continuous scale
    """
    # initialize lists, with appropriate values at the beginning
    prec = [None]*(len(preds)+1)
    rec = [None]*(len(preds)+1)
    prec[0] = rec[-1] = 0.0
    rec[0] = prec[-1] = 1.0
    
    # zip the continuous predictions together with the discrete true classes for sorting
    data = sorted(zip(preds,truth),key=itemgetter(0))
    
    # starting values for true/false pos/neg
    tn=0
    fn=0
    # everything positive starts as a true positive
    pos_count = reduce(lambda x,y:x+1 if y==pos else x,truth,0)
    tp = pos_count
    # false pos are the complement of this
    fp = len(truth) - tp
    
    # count true pos, false pos, true neg, false neg, and accumulate to prec and rec lists
    # remove the last entry to avoid division by zero on precision; we already know it's 0
    data.pop()
    i = 1
    for x in data:
        # x[1] is the true label from the zip
        if x[1] == pos:
            tp-=1; fn+=1
        else:
            fp-=1; tn+=1
        
        precision = tp/(tp+fp)
        recall = tp/(pos_count)
        prec[i] = precision
        rec[i] = recall
        i+=1
    return [prec,rec]
    

class kFolds():    
    def __init__(self,keys,k):
        if type(keys) is int:
            keys = list(range(keys))
        elif type(keys) in {list,tuple}:
            keys = list(range(len(keys)))
        elif type(keys) in {dict,set,range}:
            keys = list(keys)
        else:
            raise TypeError("kFolds does not support type {}".format(type(keys)))
        
        shuffle(keys)
        self.keys = keys
        self.k = k
        self.foldsize = float(len(keys))/k
        self._i = None
        
    def __iter__(self):
        self._i=0.0
        return self
        
    def __next__(self):
        low = int(round(self._i,8))
        if low>=len(self.keys):
            raise StopIteration
        high = int(round(self._i + self.foldsize,8))
        train = self.keys[0:low] + self.keys[high:len(self.keys)]
        test = self.keys[low:high]
        self._i+=self.foldsize
        return train, test
        
        
class CV:
    """
    Class for carrying out cross-validation of classifiers over a fixed set of folds on a fixed set of instances.
    This is data-oriented; the instances and folds are set at initialization.  Metrics, classifiers, and 
    features can then be set at will and new runs of self.validate will generate and append results for comparison.
    Any classFunction argument is assumed to 
    All metric functions are assumed to take the form metric(truth,predictions) where truth and predictions are lists of ints.
    All classifiers are assumed to have a .train(instances,classFunction,features) method, and a
    .classify(instance) method returning an int.
    """
    def __init__(self,instances,k,classFunction,features=None,metrics=None):
        self.instances = instances
        self.k = k
        self.folds = kFolds(instances,self.k)
        columns = MultiIndex(levels=[[],[]],labels=[[],[]],names=['metric','classifier'])
        rows = Index(range(self.k),name='fold')
        self.results = DataFrame(index=rows,columns=columns)
        self.classFunction = classFunction
        self.features = features
        self.metrics = metrics
        
        
    def validate(self,*classifiers,names=None,metrics=None,features=None,verbose=False):
        if not features:
            features = self.features
        if not metrics:
            if self.metrics:
                metrics = self.metrics
            else:
                raise ValueError("At least one metric function must be supplied")
        if not names:
            names = [repr(classifier) for classifier in classifiers]
            
        if len(names)!=len(classifiers):
            raise ValueError("If names are supplied, there must be as many names as classifiers")
            
        if verbose:
            print("Validating {} classifiers on {} folds:".format(len(classifiers),self.k))
        
        i = -1
        for train,test in self.folds:
            i+=1
            
            for classifier,classifierName in zip(classifiers,names):
                if verbose:
                    print("Fold {}: training {}.".format(i+1,classifierName))
                classifier.train([self.instances[key] for key in train],classFunction=self.classFunction,features=self.features)
                
                if verbose:
                    print("Fold {}: testing".format(i+1))
                preds = [classifier.classify(self.instances[key]) for key in test]
                truth = [self.classFunction(self.instances[key]) for key in test]
                
                for metric in metrics:
                    acc = metric(truth,preds)
                    if verbose:
                        print("{}: {}".format(repr(metric),acc))
                    self.results.loc[i,(repr(metric),classifierName)] = acc
                    
                if verbose:
                    print()
                    
        self.results.sort_index(axis=1,inplace=True)
        
        
    def paired_t_test(self):
        columns = Index([],name='metric')
        rows = MultiIndex(levels=[[],[]],labels=[[],[]],names=['classifier1','classifier2'])
        t_comparisons = DataFrame(index=rows,columns=columns)
        
        for metric in self.results.columns.levels[0]:
            results = self.results[metric]
            
            for classifierPair in combinations(results.columns,2):
                if (results[classifierPair[0]] - results[classifierPair[1]]).mean() < 0:
                    classifierPair = (classifierPair[1],classifierPair[0])
                    
                p = ttest_rel(results[classifierPair[0]].values,results[classifierPair[1]].values)[1]
                t_comparisons.loc[classifierPair,repr(metric)] = p
        
        return t_comparisons
