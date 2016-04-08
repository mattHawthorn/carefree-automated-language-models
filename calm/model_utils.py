from numpy import sum, log, array, zeros


def IG(classAttrArray):
    """
    Information gain of an attribute for a set of classes.
    Classes are in the rows (1st index) and attributes are in the columns.
    Entries are counts of the attribute value given the class
    """
    # counts of each class
    #classcounts = [sum[row] for row in classAttrTable]
    classcounts = classAttrArray.sum(axis=0)
    
    # attribute counts
    attrcounts = classAttrArray.sum(axis=1)
    
    # total count
    total = sum(classcounts)
    
    # MLE of class probabilities; vector operation divide by total
    p_classes = classcounts/total
    
    # attribute counts conditional on classes; 
    # divide each column by the total for that class to get P(class|attr)
    p_classes_conditional = classAttrArray/attrCounts
    
    # entropy of classes
    class_entropy = -1*p_classes*log(p_classes)
    
    # conditional entropy of classes on attributes
    
    


def chisquared(classAttrArray):
    """
    Chi-squared statistic for an attribute and a set of classes.
    Classes are in the rows (1st index) and attributes are in the columns.
    Entries are counts of the attribute value given the class
    """
    
    
def classAttrArray(corpus,classFunction,numClass,attrFunction,numAttr):
    """
    classFunction and attrFunction should both take a doc in corpus and return an int.
    Care must be taken to assure that these functions return ints in the ranges
    0:numClass-1 and 0:numAttr-1, since these will be used as array indices on a preallocated array
    """
    # corpus is iterable over docs
    A = zeros(shape=(numClass,numAttr),dtype='float')
    for doc in corpus:
        A[classFunction(doc),attrFunction(doc)] += 1.0
        
    return A
