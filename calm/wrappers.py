#coding:utf-8
from .corpus import Vocabulary
from math import floor,ceil
import os

"""
Wrappers for third party software.
So far, David Blei's Dynamic Topic Model (DTM) is supported.
"""

def write_DTM_files(corpus,prefix='/tmp/dtm/dtm',datefunc=lambda doc:doc['date'],
                    minslice=1000,maxslices=100,slicefunc=None):
    """
    corpus: a calm.corpus.BagOfWordsCorpus
    prefix: the path prefix to the dtm input files.  The files <prefix>mult.dat and <prefix>seq.dat will be written,
            as well as <prefix>-voc.dat.  Default is '/tmp/dtm-'
    datefunc: takes a calm.corpus.Document and returns a datetime.date object. Defaults to lambda doc:doc['date']
    minslice: the minimum number of documents you're willing to put in a timeslice for the DTM. Default is 1000
    maxslices: the maximum number of slices you're willing to give to the DTM. Default is 100
    slicefunc: takes a document's date and its index in sorted temporal order, and returns the index of the time slice
            for the document. If this is None (default), it is constructed from the args minslice, maxslices
    """
    
    numdocs = len(corpus.docs)
    if slicefunc is None:
        slicefunc = make_slicefunc(numdocs,minslice,maxslices)
    
    output_dir = os.path.split(prefix)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    vocab_path = prefix + '-voc.dat'
    bow_path = prefix + '-mult.dat'
    timeslice_path = prefix + '-seq.dat'
    dates_path = prefix + '-dates.dat'
    vocab_order = Vocabulary()
    
    # "words" are original ID's, "ID's" are int indices for rows in the vocab file
    vocab_order.add(corpus.vocab.token.keys())
    
    def oldID(newID):
        return vocab_order.token[newID]
    def newID(oldID):
        return vocab_order.ID[oldID]
    
    with open(vocab_path,'w') as outfile:
        for newid in range(len(vocab_order.ID)):
            # the actual token for the id for the new id
            outfile.write('{}\n'.format(corpus.vocab.token[oldID(newid)]))
            
    sorted_docs = sorted(corpus.docs.values(),key=datefunc)
    lastdate = sorted_docs[-1]['date']
    num_slices = slicefunc(lastdate,len(sorted_docs) - 1) + 1
    timeslices = [0]*num_slices
    
    with open(bow_path,'w') as outfile:
        for idx,doc in enumerate(sorted_docs):
            date = datefunc(doc)
            timeslice = slicefunc(date,idx)
            timeslices[timeslice] += 1
            
            bow = doc.bagOfWords
            outfile.write(str(len(bow)))
            for oldid,count in bow.items():
                outfile.write(' {}:{}'.format(newID(oldid),count))
            
            outfile.write('\n')
                
    with open(timeslice_path,'w') as outfile:
        outfile.write('{}\n'.format(len(timeslices)))
        for numdocs in timeslices:
            outfile.write('{}\n'.format(numdocs))
            
    with open(dates_path,'w') as outfile:
        idx = 0
        for numdocs in timeslices:
            outfile.write("{} {}\n".format(datefunc(sorted_docs[idx]),datefunc(sorted_docs[idx+numdocs-1])))
            idx += numdocs

def make_slicefunc(numdocs,minslice,maxslices):
    numslices = float(numdocs)/float(minslice)
    if ceil(numslices) > maxslices:
        minslice = float(numdocs)/float(maxslices)
    else:
        minslice = float(numdocs)/float(floor(numslices))
    slicefunc = lambda date,idx: floor(float(idx)/minslice)
    return slicefunc

