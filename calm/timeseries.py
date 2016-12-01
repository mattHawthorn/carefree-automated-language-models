#coding:utf-8

from matplotlib import pyplot as plt
from math import ceil, floor
from datetime import datetime, date
from calm.objects import BagOfWords
from calm.corpus import BagOfWordsCorpus


class TokenTimeSeriesAnalyzer:
    def __init__(self,corpus,time_field,converter = lambda x:x):
        self.corpus = corpus
        self.time_field = time_field
        self.converter = converter
        
        self.allCounts = BagOfWords()
        counts = ((t, doc.bagOfWords.total) for doc,t in self.docTimeIter())
        self.allCounts._addmanyCounts(counts)
        
    def docTimeIter(self):
        time_field = self.time_field
        converter = self.converter
        
        for doc in self.corpus:
            if doc.__hasattr__(time_field):
                if doc[time_field] == None:
                    continue
                yield doc, converter(doc[time_field])
            else:
                continue
    
    def series(self, token, normalize = True, min_time=None, max_time=None):
        if min_time is None:
            if max_time is None:
                validate = lambda doc, token, t: tokenID in doc.bagOfWords
            else:
                validate = lambda doc, token, t: tokenID in doc.bagOfWords and t <= max_time
        else:
            if max_time is None:
                validate = lambda doc, token, t: tokenID in doc.bagOfWords and min_time <= t
            else:
                validate = lambda doc, token, t: tokenID in doc.bagOfWords and min_time <= t and t <= max_time
        
        tokenID = self.corpus.vocab.ID[token]
        counts = ((t, doc.bagOfWords[tokenID]) for doc,t in self.docTimeIter() if validate(doc, tokenID, t))
        token_counts = BagOfWords()
        token_counts._addmanyCounts(counts)
        
        if len(token_counts) > 0:
            allCounts = self.allCounts
            if normalize:
                token_counts = {time:count/allCounts[time] for time,count in token_counts.items()}
            return tuple(zip(*sorted(list(token_counts.items()))))
        else:
            return ((),())
        
    def plot(self, token, normalize=True, min_time=None, max_time=None):
        times,counts = self.series(token, normalize=normalize, min_time=min_time, max_time=max_time)
        fig, ax = plt.subplots()
        #ax.xaxis.set_major_locator(mdates.YearLocator)
        ax.set_ylim(0,1.1*max(counts))
        ax.plot(times,counts)
        plt.show()


def monthDT(dt):
    month = dt.month
    year = dt.year
    return datetime(year,month,1)

def yearDT(dt):
    year = dt.year
    return datetime(year,1,1)

def dayDT(dt):
    month = dt.month
    year = dt.year
    day = dt.day
    return datetime(year,month,day)

def groupByNumDays(dt, num_days=7):
    chunks = (dt - EPOCH).days//num_days
    return EPOCH + timedelta(days=chunks*num_days)
    
def groupByYearFraction(dt, num_per_year = 52):
    year = dt.year
    year_start = datetime(year)
    year_end = datetime(year + 1)
    chunk = (year_end - year_start)/num_per_year
    delta = dt - year_start
    return year_start + (delta//chunk)*chunk

