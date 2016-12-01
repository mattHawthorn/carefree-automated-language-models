#coding:utf-8

from matplotlib import pyplot as plt
from matplotlib.colors import cnames
from itertools import cycle
from math import ceil, floor
from datetime import datetime, date

from calm.objects import BagOfWords
from calm.corpus import BagOfWordsCorpus
from datetime import datetime, date, time, timedelta

EPOCH = datetime(1970,1,1)

defaultcolors = ["blue",
                "red",
                "green",
                "orange",
                "black",
                "lightblue",
                "salmon",
                "yellowgreen",
                "gold",
                "gray",
                "midnightblue",
                "firebrick",
                "darkolivegreen",
                "darkgoldenrod",
                "lightgray"]


def getTotalWords(doc):
    return doc.bagOfWords.total

class TokenTimeSeriesAnalyzer:
    def __init__(self,corpus,time_field,converter = lambda x:x):
        self.corpus = corpus
        self.time_field = time_field
        
        if type(converter) is dict:
            if "default" not in converter:
                converters = converter.copy()
                converters["default"] = next(iter(converter.values()))
            self.converters = converters
        elif callable(converter):
            self.converters = dict(default=converter)
        else:
            raise ValueError("converter must be a callable or dict of callables, transforming the docs' time_field attribute to a plottable quantity.")
        
        self.allCounts = {}
        for converter, time_f in self.converters.items():
            self.allCounts[converter] = self.bagOfTimes(converter, counter=lambda doc: doc.bagOfWords.total)
    
    def bagOfTimes(self, converter="default", counter=getTotalWords):
        bag_of_times = BagOfWords()
        bag_of_times._addmanyCounts((t, counter(doc)) for doc,t in self.docTimeIter(converter))
        return bag_of_times
    
    def docTimeIter(self, converter="default"):
        time_field = self.time_field
        
        if not callable(converter):
            converter = self.converters[converter]
        
        for doc in self.corpus:
            if doc.__hasattr__(time_field):
                if doc[time_field] == None:
                    continue
                yield doc, converter(doc[time_field])
            else:
                continue
    
    def series(self, token, converter="default", normalize=True, min_time=None, max_time=None, label = None, docfilter=None):
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
        if docfilter is not None:
            validate = lambda doc: docfilter(doc) and validate(doc)
            
        if label is None:
            label = token
            
        tokenID = self.corpus.vocab.ID[token]
        counts = ((t, doc.bagOfWords[tokenID]) for doc,t in self.docTimeIter(converter) if validate(doc, tokenID, t))
        token_counts = BagOfWords()
        token_counts._addmanyCounts(counts)
        
        if len(token_counts) > 0:
            if not callable(converter):
                allCounts = self.allCounts[converter]
            else:
                allCounts = self.bagOfTimes(converter, counter=lambda doc: doc.bagOfWords.total)
            
            if normalize:
                token_counts = {time:count/allCounts[time] for time,count in token_counts.items()}
            
            return tuple(zip(*sorted(list(token_counts.items()))))
        else:
            return ((),())
        
    def compareTokens(self, tokens, converter="default", 
                      normalize=True, min_time=None, max_time=None, title=None, labels=None, colors=defaultcolors):
        if labels is None:
            labels = tokens
            
        series_iter = (self.series(token, converter=converter, 
                                   normalize=normalize, 
                                   min_time=min_time, max_time=max_time) for token in tokens)
        self.compareSeries(series=series_iter, labels=labels, title=title, colors=colors)

    def compareSeries(self, series, labels=None, title=None, colors=defaultcolors, show=False):
        if labels is None:
            labels = map(str, range(len(series)))
        maxcount = 0.0
        fig, ax = plt.subplots()
        
        for color, label, (times, counts) in zip(cycle(colors), labels, series):
            maxcount = max(maxcount, max(counts))
            ax.plot(times,counts,color=color,label=label)
            
        ax.set_ylim(0,1.1*maxcount)
        if title is not None:
            ax.set_title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.xticks(rotation="vertical")
        if show:
            plt.show()
        
    def plot(self, token, converter="default", normalize=True, min_time=None, max_time=None, title=None, color=None):
        times,counts = self.series(token, converter=converter, normalize=normalize, min_time=min_time, max_time=max_time,show=False)
        fig, ax = plt.subplots()
        #ax.xaxis.set_major_locator(mdates.YearLocator)
        ax.set_ylim(0,1.1*max(counts))
        ax.plot(times,counts,color=color)
        if title is None:
            title = token
        else:
            title = " - ".join(token,title)
        ax.set_title(title)
        plt.xticks(rotation="vertical")
        if show:
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

