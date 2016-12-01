# coding: utf-8

import numpy as np
from numpy import sort, cumsum, argsort, searchsorted, where, array
from .objects import *
from .corpus import *
from .distributions import head
from copy import copy, deepcopy
#from gensim.models import HdpModel, LdaModel, LdaMulticore
from collections import Mapping, Sequence

default_lda_params = dict(doc_topic_prior=0.01, topic_word_prior=0.001, optimize_doc_topic_prior=False)

class TopicModelWrapper:
    """
    Any subclass mus implement the fit, _assignBagOfWords, and _assignBagOfIDs methods
    """
    def __init__(self, implementation, **params):
        """
        implementation: one of 'gensim', 'mallet'
        params: a dict (or individually passed named params) mapping parameter name to value
            for parameters relevant to the underlying implementation.
        params shared by all implementations are:
            num_topics: int, number of topics to fit (not necessary for HDP model)
            doc_topic_prior: float or vector of float of length num_topics. Usually called 'alpha' in the lit.
            topic_word_prior: float, also called 'eta' in some lit
            optimize_doc_topic_prior: boolean; should the doc_topic_prior be optimized during fitting?
        """
        # update with defaults for args not passed
        for param,value in default_lda_params.items():
            if param not in params:
                params[param] = value
        
        # store params as original dict of kwargs, and as attributes
        self.params = params
        
        for param,value in params.items():
            setattr(self, param, value)
    
    def fit(self, corpus):
        """
        Fit a model using the underlying implementation, with the params given in the __init__.
        If the corpus is a calm.corpus.BagOfWordsCorpus, the model will be fit using the bags of words
           attached to each doc in corpus.docs
        Otherwise, the corpus is assumed to be an iterable of bag-of-words mappings.
        
        Should supply num_iter: int; number of iterations over the corpus
        
        At the end of fitting, the following traits should be present:
            num_topics: int specifying the number of topics in the mixture
            num_terms: int specifying the vocab size
            topic_distributions: a 2D array of shape (num_topics, num_terms) with word distributions in each row.
            terms: a vector of str of shape (num_terms,)
            alpha: a vector of shape (num_topics,) which represents the alpha (prevalence) for each topic
            vocab: a calm.corpus.Vocabulary (two-way mapping of token to int ID and vice-versa).
                this is inherited from the corpus in the instance that the corpus is a calm.corpus.BagOfWordsCorpus.
            num_iter: number of iterations that have been carried out so far
        """
        pass
    
    def _assignBagOfWords(self, bow):
        """
        returns array of float (topic weights)
        """
        pass
    
    def _assignBagOfIDs(self, bow):
        """
        returns array of float (topic weights)
        """
        pass
    
    def _assignListOfTokens(self, tokens):
        """
        returns array of float (topic weights)
        """
        return self._assignBagOfWords(BagOfWords(tokens))
        
    def _assignListOfIDs(self, ids):
        """
        returns array of float (topic weights)
        """
        return self._assignBagOfIDs(BagOfWords(ids))
    
    def topicSummary(self, topicID, num_words=None, total_prob=None):
        """
        topicID is an int referencing a specific topic in the topic model
        returns array of str (words), array of float (weights)
        """
        dist = self.topic_distributions[topicID,]
        indices = head(dist, p=total_prob, n=num_words)
        return self.words[indices], dist[indices]
    
    def topicLabel(self, topicID, num_words=10, probs=True, digits=2):
        words, weights = self.topicSummary(topicID, num_words=num_words)
        weights = (round(w,digits) for w in weights)
        
        return ", ".join(" ".join(tup) for tup in zip(words, weights))
        
    def docTopics(self, doc, decode=False):
        """
        returns array of str (topic summaries), array of float (weights) for a doc,
            which may be either:
            a mapping from token/ID to count (calm.object.BagOfWords or simple dict)
            a sequence of tokens/IDs (list or tuple)
        """
        if isinstance(doc, (BagOfWords, Mapping)):
            if not decode:
                assigner = self._assignBagOfWords
            else:
                assigner = self._assignBagOfIDs
        elif isinstance(doc, (Sequence)):
            if not decode:
                assigner = self._assignListOfTokens
            else:
                assigner = self._assignListOfIDs
        elif isinstance(doc, Document):
            assigner = self._assignBagOfIDs
            doc = doc.bagOfWords
        
        return assigner(doc)
    
    def docTopicSummary(self, doc, total_prob=None, min_prob=None, decode=False):
        dist = self.docTopics(doc, decode=decode)
        indices = head(weights, p=total_prob, eps=min_prob)
        return indices[0], dist[indices]


class GensimLDAModel(TopicModelWrapper):
    def __init__(self, **params):
        try:
            from gensim.models import LdaModel
            self._implementation = LdaModel
        except ImportError:
            raise ImportError("gensim is not installed or gensim.models.LdaModel is unavailable.")
        
        super(GensimLDAModel, self).__init__(self,**params)
    
    def fit(self, corpus, num_iter=None):
        if num_iter is None:
            num_iter = self.num_iter
            
        docs = corpus.docIter(return_ids=False, transform=lambda doc: list(doc.bagOfWords.items()))
        # copy this so that mucking around with the corpus won't cause problems
        self.vocab = corpus.vocab
        self.num_terms = len(self.vocab)
        
        # the model is fit at the time of init
        kwargs = dict(corpus=docs, 
                     num_topics=self.num_topics,
                     alpha=self.doc_topic_prior if not self.optimize_doc_topic_prior else 'auto',
                     eta=self.topic_word_prior,
                     id2word = self.vocab.token,
                     iterations=self.num_iter)
        #print("call is LdaModel({})".format(", ".join("{}={}".format(*tup) for tup in kwargs.items() if tup[0] != 'id2word')))
        
        # the implementation here is the gensim LDA model class
        self._model = self._implementation(**kwargs)
        
        self.alpha = self._model.alpha
        self.eta = self._model.eta

