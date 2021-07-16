# This file contains custom vectorizers that can be used in the experiments in predictions.py.

# imports
import numpy as np
import pickle
import pandas as pd

# average word embeddings in each tweet, adapted from
# https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html
class AverageGlove:
  def __init__(self, embeddings):
    ''' 
    Args: 
      embeddings : path to glove embeddings
      vocab : path to pickled vocab
    '''
    # load embeddings, adapted from https://stackoverflow.com/questions/66380331/how-do-i-create-a-dictionary-from-a-txt-file-with-whitespace-in-python
    lines = []                                                                                                                         
    with open(embeddings, 'r') as f: 
      lines = f.readlines()
    strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
    
    # build dict to look up embeddings
    self.word_vectors = dict()
    for strip in strip_list: 
      self.word_vectors[strip[0]] = np.array(strip[1:]).astype(float)

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    D = self.word_vectors.get('the').shape[0] 
    X = np.zeros((len(data), D))
    n = 0
    emptycount = 0
    for tweet in data:
      tokens = [t for t in tweet.strip().split()]
      sum = np.zeros(D)
      m = 0
      for word in tokens:
        vec = self.word_vectors.get(word)
        if vec is not None:
            sum = np.add(sum, vec)
            m += 1
      if m > 0:
        X[n] = 1/m * sum
      else:
        emptycount += 1
      n += 1
    print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)
