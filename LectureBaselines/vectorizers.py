# This file contains custom vectorizers that can be used in the experiments in predictions.py.

# imports
import numpy as np
import pickle

# average word embeddings in each tweet, adapte from
# https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html
class AverageGlove:
  def __init__(self, embeddings, vocab):
    ''' 
    Args: 
      embeddings : path to glove embeddings
      vocab : path to pickled vocab
    '''
    # load embeddings
    xs = np.load(embeddings)['x']
    ys = np.load(embeddings)['y'] 
    # load vocab
    with open(vocab, 'rb') as f:
        vocab = pickle.load(f)
    # build word to embedding dict
    self.word_vectors = {}
    for word in vocab:
        self.word_vectors[word] = np.concatenate((xs[vocab.get(word)], ys[vocab.get(word)]), axis=0)

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
