# Largely taken from the template provided in the course: https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_2.ipynb
# https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html

from numpy.testing._private.utils import tempdir
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle

class TweetVectorizer:
  def __init__(self, model):
    print("Loading in word vectors...")
    self.word_vectors = model
    print("Finished loading in word vectors")

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

# Reproducibility!
np.random.seed(1) 

# load data
tweets = []
labels = []

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)
    
load_tweets('../Data/train_neg_full_basic.txt', 0)
load_tweets('../Data/train_pos_full_basic.txt', 1)

print(f'{len(tweets)} tweets loaded')

# Convert to NumPy array to facilitate indexing
tweets = np.array(tweets)
labels = np.array(labels)

# create train / validation split
shuffled_indices = np.random.permutation(len(tweets))
split_idx = int(0.8 * len(tweets))
train_indices = shuffled_indices[:split_idx]
val_indices = shuffled_indices[split_idx:]

# Average embeddings for each tweet to create features
from sklearn.feature_extraction.text import CountVectorizer

# Load GloVe embeddings
xs = np.load('embeddings.npz')['x']
ys = np.load('embeddings.npz')['y']
with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
# build word to embedding dict
embeddings = {}
for word in vocab:
    try:
        embeddings[word] = np.concatenate((xs[vocab.get(word)], ys[vocab.get(word)]), axis=0)
    except:
        pass
print(embeddings.get('the'))

# Use our Vectorizer with the previously trained GloVe embeddings
vectorizer = TweetVectorizer(embeddings)

# Important: we call fit_transform on the training set, and only transform on the validation set
X_train = vectorizer.fit_transform(tweets[train_indices])
X_val = vectorizer.transform(tweets[val_indices])

Y_train = labels[train_indices]
Y_val = labels[val_indices]

# train linear model
print('starting training')
model = LogisticRegression(C=1e5, max_iter=100)
model.fit(X_train, Y_train)

# evaluation (accuracy)
Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)
train_accuracy = (Y_train_pred == Y_train).mean()
val_accuracy = (Y_val_pred == Y_val).mean()
print(f'Accuracy (training set): {train_accuracy:.05f}')
print(f'Accuracy (validation set): {val_accuracy:.05f}')

# make predictions on the test set
tweets = []
labels = []
load_tweets('../Data/test_data.txt', 0) # slight abuse of load_tweets() here, 0 is just a placeholder
X_test = vectorizer.transform(tweets)
predictions = model.predict(X_test)
predictions = np.where(predictions <= 0, -1, predictions) # sets zeroes to -1

# save predictions for Kaggle submission
df = pd.DataFrame(predictions, columns=['Prediction'], index=list(range(1, len(predictions)+1)))
df.to_csv("../Predictions/glove.csv", index_label ='Id', header = True)