# This script executes multiple experiments based on the approaches shown in the lecture. In particular,
# different Vectorizers and Models can be combined and trained on different datasets here. For GloVe
# experiments, the GloVe embeddings first have to be computed with the glove.py script. Our custom Vectorizers
# can be found in the vectorizer.py file. This file is heavily based on the notebook for the introduction of project 2
# in the CIL lecture: https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_2.ipynb.

# imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import vectorizers
import numpy as np
import random
import pandas as pd

# List of experiments to be conducted. Each experiment consists of a name, a model, a vectorizer, and the data sets.
EXPERIMENTS = [
    {
        'name' : 'count_vectorizer', 
        'neg_data' : '../Data/train_neg.txt', 
        'pos_data' : '../Data/train_pos.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : CountVectorizer(max_features=5000)
    },
    {
        'name' : 'tf_idf_vectorizer', 
        'neg_data' : '../Data/train_neg.txt', 
        'pos_data' : '../Data/train_pos.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : TfidfVectorizer(max_features=5000)
    }, 
    {
        'name' : 'count_vectorizer_full_25d', 
        'neg_data' : '../Data/train_neg_full.txt', 
        'pos_data' : '../Data/train_pos_full.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : CountVectorizer(max_features=5000)
    },
    {
        'name' : 'glove', 
        'neg_data' : '../Data/train_neg.txt', 
        'pos_data' :'../Data/train_pos.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.25d.txt')
    },
    {
        'name' : 'glove_full_25d', 
        'neg_data' : '../Data/train_neg_full.txt', 
        'pos_data' :'../Data/train_pos_full.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.25d.txt')
    },
    {
        'name' : 'glove_full_50d', 
        'neg_data' : '../Data/train_neg_full.txt', 
        'pos_data' :'../Data/train_pos_full.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.50d.txt')
    },
    {
        'name' : 'glove_full_100d', 
        'neg_data' : '../Data/train_neg_full.txt', 
        'pos_data' :'../Data/train_pos_full.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.100d.txt')
    },
    {
        'name' : 'glove_full_200d', 
        'neg_data' : '../Data/train_neg_full.txt', 
        'pos_data' :'../Data/train_pos_full.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.200d.txt')
    },
    {
        'name' : 'glove_full_200d_preprocessed', 
        'neg_data' : '../Data/train_neg_full_glove.txt', 
        'pos_data' :'../Data/train_pos_full_glove.txt',
        'test_data' : '../Data/test_data.txt',
        'model' : LogisticRegression(C=1e5, max_iter=500),
        'vectorizer' : vectorizers.AverageGlove('Embeddings/glove.twitter.27B.200d.txt')
    }
]
    
# Custom function for loading data here
def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)

# execute full pipeline for each experiment
for experiment in EXPERIMENTS:
    print('\n Starting experiment : ' + experiment['name'] + '...')

    # setting random seeds (reproducibility)
    np.random.seed(1)
    random.seed(1)

    # load data
    tweets = []
    labels = []
    load_tweets(experiment['neg_data'], 0)
    load_tweets(experiment['pos_data'], 1)
    print(f'{len(tweets)} tweets loaded')
    tweets = np.array(tweets)
    labels = np.array(labels)

    # create train / validation split
    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.8 * len(tweets)) # 80 / 20 split
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    # Vectorization
    X_train = experiment['vectorizer'].fit_transform(tweets[train_indices])
    X_val = experiment['vectorizer'].transform(tweets[val_indices])
    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # training model
    print('starting training...')
    experiment['model'].fit(X_train, Y_train)

    # evaluation (accuracy)
    Y_train_pred = experiment['model'].predict(X_train)
    Y_val_pred = experiment['model'].predict(X_val)
    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()
    print('Experiment : ' + experiment['name'] + f', training accuracy: {train_accuracy:.05f}')
    print('Experiment : ' + experiment['name'] + f', validation accuracy: {val_accuracy:.05f}')

    # make predictions on the test set
    tweets = []
    labels = []
    load_tweets(experiment['test_data'], 0) # slight abuse of load_tweets() here, 0 is just a placeholder
    X_test = experiment['vectorizer'].transform(tweets)
    predictions = experiment['model'].predict(X_test)
    predictions = np.where(predictions <= 0, -1, predictions) # sets zeroes to -1

    # save predictions for Kaggle
    df = pd.DataFrame(predictions, columns=['Prediction'], index=list(range(1, len(predictions)+1)))
    df.to_csv("../Predictions/" + experiment['name'] + ".csv", index_label ='Id', header = True)
