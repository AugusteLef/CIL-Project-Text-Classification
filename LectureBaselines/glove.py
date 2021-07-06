# This script contains the pipeline for building GloVe embeddings, given (preprocessed) tweets. Many parts of 
# this script are taken from the templates and master solutions of the CIL-course (e.g. solution to exercise 8).
# This script uses GNU-style sed. For MacOS, follow this thread: https://gist.github.com/andre3k1/e3a1a7133fded5de5a9ee99c87c6fa0d.

# imports
import os
import pickle
import random
import numpy as np
from scipy.sparse import coo_matrix

# different datasets for which GloVe embeddings should be computed. Each dataset consists of a name, two paths to data-files and parameters used in the script (see examples).
DATASETS = [
    {
        'name' : 'small', 
        'neg_data' : '../Data/train_neg.txt', 
        'pos_data' :'../Data/train_pos.txt',
        'embedding_dimension' : 20,
        'nmax' : 100,
        'eta' : 0.001,
        'alpha' : 3/4,
        'epochs' : 2
    }
    ]

# execute full pipeline for each dataset
for dataset in DATASETS:
    print('\n Processing: ' + dataset['name'] + '...')

    # setting random seeds (reproducibility)
    np.random.seed(1)
    random.seed(1)
    
    # build vocabulary
    os.system('cat ' + dataset['neg_data'] + ' ' + dataset['pos_data'] + ' | sed "s/ /\\n/g" | grep -v "^\s*$" | sort | uniq -c > Vocabs/' + dataset['name'] + '.txt')
    
    # cut vocabulary
    os.system('cat Vocabs/' + dataset['name'] + '.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d" " -f2 > Vocabs/' + dataset['name'] + '_cut.txt')
    
    # create vocab dict
    vocab = dict()
    with open('Vocabs/' + dataset['name'] + '_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    with open('Vocabs/' + dataset['name'] + '.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    vocab_size = len(vocab)

    # creating co-occurence matrix (two words co-occur if they appear together in a tweet)
    print('Building Co-Occurence matrix...')
    data, row, col = [], [], []
    counter = 1
    for fn in [dataset['neg_data'], dataset['pos_data']]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    cooc.sum_duplicates()
    with open('Vocabs/' + dataset['name'] + '_cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

    # learning embeddings
    print("Lerning embeddings for " + dataset['name'] + " using nmax =", dataset['nmax'], "while cooc.max() =", cooc.max())
    print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
    xs = np.random.normal(size=(cooc.shape[0], dataset['embedding_dimension']))
    ys = np.random.normal(size=(cooc.shape[1], dataset['embedding_dimension']))
    for epoch in range(dataset['epochs']):
        print('Epoch: ' + str(epoch + 1))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / dataset['nmax']) ** dataset['alpha'])
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * dataset['eta'] * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.savez('Embeddings/' + dataset['name'], x=xs, y=ys)