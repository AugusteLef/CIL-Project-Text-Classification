## Lecture Baselines

This folder contains the code used to create the baselines that we have seen in the lecture (i.e. in the master solutions or the tutorials). There are two such baselines: 
- The first one uses a simple count vectorizer and a logistic classifier. It is fully implemented in the file 'count_vectorizer_linear_model.py'. This script reads the training data, uses the #appearances of the 5000 most frequent words as features and trains a classifier. It then reads the test data and creates predictions in the format accepted by the Kaggle competition.
- The second one uses the GloVe embeddings. Here, the pipeline is more complicated:
    - build_vocab.sh creates a list of all words appearing in the training data.
    - cut_vocab.sh removes low-frequency words from that list.
    - pickle_vocab.py dumps this filtered list in vocab.pkl.
    - cooc.py is then used to create the co-occurence matrix cooc.pkl.
    - glove_embeddings.py creates the embeddings from the co-occurence matrix
    - glove_predictions.py finally trains a logistic classifier using the as features the average over word embeddings in one tweet
