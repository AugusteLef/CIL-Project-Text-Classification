This folder contains the code used to create the baselines that are based on what we have seen in the lecture (and the exercises and tutorials). Note that in contrast to other code we have written, the scripts in this folder do not take any command-line arguments which means that the source code has to be changed if you want perform other experiments than the ones we intended.

## File Structue

The scripts in this folder might assume the following file structure:
- Embeddings : Directory containing GloVe embeddings
- Vocabs : Directory containing vocabulary files

## Workflow

In order to run our experiments for the baselines, first make sure that the necessary (preprocessed) datafiles are present in ../Data/. Also make sure to install the ../requirements.txt. Once that is done, the workflow is as follows:
- run glove.py to create GloVe embeddings. This script runs multiple pre-defined experiments, each constructing GloVe embeddings a bit differently. The embeddings are placed in Embeddings/.
- run predictions.py. This will run several pre-defined experiments, each combines a vectorizer and a classification model to create predictions which can then be uploaded to Kaggle. The predictions are placed in ../Predictions. The script also outputs training and validation accuracy for each experiment.

Download and unpack GloVe embeddings: 
```
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
```

