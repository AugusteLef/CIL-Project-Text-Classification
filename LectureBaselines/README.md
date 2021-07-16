This folder contains the code used to create the baselines that are based on what we have seen in the lecture (and the exercises and tutorials). Note that in contrast to other code we have written, the scripts in this folder do not take any command-line arguments which means that the source code has to be changed if you want perform other experiments than the ones we intended.

## File Structure

The intended structure of this directory is as follows:
- vectorizers.py contains some custom vectorizers that are used in our experiments
- predictions.py contains the code for our experiments. The script trains different models and also evaluates them and creates predictions for the test-data.
- Embeddings is a directory containing GloVe embeddings. We work with existing GloVe embeddings from here: https://nlp.stanford.edu/projects/glove/
- glove_preprocessing.rb is a preprocessing ruby - script from https://nlp.stanford.edu/projects/glove/.

## Workflow
In order to run our experiments for the baselines, first make sure that the necessary (preprocessed) datafiles are present in ../Data/. Also make sure to install the ../requirements.txt. Once that is done, the workflow is as follows:
- download and unpack GloVe embeddings in the 'Embedding' directory: 
```
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```
- run predictions.py. This will run several pre-defined experiments, each combines a vectorizer and a classification model to create predictions which can then be uploaded to Kaggle. The predictions are placed in ../Predictions. The script also outputs training and validation accuracy for each experiments.



