## File Structure
Some scripts may assume the following file-structure (you might have to create missing directories):
- Data : Directory containing all training-, test- and preprocessed data
- Predictions : Directory containing all predictions for test-set.
- Preprocessing_Data : Directory containing wordlists and other data used for preprocessing.
- LectureBaselines : Directory containing implementations of the baselines from the exercises. A separate ReadMe can be found here with instructions on how to reproduce the baselines.
- Output_pp : Directory for all output (logging) files related to preprocessing
- Output_predicting : Directory for all output (logging) files related to inference on the test-set.
- Output_huggingface : Directory for all output (logging) files related to training the huggingface models.
- Output_ensembles : Directory for all output (logging) files related to training the ensembles.

The following python scripts should be contained in the main project folder:
- preprocessing.py : Used for preprocessing data-sets with different preprocessing methods.
- training_*.py : Training scripts for huggingface models and our ensemble models.
- inference_*.py : Scripts used to create predictions for the test set.
- utils_*.py : Contains some useful code-snippets used in the above scripts.

All our experiments can be reproduced by running the following files with 'source' on the Leonhard cluster (more details under 'Workflow' below):
- preprocess_all : Schedules all preprocessing jobs needed in this project. These jobs roughly take up to 1h to finish.
- train_huggingface : Schedules all jobs for training huggingface models. These jobs roughly take 24h to finish.
- train_ensembles : Schedules all jobs for training ensembles. These jobs roughly take 24h to finish.
- predict_all : Schedules all jobs for making predictions on the test-set. These jobs don't take more than a few minutes to finish.

## Dataset

TODO: Add polybox link that leads to preprocessed datasets we use in experiments.

Download the tweet dataset:
```
wget http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip
```
Move it to the Data directory:
```
unzip twitter-datasets.zip

mkdir Data

mv twitter-datasets/* Data
```
The dataset should contain the following files:
- sample_submission.csv
- train_neg.txt : a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

## Additional Datasets
Download the tweet datasets:
```
wget [download link]
```

Match the format of the original dataset and create a positive and negative datasets (save them in Data/):
- For the dataset from [Sentiment140](https://www.kaggle.com/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv)
```
python3 additional_dataset_1.py dataset1.csv

```
- For the dataset from [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/data?select=train.csv)

```
python3 additional_dataset_2.py dataset2.csv

```
Combine all datasets:
- for negative:
```
python3 combine_datasets.py Data/train_neg_full.txt Data/train_neg_add1.txt Data/train_neg_add2.txt train_neg_all_full.txt

```
- for positive:
```
python3 combine_datasets.py Data/train_pos_full.txt Data/train_pos_add1.txt Data/train_pos_add2.txt train_pos_all_full.txt

```

## Running Our Experiments

Guidelines for running our experiments are presented here. We assume that the git-directory has been cloned on the Leonhard cluster, that the correct file structure has been set up (i.e. adding missing directories according to description above) and that the datasets have been downloaded and put in the Data directory. Our experiments will store the models in your personal scratch space ($SCRATCH). 

### Further Preparations

Load necessary modules:
```
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
```
Create and start virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install dependencies (make sure to be in venv):
```bash
pip install -r requirements.txt
```

### Preprocessing
You can schedule all 10 preprocessing jobs by running:
```
source preprocess_all
```

### Training
First, the huggingface models have to be trained:
```
source train_huggingface
```
Once they are done, we can train the ensembles:
```
source train_ensembles
```

### Predictions
As a last step, we create the predictions using the trained models:
```
source predict_all
```

## Miscellaneous
I never manage to remember all of it, hence I put here a list of handy commands.

### Virtual Environment & Dependencies

Start virtual environment:
```bash
source venv/bin/activate
```

Exit virtual environment:
```
deactivate
```
Update requirements.txt:
```
pip list --format=freeze > requirements.txt
```
Install dependencies (make sure to be in venv):
```bash
pip install -r requirements.txt
```

### Leonhard Cluster

Load modules:
```
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
```
Reset modules:
```
module purge
module load StdEnv
```
Submitting job:
```
bsub -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -oo output python3 main.py [args]
```
Submitting as interactive job for testing (output to terminal):
```
bsub -I -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" python3 main.py [args]
```
Monitoring job:
```
bjobs
bbjobs
```