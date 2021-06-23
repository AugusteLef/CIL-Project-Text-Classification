## Twitter  Datasets

Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip

The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

## Build the Co-occurence Matrix

To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py

## General Workflow

- apply preprocessing scripts to raw data to build files of preprocessed data
- apply training scripts to preprocessed data, save trained model
- run inference script using a trained model

## Virtual Environment & Dependencies

Start virtual environment:
```bash
source venv/bin/activate
```

- exit virtual environment
deactivate

- list dependencies (update requirements.txt):
pip list --format=freeze > requirements.txt

- install dependencies (make sure to be in venv):
```bash
pip install -r requirements.txt
```

## Leonhard Cluster

- load modules:
module load gcc/6.3.0 python_gpu/3.8.5

- reset modules
module purge
module load StdEnv

- preloading model:
python3 preloading.py
```

### Instructions specific for leonhard

- submitting job:
bsub -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -oo output python3 main.py [args]

- submitting as interactive job for testing (output to terminal):
bsub -I -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" python3 main.py [args]

- monitoring job
bbjobs

## IMDB Dataset on Leonhard
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
mv aclImdb $SCRATCH/
tar -xf aclImdb_v1.tar.gz
