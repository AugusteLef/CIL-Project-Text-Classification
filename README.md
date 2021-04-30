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
- apply main (model) scripts to preprocessed data, save trained model
- run prediction script using a trained model

## Virtual Environment & Dependencies

- start virtual environment:
source venv/bin/activate

- exit virtual environment
deactivate

- list dependencies (update requirements.txt):
pip list --format=freeze > requirements.txt

- install dependencies (make sure to be in venv):
pip install -r requirements.txt

## Leonhard Cluster

- load modules:
module load gcc/6.3.0 python_gpu/3.8.5

- reset modules
module purge
module load StdEnv

- preloading model:
python3 preloading.py

- submitting job:
bsub -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -oo output python3 main.py [args]

- submitting as interactive job for testing (output to terminal):
bsub -I -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" python3 main.py [args]

- monitoring job
bbjobs

## IMDB Dataset on Leonhard
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
mv aclImdb $SCRATCH/
tar -xf aclImdb_v1.tar.gz