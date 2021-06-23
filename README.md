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

## Instructions

Start virtual environment:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Get Imdb dataset:
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
```

Move Imdb dataset to data directory (you will have to adjust DIR_DATA param in main.py if not $SCRATCH):
```bash
mv aclImdb $SCRATCH/
```

Preload preprocessed parameters:
```bash
python3 preloading.py
```

### Instructions specific for leonhard

Load modules:
```bash
module load gcc/6.3.0 python_gpu/3.8.5
```

Submit job:
```bash
bsub -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]" -o output python3 main.py
```
