## Dataset

Download the tweet dataset:
```
wget http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip

unzip twitter-datasets.zip

mkdir Data

mv twitter-datasets/* Data
```
The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

## General Workflow

Download and store pretrained models from huggingface (run this on login node!):
```
python3 preloading.py
```
Apply preprocessing scripts to raw data to build files of preprocessed data (on compute node):
```
python3 preprocessing.py Data/train_pos.txt Data/train_pos_basic.txt -v

python3 preprocessing.py Data/train_neg.txt Data/train_neg_basic.txt -v
```
Train on preprocessed data and save trained model (compute node):
```
python3 training.py Data/train_neg_basic.txt Data/train_pos_basic.txt Models/bart-base_basic -pm Pretrained_Models/bart-base/ -v

python3 training.py Data/train_neg_basic.txt Data/train_pos_basic.txt Models/bert-base-uncased_basic -pm Pretrained_Models/bart-base-uncased/ -v

python3 training.py Data/train_neg_basic.txt Data/train_pos_basic.txt Models/xlnet-base-cased_basic -pm Pretrained_Models/xlnet-base-cased/ -v -x
```
Create predictions for test-data (compute node):
```
python inference.py ....
```

## Virtual Environment & Dependencies

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

## Leonhard Cluster

Load modules:
```
module load gcc/6.3.0 python_gpu/3.8.5
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
bbjobs
```

## IMDB Dataset on Leonhard
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

mv aclImdb $SCRATCH/

tar -xf aclImdb_v1.tar.gz
```
