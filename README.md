## File Structure
Some scripts may assume the following file-structure:
- Data : Directory containing all training-, test- and preprocessed data
- Models : Directory containing all fine-tuned models that we use for prediction.
- Predictions : Directory containing all predictions for test-set.
- Preprocessing_Data : Directory containing wordlists and other data used for preprocessing.
- Pretrained_Models : Directory containing pretrained models downloaded form huggingface.
- Vocab : Directory containing vocab artefacts.

The following scripts should be contained in the main project folder:
- preloading.py : Downloads models from huggingface and data for nltk library.
- preprocessing.py : Used for preprocessing data-sets with different preprocessing methods.
- training.py : Used to fine-tune a pretrained model on preprocessed datasets.
- inference.py : Used to create predictions for test-set using a fine-tuned model.
- utils.py : Contains some useful code-snippets used in the above scripts.


## Dataset

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
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

## General Workflow

Download and store pretrained models from huggingface & nltk data (run this on login node!):
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
bbjobs
```

## IMDB Dataset on Leonhard
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

mv aclImdb $SCRATCH/

tar -xf aclImdb_v1.tar.gz
```
